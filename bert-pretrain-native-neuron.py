"""BERT masked-language-model (MLM) pretraining on Trainium via native torch-neuronx.

Native PyTorch backend (PrivateUse1), NOT torch-xla. Runs eager mode on
`torch.device('neuron')`. Supports single-core, multi-core DDP, and multi-core
FSDP2. Uses HuggingFace `BertForMaskedLM` and the standard MLM masking collator.

Data: defaults to a synthetic in-memory corpus so the script runs with zero
downloads. Pass --dataset to stream a real HF dataset (e.g. wikitext) instead.

Environment (torch-neuronx native beta, see .kiro/steering/beta-setup-log.md):
    ssh trn2 && cd ~/workspace && source native_venv/bin/activate

Usage
-----
Single core:
    python bert-pretrain-native-neuron.py --steps 50

Multi-core DDP (trn2.3xlarge: 4 physical cores, LNC2 => 2 logical cores):
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
    torchrun --nproc_per_node=2 --rdzv_backend=c10d \
             --rdzv_endpoint=localhost:29500 \
             bert-pretrain-native-neuron.py --steps 50 --parallel ddp

Multi-core FSDP2:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
    torchrun --nproc_per_node=2 --rdzv_backend=c10d \
             --rdzv_endpoint=localhost:29500 \
             bert-pretrain-native-neuron.py --steps 50 --parallel fsdp

torch.compile for production throughput (any parallel mode):
    python bert-pretrain-native-neuron.py --steps 50 --compile

    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
    torchrun --nproc_per_node=2 --rdzv_backend=c10d \
             --rdzv_endpoint=localhost:29500 \
             bert-pretrain-native-neuron.py --steps 50 --parallel fsdp --compile

Real dataset (streamed, no full download):
    python bert-pretrain-native-neuron.py --dataset wikitext \
           --dataset-config wikitext-103-raw-v1 --steps 200
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
)


def is_distributed() -> bool:
    """torchrun sets RANK/WORLD_SIZE; single-core runs do not."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    """Init the Neuron process group and bind this rank to a logical core.

    Returns (rank, world_size, device). For single-core runs this returns
    (0, 1, neuron:0) without initializing a process group.
    """
    if not is_distributed():
        device = torch.device("neuron:0")
        return 0, 1, device

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # torch.neuron.set_device binds this process to a specific NeuronCore so
    # each rank owns one logical core. Mirrors the krai multi-core examples.
    torch.neuron.set_device(rank)
    device = torch.device(f"neuron:{rank}")
    return rank, world_size, device


def build_synthetic_corpus(num_docs: int = 2048):
    """A tiny deterministic corpus so the script runs with no downloads.

    Real pretraining wants billions of tokens; this exists to validate the
    training loop end-to-end on-device, not to produce a useful model.
    """
    topics = [
        "trainium accelerates transformer training with native pytorch",
        "the neuron sdk exposes a privateuse1 backend for eager execution",
        "masked language modeling predicts tokens hidden by the collator",
        "attention layers mix information across the sequence dimension",
        "distributed data parallel replicates the model across cores",
        "fully sharded data parallel shards parameters to save memory",
    ]
    return [topics[i % len(topics)] + f" example number {i}" for i in range(num_docs)]


def build_dataloader(args, tokenizer, rank, world_size):
    """Return an iterable dataloader yielding MLM-masked batches.

    Uses DataCollatorForLanguageModeling(mlm=True) to build the 15% masked
    inputs and -100 ignore labels on the fly, matching standard BERT MLM.
    """
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    if args.dataset is None:
        texts = build_synthetic_corpus()
    else:
        # Streamed to avoid a full local download; materialize a slice.
        from datasets import load_dataset

        stream = load_dataset(
            args.dataset, args.dataset_config, split="train", streaming=True
        )
        texts = []
        for row in stream:
            line = row.get("text", "").strip()
            if line:
                texts.append(line)
            if len(texts) >= args.max_docs:
                break

    class TextDataset(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                texts[idx],
                truncation=True,
                max_length=args.seq_len,
                padding="max_length",
                return_tensors=None,
            )
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    dataset = TextDataset()

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
    )


def build_model(args, device, parallel: str):
    """Construct BertForMaskedLM from scratch (pretraining, not fine-tuning).

    A config is built explicitly so no pretrained weights are downloaded; this
    is a from-scratch MLM pretraining setup.
    """
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.seq_len,
        # eager attention: the fused/sdpa paths are not the native-beta default
        attn_implementation="eager",
    )
    model = BertForMaskedLM(config)
    model = model.to(dtype=torch.bfloat16, device=device)

    if parallel == "ddp":
        model = DDP(model)
    elif parallel == "fsdp":
        from torch.distributed._composable.fsdp import fully_shard

        # Shard each encoder layer, then the whole model (FSDP2 wrapping order).
        for layer in model.bert.encoder.layer:
            fully_shard(layer)
        fully_shard(model)

    if args.compile:
        # torch.compile(backend="neuron") lowers the graph through the
        # torch-neuronx MLIR backend. Applied AFTER the parallel wrap so DDP
        # all-reduce / FSDP collectives are captured in the compiled graph.
        # Expect a long first-step TTFI while the NEFF compiles; use eager
        # (--compile off) for fast iteration and compile for throughput.
        model = torch.compile(model, backend="neuron")

    return model


def train(args):
    rank, world_size, device = setup_distributed()
    parallel = args.parallel if world_size > 1 else "none"

    if rank == 0:
        mode = "compile" if args.compile else "eager"
        print(
            f"World size: {world_size} | Device: {device} | "
            f"Parallel: {parallel} | Mode: {mode}"
        )
        print(
            f"Model: BERT L={args.num_layers} H={args.hidden_size} "
            f"heads={args.num_heads} vocab={args.vocab_size} seq={args.seq_len}"
        )

    # Fast WordPiece tokenizer; bert-base-uncased vocab is downloaded once.
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size

    dataloader = build_dataloader(args, tokenizer, rank, world_size)
    model = build_model(args, device, parallel)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    done = False
    for epoch in range(args.epochs):
        # Only DistributedSampler has set_epoch; the default RandomSampler does not.
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if rank == 0 and step % args.log_every == 0:
                # loss is a scalar; .item() forces a device sync (fine for logging).
                print(f"Step {step}/{args.steps} - MLM loss: {loss.item():.4f}")

            if step >= args.steps:
                done = True
                break
        if done:
            break

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        print("\nBERT MLM pretraining loop complete.")

    if args.save_dir and rank == 0:
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved checkpoint to {args.save_dir}")

    if world_size > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Native torch-neuronx BERT MLM pretraining")
    # Model
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=30522)  # overwritten by tokenizer
    p.add_argument("--seq-len", type=int, default=128)
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1000)  # step-count is the real stop
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument(
        "--compile",
        action="store_true",
        help="wrap model in torch.compile(backend='neuron') for production throughput",
    )
    # Data
    p.add_argument("--dataset", type=str, default=None, help="HF dataset name, e.g. wikitext")
    p.add_argument("--dataset-config", type=str, default=None, help="e.g. wikitext-103-raw-v1")
    p.add_argument("--max-docs", type=int, default=20000, help="cap streamed docs")
    # Checkpoint
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
