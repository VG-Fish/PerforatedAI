"""BERT masked-language-model (MLM) pretraining on local CUDA GPU.

Simplified single-GPU version for local development. Uses standard PyTorch CUDA
backend with HuggingFace `BertForMaskedLM` and the standard MLM masking collator.

Data: defaults to a synthetic in-memory corpus so the script runs with zero
downloads. Pass --dataset to stream a real HF dataset (e.g. Salesforce/wikitext) instead.

Usage
-----
Basic training:
    python bert-pretrain-cuda-neuron.py --steps 50

With torch.compile for better performance:
    python bert-pretrain-cuda-neuron.py --steps 50 --compile

Real dataset (streamed, no full download):
    python bert-pretrain-cuda-neuron.py --dataset Salesforce/wikitext \
           --dataset-config wikitext-103-raw-v1 --steps 200

Larger batch size:
    python bert-pretrain-cuda-neuron.py --steps 100 --batch-size 32 --compile
"""

import argparse
import time

import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
)


def build_synthetic_corpus(num_docs: int = 2048):
    """A tiny deterministic corpus so the script runs with no downloads.

    Real pretraining wants billions of tokens; this exists to validate the
    training loop end-to-end on-device, not to produce a useful model.
    """
    topics = [
        "cuda accelerates transformer training with optimized kernels",
        "pytorch provides a flexible deep learning framework",
        "masked language modeling predicts tokens hidden by the collator",
        "attention layers mix information across the sequence dimension",
        "gradient descent optimizes model parameters iteratively",
        "neural networks learn representations from data",
    ]
    return [topics[i % len(topics)] + f" example number {i}" for i in range(num_docs)]


def build_dataloader(args, tokenizer):
    """Return an iterable dataloader yielding MLM-masked batches.

    Uses DataCollatorForLanguageModeling(mlm=True) to build the 15% masked
    inputs and -100 ignore labels on the fly, matching standard BERT MLM.
    """
    from torch.utils.data import DataLoader, Dataset

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

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )


def build_model(args, device):
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
    )
    model = BertForMaskedLM(config)
    model = model.to(dtype=torch.bfloat16, device=device)

    if args.compile:
        # torch.compile with default backend (inductor) for optimized CUDA kernels.
        # Expect a compile time on the first step; use eager (--compile off) for
        # fast iteration and compile for throughput.
        model = torch.compile(model)

    return model


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        print("WARNING: CUDA not available, falling back to CPU (will be very slow)")
    
    mode = "compile" if args.compile else "eager"
    print(f"Device: {device} | Mode: {mode}")
    print(
        f"Model: BERT L={args.num_layers} H={args.hidden_size} "
        f"heads={args.num_heads} vocab={args.vocab_size} seq={args.seq_len}"
    )

    # Fast WordPiece tokenizer; bert-base-uncased vocab is downloaded once.
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size

    dataloader = build_dataloader(args, tokenizer)
    model = build_model(args, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    done = False
    training_start = time.time()
    first_step_time = None
    step_times = []
    
    for epoch in range(args.epochs):
        for batch in dataloader:
            step_start = time.time()
            
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

            step_time = time.time() - step_start
            step += 1
            
            if step == 1:
                first_step_time = step_time
            else:
                step_times.append(step_time)
            
            if step % args.log_every == 0:
                avg_step_time = sum(step_times) / len(step_times) if step_times else step_time
                print(f"Step {step}/{args.steps} - MLM loss: {loss.item():.4f} - "
                      f"Step time: {step_time:.3f}s (avg: {avg_step_time:.3f}s)")

            if step >= args.steps:
                done = True
                break
        if done:
            break
    
    total_time = time.time() - training_start
    print("\n" + "="*60)
    print("BERT MLM pretraining loop complete.")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
    if first_step_time:
        print(f"First step time (TTFI): {first_step_time:.2f}s")
    if step_times:
        avg_step = sum(step_times) / len(step_times)
        print(f"Average step time (excluding first): {avg_step:.3f}s")
        print(f"Throughput: {args.batch_size / avg_step:.1f} samples/sec")
    print("="*60)

    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved checkpoint to {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU CUDA BERT MLM pretraining")
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
    p.add_argument(
        "--compile",
        action="store_true",
        help="wrap model in torch.compile for production throughput",
    )
    # Data
    p.add_argument("--dataset", type=str, default=None, help="HF dataset name, e.g. Salesforce/wikitext")
    p.add_argument("--dataset-config", type=str, default=None, help="e.g. wikitext-103-raw-v1")
    p.add_argument("--max-docs", type=int, default=20000, help="cap streamed docs")
    # Checkpoint
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
