"""BERT MLM pretraining on Trainium — hook overhead isolation test.

Based on bert-pretrain-neuron-validation.py with an added --debug-perforated flag
that registers backward hooks on linear layer outputs during n-mode training,
mimicking what PAINeuronModule does in perforatedbp without any actual PAI logic.

The goal is to determine whether the presence of Python backward hooks alone
(even no-op ones) causes the Trainium runtime to fall back to CPU and slow down.

Usage
-----
Baseline (no hooks, identical to bert-pretrain-neuron-validation.py):
    python bert-pretrain-neuron-validation-testing.py --steps 50

With simulated PAI n-mode backward hooks on all Linear layers:
    python bert-pretrain-neuron-validation-testing.py --steps 50 --debug-perforated

Compare the average step times printed at the end.
"""

import argparse
import os
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        device = torch.device("neuron:0")
        return 0, 1, device

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.neuron.set_device(rank)
    device = torch.device(f"neuron:{rank}")
    return rank, world_size, device


def build_synthetic_corpus(num_docs: int = 2048):
    topics = [
        "trainium accelerates transformer training with native pytorch",
        "the neuron sdk exposes a privateuse1 backend for eager execution",
        "masked language modeling predicts tokens hidden by the collator",
        "attention layers mix information across the sequence dimension",
        "distributed data parallel replicates the model across cores",
        "fully sharded data parallel shards parameters to save memory",
    ]
    return [topics[i % len(topics)] + f" example number {i}" for i in range(num_docs)]


def build_dataloader(args, tokenizer, rank, world_size, split="train"):
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    if args.dataset is None:
        all_texts = build_synthetic_corpus(num_docs=2048)
        if split == "validation":
            texts = all_texts[:256]
        else:
            texts = all_texts[256:]
    else:
        from datasets import load_dataset
        hf_split = "validation" if split == "validation" else "train"
        stream = load_dataset(
            args.dataset, args.dataset_config, split=hf_split, streaming=True
        )
        texts = []
        max_docs = args.max_val_docs if split == "validation" else args.max_docs
        for row in stream:
            line = row.get("text", "").strip()
            if line:
                texts.append(line)
            if len(texts) >= max_docs:
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
    if world_size > 1 and split == "train":
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        collate_fn=collator,
        drop_last=True,
    )


# ─── Debug hook machinery ────────────────────────────────────────────────────
#
# PAINeuronModule.forward() does this on every forward call:
#
#   if out.requires_grad:
#       out.register_hook(lambda grad: filter_backward(grad, self.dendrite_module.dendrite_values))
#
# During n-mode, filter_backward:
#   1. enters torch.no_grad()
#   2. calls grad.detach()
#   3. on first call: checks output dimensions, sets up arrays
#   4. calls MPB.filter_backward_pb(val, values)
#      → which just checks "mode == 'p'" and returns immediately in n-mode
#
# We replicate that cost without any PAI state.

def _make_nmode_backward_hook(layer_name, initialized_ref):
    """Return a backward hook that matches filter_backward's n-mode cost."""

    def hook(grad):
        with torch.no_grad():
            # Mirrors: val = grad_out.detach()
            val = grad.detach()

            # Mirrors: first-pass shape inspection in filter_backward
            if not initialized_ref[0]:
                _shape = val.shape          # read shape
                _ndim  = len(_shape)        # compute ndim
                initialized_ref[0] = True

            # Mirrors: MPB.filter_backward_pb — in n-mode the only check is:
            #   if GPA.pai_tracker.member_vars["mode"] == "p": ...
            # which is False in n-mode, so nothing happens.
            # We replicate the branch-not-taken cost with an equivalent compare.
            _mode = "n"
            if _mode == "p":
                pass  # never taken — mirrors the early-return in n-mode

        # Return None: gradient is passed through unchanged (same as filter_backward)
        return None

    return hook


def register_debug_hooks(model, rank):
    """Register PAI-style backward hooks on every Linear layer output.

    Uses a forward hook to attach the backward hook each step, exactly as
    PAINeuronModule.forward() does via out.register_hook(). Returns the list
    of forward-hook handles so they can be removed if needed.
    """
    handles = []
    layer_count = 0

    def make_fwd_hook(name):
        initialized_ref = [False]
        bk_hook = _make_nmode_backward_hook(name, initialized_ref)

        def fwd_hook(module, input, output):
            # Only attach when the output is a gradient-tracking tensor,
            # mirroring: if out.requires_grad: out.register_hook(...)
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(bk_hook)

        return fwd_hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_fwd_hook(name))
            handles.append(h)
            layer_count += 1

    if rank == 0:
        print(f"[debug-perforated] Registered backward hooks on {layer_count} Linear layers")

    return handles
# ─────────────────────────────────────────────────────────────────────────────


def build_model(args, device, parallel: str):
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.seq_len,
        attn_implementation="eager",
    )
    model = BertForMaskedLM(config)
    model = model.to(dtype=torch.bfloat16, device=device)

    if parallel == "ddp":
        model = DDP(model)
    elif parallel == "fsdp":
        from torch.distributed._composable.fsdp import fully_shard
        for layer in model.bert.encoder.layer:
            fully_shard(layer)
        fully_shard(model)

    if args.compile:
        model = torch.compile(model, backend="neuron")

    return model


@torch.no_grad()
def evaluate(model, val_loader, device, rank, world_size):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss, num_batches], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / loss_tensor[1].item()

    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    model.train()

    return avg_loss, perplexity


def train(args):
    rank, world_size, device = setup_distributed()
    parallel = args.parallel if world_size > 1 else "none"

    if rank == 0:
        mode = "compile" if args.compile else "eager"
        hook_mode = "WITH debug-perforated hooks" if args.debug_perforated else "baseline (no hooks)"
        print(
            f"World size: {world_size} | Device: {device} | "
            f"Parallel: {parallel} | Mode: {mode}"
        )
        print(
            f"Model: BERT L={args.num_layers} H={args.hidden_size} "
            f"heads={args.num_heads} vocab={args.vocab_size} seq={args.seq_len}"
        )
        print(f"Hook mode: {hook_mode}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size

    train_loader = build_dataloader(args, tokenizer, rank, world_size, split="train")
    val_loader = build_dataloader(args, tokenizer, rank, world_size, split="validation")

    model = build_model(args, device, parallel)
    model.train()

    # Register debug hooks AFTER the model is built and moved to device,
    # matching the point in perforatedbp where perforate_model is called.
    if args.debug_perforated:
        _debug_hook_handles = register_debug_hooks(model, rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    done = False
    training_start = time.time()
    first_step_time = None
    step_times = []
    best_val_loss = float('inf')
    best_step = 0
    patience_counter = 0

    if rank == 0:
        print(f"\nValidation will run every {args.eval_every} steps")
        if args.patience > 0:
            print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
        print()

    for epoch in range(args.epochs):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for batch in train_loader:
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

            if rank == 0 and step % args.log_every == 0:
                avg_step_time = sum(step_times) / len(step_times) if step_times else step_time
                print(f"Step {step}/{args.steps} - Train loss: {loss.item():.4f} - "
                      f"Step time: {step_time:.3f}s (avg: {avg_step_time:.3f}s)")

            if step % args.eval_every == 0:
                eval_start = time.time()
                val_loss, val_ppl = evaluate(model, val_loader, device, rank, world_size)
                eval_time = time.time() - eval_start

                improved = val_loss < (best_val_loss - args.min_delta)

                if rank == 0:
                    print(f"{'='*60}")
                    print(f"Validation @ step {step}:")
                    print(f"  Val loss: {val_loss:.4f}")
                    print(f"  Val perplexity: {val_ppl:.2f}")
                    print(f"  Eval time: {eval_time:.2f}s")

                    if improved:
                        best_val_loss = val_loss
                        best_step = step
                        patience_counter = 0
                        print(f"  ** New best validation loss! **")
                    else:
                        patience_counter += 1
                        print(f"  No improvement (patience: {patience_counter}/{args.patience})")

                    print(f"  Best val loss: {best_val_loss:.4f} @ step {best_step}")
                    print(f"{'='*60}\n")

                if args.patience > 0 and patience_counter >= args.patience:
                    if rank == 0:
                        print(f"\nEarly stopping triggered after {patience_counter} evaluations without improvement.")
                        print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
                    done = True
                    break

            if step >= args.steps:
                done = True
                break
        if done:
            break

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("\nRunning final validation...")
    final_val_loss, final_val_ppl = evaluate(model, val_loader, device, rank, world_size)

    total_time = time.time() - training_start
    if rank == 0:
        hook_mode = "WITH debug-perforated hooks" if args.debug_perforated else "baseline (no hooks)"
        print("\n" + "="*60)
        print(f"BERT MLM pretraining loop complete  [{hook_mode}]")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Total steps: {step}")
        if first_step_time:
            print(f"First step time (TTFI): {first_step_time:.2f}s")
        if step_times:
            avg_step = sum(step_times) / len(step_times)
            print(f"Average step time (excluding first): {avg_step:.3f}s")
            print(f"Throughput: {args.batch_size / avg_step:.1f} samples/sec")
            total_samples = step * args.batch_size
            print(f"Total samples processed: {total_samples:,}")
        print(f"\nFinal validation metrics:")
        print(f"  Val loss: {final_val_loss:.4f}")
        print(f"  Val perplexity: {final_val_ppl:.2f}")
        print(f"  Best val loss: {best_val_loss:.4f} @ step {best_step}")
        if args.patience > 0:
            stopped_early = step < args.steps
            print(f"  Early stopping: {'Yes' if stopped_early else 'No (reached max steps)'}")
        print("="*60)

    if args.save_dir and rank == 0:
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved checkpoint to {args.save_dir}")

    if world_size > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="BERT MLM hook overhead isolation test")
    # Model
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=30522)
    p.add_argument("--seq-len", type=int, default=128)
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument(
        "--compile",
        action="store_true",
        help="wrap model in torch.compile(backend='neuron')",
    )
    # Validation
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.01)
    # Data
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--max-docs", type=int, default=8000)
    p.add_argument("--max-val-docs", type=int, default=1000)
    # Checkpoint
    p.add_argument("--save-dir", type=str, default=None)
    # Debug flag
    p.add_argument(
        "--debug-perforated",
        action="store_true",
        help=(
            "Register PAI-style backward hooks on all Linear layer outputs, "
            "mimicking what PAINeuronModule does during n-mode training. "
            "Compares step time vs baseline to isolate hook overhead."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
