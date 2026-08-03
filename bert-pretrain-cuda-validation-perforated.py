"""BERT masked-language-model (MLM) pretraining on local CUDA GPU with validation and PerforatedAI.

Simplified single-GPU version for local development. Uses standard PyTorch CUDA
backend with HuggingFace `BertForMaskedLM` and the standard MLM masking collator.

Includes validation support with periodic evaluation, perplexity tracking,
early stopping, and PerforatedAI dendritic learning optimization.

Data: defaults to a synthetic in-memory corpus so the script runs with zero
downloads. Pass --dataset to stream a real HF dataset (e.g. Salesforce/wikitext) instead.

Usage
-----
Quick proof of concept (<2 hours with early stopping):
    python bert-pretrain-cuda-validation-perforated.py

With real dataset:
    python bert-pretrain-cuda-validation-perforated.py --dataset Salesforce/wikitext \
           --dataset-config wikitext-103-raw-v1

Disable early stopping (run to max steps):
    python bert-pretrain-cuda-validation-perforated.py --patience 0 --steps 5000

With torch.compile for better performance:
    python bert-pretrain-cuda-validation-perforated.py --compile

Larger batch size:
    python bert-pretrain-cuda-validation-perforated.py --batch-size 32 --compile
"""

import argparse
import time
import math

import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
)

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Allow BERT model classes in torch.load (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([BertForMaskedLM])


def setup_device():
    """Setup CUDA device or fallback to CPU.
    
    Returns device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return device
    else:
        print("WARNING: CUDA not available, falling back to CPU (will be very slow)")
        return torch.device("cpu")


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


def build_dataloader(args, tokenizer, split="train"):
    """Return an iterable dataloader yielding MLM-masked batches.

    Uses DataCollatorForLanguageModeling(mlm=True) to build the 15% masked
    inputs and -100 ignore labels on the fly, matching standard BERT MLM.
    
    Args:
        split: "train" or "validation" to control which data to use
    """
    from torch.utils.data import DataLoader, Dataset

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    if args.dataset is None:
        # Synthetic data: use different portions for train/val
        all_texts = build_synthetic_corpus(num_docs=2048)
        if split == "validation":
            texts = all_texts[:256]  # First 256 for validation
        else:
            texts = all_texts[256:]  # Rest for training
    else:
        # Real dataset: load the appropriate split
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

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        drop_last=True,
    )


def build_model(args, device):
    """Construct BertForMaskedLM from scratch (pretraining, not fine-tuning).

    A config is built explicitly so no pretrained weights are downloaded; this
    is a from-scratch MLM pretraining setup.
    """
    # Configure PerforatedAI
    GPA.pc.set_output_dimensions([-1, -1, 0])
    GPA.pc.set_module_names_to_track(["BertEncoder", "BertEmbeddings"])
    GPA.pc.set_module_ids_to_track([".cls.predictions.decoder"])
    GPA.pc.append_module_names_to_perforate(["BertPredictionHeadTransform"])
    GPA.pc.set_using_safe_tensors(False)
    GPA.pc.set_testing_dendrite_capacity(False)
    
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
    
    # Perforate the model with dendrites (minimizing loss)
    model = UPA.perforate_model(model, save_name="bert_mlm_dendritic_cuda", maximizing_score=False)
    model = model.to(dtype=torch.bfloat16, device=device)

    if args.compile:
        # torch.compile uses default "inductor" backend for CUDA optimization
        model = torch.compile(model)

    return model


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Run validation and compute metrics.
    
    Returns (avg_loss, perplexity) computed over the validation set.
    """
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
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    model.train()
    
    return avg_loss, perplexity


def train(args):
    device = setup_device()

    mode = "compile" if args.compile else "eager"
    print(f"Device: {device} | Mode: {mode}")
    print(
        f"Model: BERT L={args.num_layers} H={args.hidden_size} "
        f"heads={args.num_heads} vocab={args.vocab_size} seq={args.seq_len}"
    )

    # Fast WordPiece tokenizer; bert-base-uncased vocab is downloaded once.
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.vocab_size

    train_loader = build_dataloader(args, tokenizer, split="train")
    val_loader = build_dataloader(args, tokenizer, split="validation")
    
    model = build_model(args, device)
    model.train()

    # Setup optimizer with PerforatedAI
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(None)  # No scheduler
    optimArgs = {'params': model.parameters(), 'lr': args.lr}
    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, {})

    step = 0
    done = False
    training_start = time.time()
    first_step_time = None
    step_times = []
    best_val_loss = float('inf')
    best_step = 0
    patience_counter = 0
    
    # Track training loss for PAI extra score
    train_loss_accum = 0.0
    train_loss_count = 0
    
    print(f"\nValidation will run every {args.eval_every} steps")
    if args.patience > 0:
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    print()
    
    for epoch in range(args.epochs):
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
            
            # Accumulate training loss for PAI tracking
            train_loss_accum += loss.item()
            train_loss_count += 1

            step_time = time.time() - step_start
            step += 1
            
            if step == 1:
                first_step_time = step_time
            else:
                step_times.append(step_time)
            
            if step % args.log_every == 0:
                avg_step_time = sum(step_times) / len(step_times) if step_times else step_time
                print(f"Step {step}/{args.steps} - Train loss: {loss.item():.4f} - "
                      f"Step time: {step_time:.3f}s (avg: {avg_step_time:.3f}s)")
            
            # Run validation
            if step % args.eval_every == 0:
                eval_start = time.time()
                val_loss, val_ppl = evaluate(model, val_loader, device)
                eval_time = time.time() - eval_start
                
                # PerforatedAI: Track training loss as extra score (helps with optimization)
                avg_train_loss = train_loss_accum / train_loss_count if train_loss_count > 0 else 0.0
                GPA.pai_tracker.add_extra_score(avg_train_loss, "train")
                
                # Reset training loss accumulation for next validation period
                train_loss_accum = 0.0
                train_loss_count = 0
                
                # PerforatedAI: Track validation score and check for restructuring
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_loss, model)
                model = model.to(dtype=torch.bfloat16, device=device)
                
                if training_complete:
                    print("\nPerforatedAI training complete!")
                    done = True
                    break
                
                elif restructured and not training_complete:
                    print("\nModel restructured (dendrites added/incorporated)!")
                    # Reinitialize optimizer with same settings
                    optimArgs = {'params': model.parameters(), 'lr': args.lr}
                    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, {})
                
                # Check for improvement (early stopping logic)
                improved = val_loss < (best_val_loss - args.min_delta)
                
                print(f"{'='*60}")
                print(f"Validation @ step {step}:")
                print(f"  Val loss: {val_loss:.4f}")
                print(f"  Val perplexity: {val_ppl:.2f}")
                print(f"  Eval time: {eval_time:.2f}s")
                if restructured:
                    print(f"  ** Dendrites restructured! **")
                
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
        
        if done:
            break
    
    # Final validation
    print("\nRunning final validation...")
    final_val_loss, final_val_ppl = evaluate(model, val_loader, device)
    
    total_time = time.time() - training_start
    print("\n" + "="*60)
    print("BERT MLM pretraining loop complete.")
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

    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved checkpoint to {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="CUDA BERT MLM pretraining with validation and PerforatedAI")
    # Model
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=30522)  # overwritten by tokenizer
    p.add_argument("--seq-len", type=int, default=128)
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000, help="max training steps (early stopping may end sooner)")
    p.add_argument("--epochs", type=int, default=1000)  # step-count or early stopping is the real stop
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--compile",
        action="store_true",
        help="wrap model in torch.compile() for better throughput",
    )
    # Validation
    p.add_argument("--eval-every", type=int, default=100, help="run validation every N steps")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (0 to disable)")
    p.add_argument("--min-delta", type=float, default=0.01, help="minimum improvement for early stopping")
    # Data
    p.add_argument("--dataset", type=str, default=None, help="HF dataset name, e.g. Salesforce/wikitext")
    p.add_argument("--dataset-config", type=str, default=None, help="e.g. wikitext-103-raw-v1")
    p.add_argument("--max-docs", type=int, default=8000, help="cap streamed training docs (sized for <2hr runs)")
    p.add_argument("--max-val-docs", type=int, default=1000, help="cap streamed validation docs")
    # Checkpoint
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
