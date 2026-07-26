#!/usr/bin/env python
"""
BERT Classification Training Script with PAI integration on AWS Trainium

This script trains a Transformer model (RoBERTa or BERT) on either an IMDB
or SNLI dataset using PAI conversion on AWS Trainium/Neuron. It uses command‐line 
arguments for all configuration parameters.

Features:
  • Supports either IMDB or SNLI datasets.
  • Includes a flag (--dsn) that, when enabled, sets the number of encoder layers to 0.
  • Uses Perforated Backpropagation implemented through the PerforatedAI library.
  • Runs on AWS Trainium using torch_neuronx (Beta 3 stack).

Setup for PAI on Trainium:
    git clone https://github.com/PerforatedAI/PerforatedAI-Transformers.git
    cd PerforatedAI-Transformers
    pip install -e .
    pip install perforatedai evaluate scikit-learn 
    
    # Install Neuron SDK
    pip install torch-neuronx transformers datasets

    export PAIEMAIL=<your_email>
    export PAITOKEN=<your_pai_token>

Usage example for IMDB (uses all defaults - no arguments required):
    python train_bert_pai_neuron.py

Usage example for SNLI (only specify parameters that differ from IMDB defaults):
    python train_bert_pai_neuron.py --dataset snli --batch_size 256 --max_len 128 \
       --model_save_location ./model_output_PAI_SNLI_neuron
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BertConfig,
)

# Import PAI modules
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# For SNLI dataset
from datasets import load_dataset, Dataset, DatasetDict

from models import (
    BertForSequenceClassificationPB,
    RobertaForSequenceClassificationPB,
    ClassifierWrapper,
)


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Import torch_neuronx to register neuron device and enable XLA
try:
    import torch_neuronx  # noqa: F401
    import torch_xla.core.xla_model as xm
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False
    print("Warning: torch_neuronx not available. Falling back to CPU/GPU.")


# =============================================================================
# Helper Dataset Class
# =============================================================================
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Returns a dictionary with tokenized inputs and the label.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# =============================================================================
# Utility Functions
# =============================================================================
def count_model_parameters(model):
    regular_params = 0
    pb_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        if "parentModule" in name or "PBtoTop" in name:
            pb_params += num_params
            print(f"PB: {name} | {num_params:,}")
        elif "pb.layers" in name:
            # print(f"EXCLUDING SHARED PB: {name} | {num_params:,}")
            pass
        else:
            regular_params += num_params
            print(f"Regular: {name} | {num_params:,}")

    total_params = regular_params + pb_params

    print(f"\nRegular Model Parameters: {regular_params:,}")
    print(f"PB Parameters: {pb_params:,}")
    print(f"Total Parameters: {total_params:,}\n")

    return regular_params, pb_params, total_params


def load_imdb_dataset(
    tokenizer, max_len, seed=42, reduce_lines=False, cache_dir="./cached_datasets"
):
    """
    Downloads and loads the IMDB dataset directly from Hugging Face.
    Creates a stratified split of the training data to get a dev set.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Download the IMDB dataset
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)

    # Function to create stratified split
    def stratified_split(dataset, test_size=0.1, seed=42):
        # Get indices for each class
        pos_indices = np.where(np.array(dataset["label"]) == 1)[0]
        neg_indices = np.where(np.array(dataset["label"]) == 0)[0]

        # Calculate split sizes
        n_pos_test = int(len(pos_indices) * test_size)
        n_neg_test = int(len(neg_indices) * test_size)

        # Random split for each class
        np.random.seed(seed)
        pos_test_indices = np.random.choice(pos_indices, n_pos_test, replace=False)
        neg_test_indices = np.random.choice(neg_indices, n_neg_test, replace=False)

        # Combine test indices
        test_indices = np.concatenate([pos_test_indices, neg_test_indices])
        train_indices = np.array(
            [i for i in range(len(dataset)) if i not in test_indices]
        )

        return dataset.select(train_indices), dataset.select(test_indices)

    # Split train set into 90% train, 10% dev while maintaining class balance
    train_data, dev_data = stratified_split(dataset["train"], test_size=0.1, seed=seed)
    test_data = dataset["test"]

    # Optionally reduce dataset size for faster testing
    if reduce_lines:
        train_data = train_data.select(range(min(1000, len(train_data))))
        dev_data = dev_data.select(range(min(100, len(dev_data))))
        test_data = test_data.select(range(min(100, len(test_data))))

    # Print dataset statistics
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")

    # Print class distribution
    train_labels = train_data["label"]
    train_class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    print("\nClass distribution in train set:")
    for label, count in train_class_counts.items():
        print(f"Class {label}: {count} samples")

    # Tokenize datasets using mapped format for HuggingFace datasets
    train_dataset = train_data.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_len
        ),
        batched=True,
    )
    dev_dataset = dev_data.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_len
        ),
        batched=True,
    )
    test_dataset = test_data.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_len
        ),
        batched=True,
    )

    return train_dataset, dev_dataset, test_dataset


def load_snli_dataset(
    tokenizer, max_len, seed, reduce_lines=False, cache_dir="./cached_datasets"
):
    """
    Loads the SNLI dataset from Hugging Face.
    Filters out examples with label -1 and tokenizes using premise and hypothesis.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset("stanfordnlp/snli", cache_dir=cache_dir)

    # Filter out examples with no gold label.
    dataset = dataset.filter(lambda ex: ex["label"] != -1)
    dataset = dataset.shuffle(seed=seed)

    if reduce_lines:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))

    def tokenize_fn(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    train_dataset = SentimentDataset(
        {k: tokenized_datasets["train"][k] for k in tokenizer.model_input_names},
        tokenized_datasets["train"]["label"],
    )
    dev_dataset = SentimentDataset(
        {k: tokenized_datasets["validation"][k] for k in tokenizer.model_input_names},
        tokenized_datasets["validation"]["label"],
    )
    test_dataset = SentimentDataset(
        {k: tokenized_datasets["test"][k] for k in tokenizer.model_input_names},
        tokenized_datasets["test"]["label"],
    )
    return train_dataset, dev_dataset, test_dataset


def set_GPA_params(args):
    """
    Set PAI global parameters based on command-line arguments.
    """
    GPA.pc.set_save_name(args.pai_save_name)
    if args.switch_mode == "DOING_HISTORY":
        GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    elif args.switch_mode == "DOING_FIXED_SWITCH":
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_fixed_switch_num(args.fixed_switch_num)
        GPA.pc.set_firstfixed_switch_num(args.first_fixed_switch_num)
    else:
        raise ValueError(f"Invalid switch_mode: {args.switch_mode}")
    GPA.pc.set_n_epochs_to_switch(args.n_epochs_to_switch)
    GPA.pc.set_p_epochs_to_switch(args.p_epochs_to_switch)
    GPA.pc.set_output_dimensions([-1, -1, 0])
    GPA.pc.set_history_lookback(args.history_lookback)
    GPA.pc.set_max_dendrites(args.max_dendrites)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_improvement_threshold(args.improvement_threshold)
    GPA.pc.set_pai_improvement_threshold(args.pb_improvement_threshold)
    GPA.pc.set_improvement_threshold_raw(args.pb_improvement_threshold_raw)
    GPA.pc.set_unwrapped_modules_confirmed(args.unwrapped_modules_confirmed)
    GPA.pc.set_cap_at_n(args.cap_at_n)
    GPA.pc.set_debugging_output_dimensions(1)
    GPA.pc.set_metric('eval_accuracy')
    GPA.pc.set_weight_decay_accepted(True)
    
def resize_model_hidden_size(config, width_factor):
    if width_factor <= 0 or width_factor > 1.0:
        raise ValueError(f"Width factor must be in range (0, 1.0], got {width_factor}")
    if width_factor == 1.0:
        return config
    print(f"Resizing model hidden dimensions by factor {width_factor}")
    if hasattr(config, "hidden_size"):
        config.hidden_size = int(config.hidden_size * width_factor)
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = int(config.intermediate_size * width_factor)
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = max(
            8, int(config.num_attention_heads * width_factor) // 8 * 8
        )
    return config


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model with PAI integration on AWS Trainium."
    )
    # Model and tokenizer settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Name or path of the model (e.g., roberta-base or bert-base-uncased)",
    )
    # Dataset selection: imdb or snli
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imdb", "snli"],
        default="imdb",
        help="Dataset type: imdb or snli",
    )
    # PAI parameters
    parser.add_argument(
        "--wrap_full_model",
        action="store_true",
        help="If True then wrap encoder + classifier layers; if not set then only wrap classifier",
    )
    parser.add_argument(
        "--maximizing_score",
        type=bool,
        default=True,
        help="If True then maximize score for PAI; if not set then minimize score",
    )
    parser.add_argument(
        "--pai_save_name", type=str, default="default_pai", help="PAI save name"
    )
    parser.add_argument(
        "--switch_mode",
        type=str,
        choices=["DOING_HISTORY", "DOING_FIXED_SWITCH"],
        default="DOING_HISTORY",
        help="PAI switch mode",
    )
    parser.add_argument(
        "--history_lookback", type=int, default=1, help="History lookback for PAI"
    )
    parser.add_argument(
        "--fixed_switch_num",
        type=int,
        default=1,
        help="Fixed switch number (if using doingFixedSwitch)",
    )
    parser.add_argument(
        "--first_fixed_switch_num",
        type=int,
        default=1,
        help="First fixed switch number (if using doingFixedSwitch)",
    )
    parser.add_argument(
        "--n_epochs_to_switch", type=int, default=10, help="Number of epochs to switch"
    )
    parser.add_argument(
        "--p_epochs_to_switch", type=int, default=10, help="Percentage epochs to switch"
    )
    parser.add_argument(
        "--max_dendrites", type=int, default=5, help="Maximum dendrites"
    )
    parser.add_argument(
        "--improvement_threshold",
        type=float,
        default=0.0001,
        help="Improvement threshold",
    )
    parser.add_argument(
        "--pb_improvement_threshold",
        type=float,
        default=0.01,
        help="PB improvement threshold",
    )
    parser.add_argument(
        "--pb_improvement_threshold_raw",
        type=float,
        default=0.001,
        help="Raw PB improvement threshold",
    )
    parser.add_argument(
        "--unwrapped_modules_confirmed",
        type=bool,
        default=True,
        help="Unwrapped modules confirmed",
    )
    parser.add_argument("--cap_at_n", action="store_true", help="Whether to cap at N")
    # DSN flag: if enabled, set the number of encoder layers to 0.
    parser.add_argument(
        "--dsn",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Enable DSN mode (set number of encoder layers to 0)",
    )
    # Compression for model width
    parser.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="Width factor to shrink the model (between 0 and 1)",
    )
    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="Hidden dropout probability",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="Attention dropout probability",
    )
    # Saving and early stopping parameters
    parser.add_argument(
        "--model_save_location",
        type=str,
        default="./model_output_PAI_IMDB_neuron",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=6, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Early stopping threshold",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Number of steps between saves"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluation strategy (e.g., epoch or steps)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation steps (if strategy is steps)",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Save strategy (e.g., epoch or steps)",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit total number of saved checkpoints",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_accuracy",
        help="Metric for selecting best model",
    )
    parser.add_argument(
        "--greater_is_better",
        type=bool,
        default=True,
        help="Whether a higher metric is better",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="reduce_lr_on_plateau",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=2, help="Scheduler patience"
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.5, help="Scheduler factor"
    )
    parser.add_argument(
        "--scheduler_min_lr", type=float, default=1e-7, help="Minimum learning rate"
    )
    # For testing speed
    parser.add_argument(
        "--reduce_lines_for_testing",
        action="store_true",
        help="Reduce dataset size for testing",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set PAI parameters.
    set_GPA_params(args)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Set number of labels and load model configuration.
    if args.dataset == "imdb":
        num_labels = 2
    else:  # snli
        num_labels = 3

    config = BertConfig.from_pretrained(args.model_name, num_labels=num_labels)
    # Compression for model width
    if args.width < 1.0:
        config = resize_model_hidden_size(config, args.width)

    # Apply dropout settings.
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob

    # If DSN flag is enabled, set the number of encoder layers to 0.
    if args.dsn:
        config.num_hidden_layers = 0

    # Load pretrained model
    print(f"Loading pretrained model from {args.model_name}...")
    base_model = BertForSequenceClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True
    )

    # Wrap the base model for compatibility with PAI
    if "roberta" in args.model_name:
        GPA.pc.append_modules_to_replace([RobertaForSequenceClassification])
        GPA.pc.append_replacement_modules([RobertaForSequenceClassificationPB])
        GPA.pc.append_module_names_to_track(["RobertaEmbeddings"])
        GPA.pc.append_module_names_to_track(["Embedding"])

        model = RobertaForSequenceClassificationPB(
            base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob
        )
    elif "bert" in args.model_name:
        GPA.pc.append_modules_to_replace([BertForSequenceClassification])
        GPA.pc.append_replacement_modules([BertForSequenceClassificationPB])
        GPA.pc.append_module_names_to_track(["Embeddings"])
        GPA.pc.append_module_names_to_track(["Embedding"])
        GPA.pc.append_module_names_to_track(["LayerNorm"])

        model = BertForSequenceClassificationPB(
            base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob
        )
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Move model to device (Neuron if available, otherwise fallback to CUDA/CPU)
    if NEURON_AVAILABLE:
        device = xm.xla_device()
        print(f"Using Neuron device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    
    model = model.to(device)

    # Print parameter counts.
    count_model_parameters(model)

    # Load dataset.
    if args.dataset == "imdb":
        train_dataset, dev_dataset, test_dataset = load_imdb_dataset(
            tokenizer,
            args.max_len,
            seed=args.seed,
            reduce_lines=args.reduce_lines_for_testing,
        )
    else:  # SNLI
        train_dataset, dev_dataset, test_dataset = load_snli_dataset(
            tokenizer,
            args.max_len,
            seed=args.seed,
            reduce_lines=args.reduce_lines_for_testing,
        )

    # Prepare training arguments.
    if args.evaluation_strategy == "steps" and args.eval_steps is None:
        args.eval_steps = int(len(train_dataset) / args.batch_size)
    
    # Use bf16 for Neuron (recommended for Trainium), avoid fp16 for PAI
    use_bf16 = NEURON_AVAILABLE
    
    training_args = TrainingArguments(
        output_dir=(
            args.model_save_location if args.model_save_location else "./results_pai_neuron"
        ),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=(
            int(1000 / args.batch_size) if not args.reduce_lines_for_testing else 1
        ),
        weight_decay=0.01,
        logging_dir="./logs_pai_neuron",
        load_best_model_at_end=args.early_stopping,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        seed=args.seed,
        bf16=use_bf16,  # Use bf16 for Neuron
        # fp16 disabled for PAI to avoid precision issues
        run_name=f"run_pai_neuron_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        lr_scheduler_type=args.scheduler_type,
        lr_scheduler_kwargs=(
            {
                "patience": args.scheduler_patience,
                "factor": args.scheduler_factor,
                "min_lr": args.scheduler_min_lr,
            }
            if args.scheduler_type == "reduce_lr_on_plateau"
            else None
        ),
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        print("Early stopping enabled.")

    # Create Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=lambda pred: {
            "eval_accuracy": float(
                (pred.predictions.argmax(-1) == pred.label_ids).mean()
            )
        },
        callbacks=callbacks,
        using_perforatedai=True,
    )

    # Start training.
    print("Starting training with PAI on Trainium/Neuron...")
    # Convert model to be compatible with PB and initialize PB tracking
    UPA.perforate_model(
        model,
        save_name=GPA.pc.get_save_name(),
        maximizing_score=args.maximizing_score,
    )
    # Set input dimensions for the model
    for layer_name, layer in dict(model.named_modules()).items():
        try:
            if "roberta" in args.model_name:
                if layer_name in ["roberta.pooler", "roberta.pooler.dense", "classifier", "classifier.dense", "classifier.out_proj"]:
                    layer.set_this_output_dimensions([-1, 0])
            elif "bert" in args.model_name:
                if layer_name in ["bert.pooler", "bert.pooler.dense", "classifier"]:
                    layer.set_this_output_dimensions([-1, 0])
        except Exception as e:
            print(f"Could not set input dimensions for {layer_name}: {e}")

    trainer.train()

    # Evaluate on test set.
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = test_results.predictions.argmax(-1)
    test_labels = test_results.label_ids
    test_accuracy = (test_preds == test_labels).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save final model and tokenizer.
    if args.model_save_location:
        print(f"Saving model to {args.model_save_location}...")
        os.makedirs(args.model_save_location, exist_ok=True)
        trainer.save_model(args.model_save_location)
        tokenizer.save_pretrained(args.model_save_location)


if __name__ == "__main__":
    main()
