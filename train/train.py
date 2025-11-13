#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-3B-Instruct on Glaive function calling dataset using MLX LoRA.

This script:
1. Loads the Glaive function calling dataset
2. Converts it to MLX chat format
3. Fine-tunes using LoRA
4. Saves the adapter weights
"""

import json
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress
import subprocess
import sys

console = Console()

# Training configuration
CONFIG = {
    "model": "mlx-community/Qwen3-0.6B-4bit",  # Small model, NO native function calling
    "dataset": "glaiveai/glaive-function-calling-v2",
    "adapter_path": "./adapters",
    "train_samples": 40000,  # Increased from 10k for better coverage (full dataset has 112k)
    "iters": 2500,  # Increased from 1200 for better convergence
    "batch_size": 4,  # Not actually used - overridden by memory optimizations
    "learning_rate": 1e-5,
    "val_samples": 1000,  # Increased proportionally
}


def convert_to_chat_format(example):
    """
    Convert Glaive format to MLX chat format.

    Glaive format:
    {
        "system": "SYSTEM: You are a helpful assistant with access to...",
        "chat": "USER: ...\n\n\nA: <functioncall> {...} <|endoftext|>\n\n\nFUNCTION RESPONSE: {...}\n\n\nA: ..."
    }

    MLX format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    import re

    messages = []

    # Add system message
    if example["system"]:
        # Remove "SYSTEM: " prefix if present
        system_content = example["system"].replace("SYSTEM: ", "")
        messages.append({"role": "system", "content": system_content})

    # Parse chat turns using regex to split on role markers
    # The chat format uses patterns like "USER:", "ASSISTANT:", "FUNCTION RESPONSE:"
    # separated by blank lines (\n\n\n)
    chat = example["chat"]

    # Split by role markers while preserving the marker
    # Pattern: Split on USER:, ASSISTANT:, or FUNCTION RESPONSE: at start of segment
    parts = re.split(r'\n\n+(?=(?:USER:|ASSISTANT:|FUNCTION RESPONSE:))', chat)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("USER:"):
            # Extract user message
            content = part.replace("USER:", "", 1).strip()
            # Remove endoftext marker if present
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({"role": "user", "content": content})

        elif part.startswith("ASSISTANT:"):
            # Extract assistant message
            content = part.replace("ASSISTANT:", "", 1).strip()
            # Remove endoftext marker if present
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({"role": "assistant", "content": content})

        elif part.startswith("FUNCTION RESPONSE:"):
            # Skip function responses - they're simulated results
            # The model should only learn to call functions, not generate fake results
            continue

    return {"messages": messages}


def prepare_dataset():
    """Load and prepare the dataset for training."""
    console.print("\n[bold cyan]Preparing Dataset...[/bold cyan]")

    # Load dataset
    console.print("  • Loading Glaive function calling dataset...")
    dataset = load_dataset(CONFIG["dataset"])

    console.print(f"  • Total examples: {len(dataset['train']):,}")

    # Take subset for training, validation, and test
    train_size = CONFIG["train_samples"]
    val_size = CONFIG["val_samples"]
    test_size = CONFIG["val_samples"]  # Same size as validation

    console.print(f"  • Using {train_size:,} training samples")
    console.print(f"  • Using {val_size:,} validation samples")
    console.print(f"  • Using {test_size:,} test samples")

    # Split dataset: train | validation | test
    train_dataset = dataset["train"].select(range(train_size))
    val_dataset = dataset["train"].select(range(train_size, train_size + val_size))
    test_dataset = dataset["train"].select(range(train_size + val_size, train_size + val_size + test_size))

    # Convert to chat format
    console.print("  • Converting to MLX chat format...")

    train_data = []
    first_example_shown = False

    for idx, example in enumerate(train_dataset):
        try:
            converted = convert_to_chat_format(example)
            if converted["messages"]:  # Only add if we got valid messages
                train_data.append(converted)

                # Show the first example's conversion
                if not first_example_shown:
                    first_example_shown = True
                    console.print("\n[bold yellow]Example Conversion (First Training Sample):[/bold yellow]")
                    console.print("[bold cyan]ORIGINAL (Glaive format):[/bold cyan]")
                    console.print(f"  System: {example['system'][:100]}...")
                    console.print(f"  Chat: {example['chat'][:200]}...")
                    console.print("\n[bold green]CONVERTED (MLX format):[/bold green]")
                    import json
                    console.print(json.dumps(converted, indent=2))
                    console.print()
        except Exception as e:
            # Skip problematic examples
            continue

    val_data = []
    for example in val_dataset:
        try:
            converted = convert_to_chat_format(example)
            if converted["messages"]:
                val_data.append(converted)
        except Exception as e:
            continue

    test_data = []
    for example in test_dataset:
        try:
            converted = convert_to_chat_format(example)
            if converted["messages"]:
                test_data.append(converted)
        except Exception as e:
            continue

    console.print(f"  • Converted {len(train_data):,} training examples")
    console.print(f"  • Converted {len(val_data):,} validation examples")
    console.print(f"  • Converted {len(test_data):,} test examples")

    # Save to JSONL files
    train_file = Path("data/train.jsonl")
    val_file = Path("data/valid.jsonl")
    test_file = Path("data/test.jsonl")

    train_file.parent.mkdir(exist_ok=True)

    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_file, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"  • [bold green]✓ Saved to {train_file}, {val_file}, and {test_file}[/bold green]\n")

    return train_file, val_file


def start_training(train_file, val_file):
    """Start MLX LoRA training using config file."""
    console.print("[bold cyan]Starting Training...[/bold cyan]")

    # Create adapter directory
    adapter_path = Path(CONFIG["adapter_path"])
    adapter_path.mkdir(exist_ok=True)

    # Use config file for cleaner configuration management
    config_file = Path(__file__).parent / "config.yaml"

    cmd = [
        "mlx_lm.lora",
        "--config", str(config_file),
    ]

    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  • Config file: {config_file}")
    console.print(f"  • Model: {CONFIG['model']}")
    console.print(f"  • Training samples: {CONFIG['train_samples']:,}")
    console.print(f"  • Iterations: {CONFIG['iters']:,}")
    console.print(f"  • Batch size: 1 (effective: 4 via gradient accumulation)")
    console.print(f"  • Trainable layers: 8")
    console.print(f"  • Learning rate: {CONFIG['learning_rate']}")
    console.print()

    console.print("[bold yellow]Training will take 1-2 hours...[/bold yellow]")
    console.print("[dim]You can monitor progress below:[/dim]\n")
    console.print("=" * 80)

    # Run training and track time
    import time
    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)

        # Calculate training time
        end_time = time.time()
        training_duration = end_time - start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        seconds = int(training_duration % 60)

        console.print("=" * 80)
        console.print("\n[bold green]✓ Training completed successfully![/bold green]")
        console.print(f"[bold]Total training time:[/bold] {hours}h {minutes}m {seconds}s")
        console.print(f"[bold]Adapters saved to:[/bold] {adapter_path.absolute()}\n")

        return True

    except subprocess.CalledProcessError as e:
        console.print("=" * 80)
        console.print(f"\n[bold red]✗ Training failed with error:[/bold red]")
        console.print(f"  {e}")
        return False
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Training interrupted by user[/yellow]")
        console.print(f"[dim]Partial adapters may be saved in {adapter_path}[/dim]")
        return False


def main():
    """Main training pipeline."""
    try:
        console.print("\n[bold]MLX LoRA Fine-Tuning for Function Calling[/bold]")
        console.print("=" * 80)

        # Show configuration
        console.print("\n[bold]Configuration:[/bold]")
        for key, value in CONFIG.items():
            console.print(f"  • {key}: {value}")

        console.print("\n" + "=" * 80)

        # Prepare dataset
        train_file, val_file = prepare_dataset()

        # Start training
        success = start_training(train_file, val_file)

        if success:
            console.print("[bold cyan]Next Steps:[/bold cyan]")
            console.print("  1. Test the fine-tuned model:")
            console.print(f"     uv run python test_finetuned_v2.py")
            console.print("  2. Compare with baseline results")
            console.print("  3. Iterate on training if needed\n")

        sys.exit(0 if success else 1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
