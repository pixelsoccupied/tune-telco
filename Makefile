.PHONY: demo train install clean help

# Default target
.DEFAULT_GOAL := help

# Run the MCP demo with side-by-side comparison
demo:
	@echo "Running MCP demo..."
	uv run mcp/client.py

# Alias for demo
run: demo

# Train the model
train:
	@echo "Starting training..."
	uv run train/train.py

# Install dependencies (if needed)
install:
	@echo "Installing dependencies..."
	uv sync

# Clean generated data and cache
clean:
	@echo "Cleaning up..."
	rm -rf train/data/*.jsonl
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean training artifacts (adapters, data)
clean-all: clean
	@echo "Removing training artifacts..."
	rm -rf train/adapters/
	rm -rf train/data/

# Help target
help:
	@echo "Available targets:"
	@echo "  make demo      - Run the MCP demo (base vs fine-tuned comparison)"
	@echo "  make run       - Alias for demo"
	@echo "  make train     - Train the model using train/config.yaml"
	@echo "  make install   - Install dependencies using uv"
	@echo "  make clean     - Remove cache and generated JSONL files"
	@echo "  make clean-all - Remove all training artifacts (adapters, data)"
	@echo "  make help      - Show this help message"