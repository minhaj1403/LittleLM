import argparse
import torch
import time

# Command-line argument parser
parser = argparse.ArgumentParser(description="Hyperparameter configuration")
parser.add_argument('--n_layer', type=int, default=8, help="Number of layers (default: 12)")
parser.add_argument('--n_head', type=int, default=8, help="Number of heads (default: 8)")
parser.add_argument('--n_embd', type=int, default=384, help="Embedding size (default: 768)")
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate (default: 0.1)")
parser.add_argument('--block_size', type=int, default=64, help="Block size (default: 64)")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size (default: 128)")
parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate (default: 3e-4)")
parser.add_argument('--max_iters', type=int, default=3000, help="Maximum iterations (default: 3000)")
parser.add_argument('--eval_iters', type=int, default=100, help="Evaluation iterations (default: 100)")
parser.add_argument('--eval_interval', type=int, default=500, help="Evaluation interval (default: 500)")
parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to save/load checkpoints")

# Parse arguments
args = parser.parse_args()

# Assign parsed values or defaults
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
block_size = args.block_size
batch_size = args.batch_size
learning_rate = args.learning_rate
max_iters = args.max_iters
eval_iters = args.eval_iters
eval_interval = args.eval_interval
checkpoint_path = args.checkpoint_path

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
time.sleep(0.4)
print("Loading configuration for LittleLM")
time.sleep(2)
print("#" * 30)
print(f"Using device: {device}")
time.sleep(0.4)
print(f"Number of layers: {n_layer}")
time.sleep(0.4)
print(f"Number of heads: {n_head}")
time.sleep(0.4)
print(f"Embedding size: {n_embd}")
time.sleep(0.4)
print(f"Dropout rate: {dropout}")
time.sleep(0.4)
print(f"Block size: {block_size}")
time.sleep(0.4)
print(f"Batch size: {batch_size}")
time.sleep(0.4)
print(f"Learning rate: {learning_rate}")
time.sleep(0.4)
print(f"Maximum iterations: {max_iters}")
time.sleep(0.4)
print(f"Evaluation iterations: {eval_iters}")
time.sleep(0.4)
print(f"Evaluation interval: {eval_interval}")
time.sleep(0.4)
print(f"Checkpoint path: {checkpoint_path}")
print("#" * 30)
print('#' * 30)
print('#' * 30)
print("\n")