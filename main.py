from config import *
from models.model import GPTLanguageModel
from train import train_model
from data.vocab import create_mappings

# Assuming `get_batch` is defined elsewhere or replace with actual implementation
def get_batch(split):
    raise NotImplementedError("Replace with actual get_batch function")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Main script with checkpoint support")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to save/load checkpoints")
    args = parser.parse_args()
    
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create mappings
    string_to_int, int_to_string, encode, decode = create_mappings(content)

    vocab_size = 100  # Replace with actual vocab size
    model = GPTLanguageModel(vocab_size).to(device)

    train_model(
        model=model,
        get_batch=get_batch,
        max_iters=max_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        checkpoint_path=args.checkpoint_path
    )
    print("Training Complete!")
