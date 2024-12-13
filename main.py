from config import *
from models.model import LittleLM
from train import train_model
from utils.load_data import prepare_data, get_batch

if __name__ == "__main__":
    # Prepare data
    prepare_data(folder_path='../data', output_file_train='../data/train_split.txt', output_file_val='../data/val_split.txt')
    
    # Read the actual vocabulary size
    with open('../data/vocab.txt', 'r', encoding='utf-8') as f:
        vocab_size = len(f.readlines())

    model = LittleLM(vocab_size).to(device)

    train_model(
        model=model,
        get_batch=get_batch,
        max_iters=max_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path
    )
    print("Training Complete!")