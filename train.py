import torch
from torch.optim import AdamW
import os

@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, get_batch, max_iters, eval_iters, eval_interval, learning_rate, checkpoint_path=None):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    start_iter = 0

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(model, optimizer, checkpoint_path)

    for iter in range(start_iter, max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"Step: {iter}, Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}")

            # Save checkpoint
            if checkpoint_path:
                save_checkpoint(model, optimizer, iter, checkpoint_path)

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Final Loss:", loss.item())


    print("Final Loss:", loss.item())

def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}")
    return epoch
