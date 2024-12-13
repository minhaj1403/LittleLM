import os
import lzma
from tqdm import tqdm
import mmap
import random
import torch
from config import device, block_size, batch_size
from vocab import encode

def get_random_chunk(split, train_split='data/train_split.txt', val_split='data/val_split.txt'):
    filename = train_split if split == 'train' else val_split
    with open(filename, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split, train_split='data/train_split.txt', val_split='data/val_split.txt'):
    data = get_random_chunk(split, train_split, val_split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def xz_files_in_dir(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith('.xz') and os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    print(f"Found {len(files)} .xz files in directory.")
    return files

def prepare_data(folder_path, output_file_train, output_file_val):
    files = xz_files_in_dir(folder_path)
    total_files = len(files)
    split_index = int(total_files * 0.9)
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    vocab = set()
    
    with open(output_file_train, 'w', encoding='utf-8') as outfile:
        for file in tqdm(train_files, total=len(train_files)):
            with lzma.open(os.path.join(folder_path, file), 'rt', encoding='utf-8') as f:
                text = f.read()
                outfile.write(text)
                vocab.update(set(text))
    
    with open(output_file_val, 'w', encoding='utf-8') as outfile:
        for file in tqdm(val_files, total=len(val_files)):
            with lzma.open(os.path.join(folder_path, file), 'rt', encoding='utf-8') as f:
                text = f.read()
                outfile.write(text)
                vocab.update(set(text))
    
    return vocab
