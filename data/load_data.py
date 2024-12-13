import os
import lzma
from tqdm import tqdm

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
