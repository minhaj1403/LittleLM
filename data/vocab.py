def create_mappings(content):
    string_to_int = {ch: i for i, ch in enumerate(sorted(set(content)))}
    int_to_string = {i: ch for i, ch in enumerate(sorted(set(content)))}
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])
    return string_to_int, int_to_string, encode, decode

def save_vocab(vocab, vocab_file):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in sorted(vocab):
            f.write(word + '\n')
    print(f"Vocabulary saved to {vocab_file}.")
