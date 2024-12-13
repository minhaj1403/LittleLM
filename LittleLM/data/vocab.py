def save_vocab(vocab, vocab_file):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in sorted(vocab):
            f.write(word + '\n')
    print(f"Vocabulary saved to {vocab_file}.")
