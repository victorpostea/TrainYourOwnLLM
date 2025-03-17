from datasets import load_dataset
from tqdm import tqdm

# Load WikiText-2 dataset (raw version)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Use the train split and validation split provided by WikiText-2.
train_data = dataset["train"]
val_data = dataset["validation"]

# Define output files
output_file_train = "train_split.txt"
output_file_val = "val_split.txt"
vocab_file = "vocab.txt"

def save_text_data(data, output_file):
    vocab = set()
    with open(output_file, "w", encoding="utf-8") as outfile:
        for example in tqdm(data, desc=f"Writing {output_file}"):
            text = example["text"]
            outfile.write(text + "\n")
            vocab.update(text)  # Collect unique characters
    return vocab

# Save training data and validation data
vocab_train = save_text_data(train_data, output_file_train)
vocab_val = save_text_data(val_data, output_file_val)

# Combine vocabularies and save to vocab.txt (one line with all characters)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    vfile.write("".join(sorted(vocab_train.union(vocab_val))))

print("Processing complete. Data saved to train_split.txt, val_split.txt, and vocab.txt.")