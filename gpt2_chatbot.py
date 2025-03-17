import torch
import pickle
from transformers import GPT2Tokenizer
from model import GPTLanguageModel  #  model definitions in model.py

model_to_use = "model-01.pkl"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize GPT-2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(f"Using GPT-2 Tokenizer with vocab size: {tokenizer.vocab_size}")

def encode(text):
    return tokenizer.encode(text, truncation=True, max_length=256)

def decode(tokens):
    return tokenizer.decode(tokens)

# Define a custom unpickler to remap __main__ to model
class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "model"  # remap __main__ to model
        return super().find_class(module, name)

# Load the trained model using the custom unpickler
with open(model_to_use, 'rb') as f:
    model = ModelUnpickler(f).load()

model.to(device)
model.eval()
print("Model loaded successfully!")

print("Enter a prompt (Ctrl+C to exit):")
while True:
    prompt = input("Prompt:")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_tokens = model.generate(context.unsqueeze(0), max_new_tokens=150)
    output_text = decode(generated_tokens[0].tolist())
    print("Completion:\n" + output_text)
