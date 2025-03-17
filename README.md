# GPT-2 Chatbot with Custom Tokenizer

## Overview
This repository contains scripts for training and running a GPT-2 chatbot with either a custom tokenizer or the standard GPT-2 tokenizer. Users can train a model using a simple dataset (`wizard_of_oz.txt`) or extract and use the `wikitext-2` dataset for training.

## File Structure
- **`.gitignore`** - Ignores unnecessary files.
- **`chatbot_own_tokenizer.py`** - Chatbot script for models trained with the custom tokenizer.
- **`gpt2_chatbot.py`** - Chatbot script for models trained with the standard GPT-2 tokenizer.
- **`model.py`** - Model architecture and utility functions.
- **`training_gpt2_tokenizer.py`** - Training script using the standard GPT-2 tokenizer.
- **`training_own_tokenizer.py`** - Training script using a custom tokenizer.
- **`requirements.txt`** - Required dependencies.
- **`wizard_of_oz.txt`** - Sample dataset for training.
- **`data-extract.py`** - Script to extract and prepare the `wikitext-2` dataset.

## Setup
### 1. Set Up CUDA Virtual Environment
If using a virtual environment with CUDA support, create and activate it before installing dependencies:
```bash
python -m venv cuda_env  # Create a virtual environment
source cuda_env/bin/activate  # Activate it (use `cuda_env\Scripts\activate` on Windows)
```

### 2. Install Dependencies
Ensure you have Python installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Enable CUDA (If Using GPU)
If you plan to train using a GPU, ensure CUDA is installed and activated:
```bash
export CUDA_VISIBLE_DEVICES=0  # Adjust device index as needed
```

### 4. Prepare Training Data
You have two options for training data:
- Use the provided `wizard_of_oz.txt` dataset.
- Extract the `wikitext-2` dataset using:
  ```bash
  python data-extract.py
  ```

If using `wizard_of_oz.txt`, update the file path in the training script (`training_gpt2_tokenizer.py` or `training_own_tokenizer.py`).

## Training a Model
Run the appropriate training script:
- For GPT-2 tokenizer:
  ```bash
  python training_gpt2_tokenizer.py
  ```
- For custom tokenizer:
  ```bash
  python training_own_tokenizer.py
  ```

### Customizing Training
- **Set a model name**: Change `model_to_save` at the top of the training script.
- **Adjust hyperparameters**: Modify values at the top of the script to match your computational power.
- **Checkpointing**: Every 500 iterations, a checkpoint (`model-checkpoint.pkl`) is saved, allowing you to stop training and use the latest version.

## Running the Chatbot
After training, use the corresponding chatbot script:
- For a model trained with GPT-2 tokenizer:
  ```bash
  python gpt2_chatbot.py
  ```
- For a model trained with the custom tokenizer:
  ```bash
  python chatbot_own_tokenizer.py
  ```

### Selecting a Model
Set `model_to_use` at the top of the chatbot script to specify which trained model to load.

## Notes
- Training scripts allow flexibility in dataset selection and model configuration.
- Checkpointing enables stopping and resuming training at regular intervals.
- Hyperparameters are easily adjustable to accommodate different computational resources.
- If using a GPU, ensure CUDA is properly configured to accelerate training.
- When using a virtual environment, activate it before running scripts to ensure dependencies are correctly loaded.

## License
This project is open-source and available for modification and improvement.

---