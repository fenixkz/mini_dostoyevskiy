import os
import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizer_utils import RuTokenizer
from tqdm import tqdm

# --- Configuration ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 15000


def main():
    """
    This script should be run once, before starting any training.
    It prepares all necessary data files.
    """
    print("--- Starting Data Preparation ---")
    
    # Define file paths
    train_data_path = f'{PROJECT_DIR}/data/wiki/train.txt'
    val_data_path = f'{PROJECT_DIR}/data/wiki/val.txt'
    
    train_bin_path = f'{PROJECT_DIR}/data/wiki/train.bin'
    val_bin_path = f'{PROJECT_DIR}/data/wiki/val.bin'
    tokenizer_path = f'{PROJECT_DIR}/data/tokenizer/tokenizer_{VOCAB_SIZE}.json'

    tokenizer_utils = RuTokenizer(tokenizer_path, VOCAB_SIZE)
    # 1. Train the tokenizer if it doesn't exist
    if not os.path.exists(tokenizer_path):
        tokenizer_utils.train_tokenizer(train_data_path)

    # 2. Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 3. Tokenize and save the training data
    if not os.path.exists(train_bin_path):
        tokenizer_utils.tokenize_and_save_in_batches(tokenizer, train_data_path, train_bin_path)

    # 4. Tokenize and save the validation data
    if not os.path.exists(val_bin_path):
        tokenizer_utils.tokenize_and_save_in_batches(tokenizer, val_data_path, val_bin_path)

    print("\n--- Data Preparation Complete ---")
    print(f"Tokenizer is at: {tokenizer_path}")
    print(f"Training data is at: {train_bin_path}")
    print(f"Validation data is at: {val_bin_path}")

if __name__ == '__main__':
    main()