import os
import re
import unicodedata
import subprocess # We'll use this to call wikiextractor
import glob       # To find all the output files
import random     # To shuffle the articles
from tqdm import tqdm

def final_clean_text(text: str) -> str:
    """
    A gentle final cleaning function for the text after wikiextractor has
    removed the heavy markup. This is mostly your original function.
    """
    if not text:
        return ""
    
    # Remove leftover <doc ...> and </doc> tags from wikiextractor's output
    text = re.sub(r'<doc.*?>|</doc>', '', text)
    
    # Normalize unicode characters to a standard form (NFC).
    text = unicodedata.normalize('NFC', text)
    
    # Replace the non-breaking space with a regular space.
    text = text.replace('\xa0', ' ')
    
    # Collapse sequences of 3 or more newlines into just two.
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Collapse multiple spaces into a single space.
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def create_datasets():
    """
    This script now orchestrates a three-step process:
    1. Run wikiextractor to extract clean prose from the Wikipedia dump.
    2. Shuffle the extracted articles and split them into train/validation sets.
    3. Consolidate the articles into train.txt and val.txt.
    """
    # --- Configuration ---
    WIKI_DUMP_PATH = 'ruwiki-latest-pages-articles.xml.bz2'
    EXTRACTED_DIR = 'cleaned_wiki_text'  # Directory where wikiextractor will save its output
    TRAIN_FILE_PATH = 'train.txt'
    VAL_FILE_PATH = 'val.txt'
    VAL_SPLIT_RATIO = 0.05 # Use 5% of the data for validation

    # --- Step 1: Run WikiExtractor ---
    if not os.path.exists(EXTRACTED_DIR):
        print(f"--- Step 1: Running WikiExtractor on {WIKI_DUMP_PATH} ---")
        print("This may take a very long time...")
        os.makedirs(EXTRACTED_DIR, exist_ok=True)
        
        command = [
            'python', '-m', 'wikiextractor.WikiExtractor',
            '--processes', '8',
            '--no-templates',
            '-b', '100M',
            '-o', EXTRACTED_DIR,
            WIKI_DUMP_PATH
        ]
        
        try:
            subprocess.run(command, check=True)
            print("WikiExtractor finished successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n--- ERROR: Failed to run WikiExtractor: {e} ---")
            return
    else:
        print(f"--- Step 1: Skipped. Output directory '{EXTRACTED_DIR}' already exists. ---")


    # --- Step 2: Get and Shuffle Article Files ---
    print(f"\n--- Step 2: Discovering and shuffling article files ---")
    
    file_paths = glob.glob(os.path.join(EXTRACTED_DIR, '**', 'wiki_*'), recursive=True)

    if not file_paths:
        print(f"\n--- ERROR: No processed files found in '{EXTRACTED_DIR}'. ---")
        return

    random.shuffle(file_paths) # Shuffle the list of articles in-place
    
    # Split the shuffled list of files into training and validation sets
    split_index = int(len(file_paths) * (1 - VAL_SPLIT_RATIO))
    train_files = file_paths[:split_index]
    val_files = file_paths[split_index:]

    print(f"Total articles found: {len(file_paths):,}")
    print(f"Assigning {len(train_files):,} articles to training set.")
    print(f"Assigning {len(val_files):,} articles to validation set.")

    # --- Step 3: Consolidate Files into train.txt and val.txt ---
    def consolidate_files(files_to_process, output_path):
        """Helper function to consolidate a list of files into one master file."""
        print(f"\nConsolidating into {output_path}...")
        seen_paragraphs = set() # De-duplicate within each set
        
        with open(output_path, 'w', encoding='utf-8') as master_file:
            for file_path in tqdm(files_to_process, desc=f"Writing to {os.path.basename(output_path)}"):
                with open(file_path, 'r', encoding='utf-8') as extracted_file:
                    content = extracted_file.read()
                    paragraphs = content.split('\n\n')

                    for paragraph in paragraphs:
                        cleaned = final_clean_text(paragraph)
                        record_hash = hash(cleaned)
                        
                        if cleaned and record_hash not in seen_paragraphs:
                            master_file.write(cleaned + '\n\n')
                            seen_paragraphs.add(record_hash)

    # Consolidate training files
    consolidate_files(train_files, TRAIN_FILE_PATH)
    # Consolidate validation files
    consolidate_files(val_files, VAL_FILE_PATH)

    print("\n--- Data Preparation Complete ---")
    print(f"Training data is at: {TRAIN_FILE_PATH}")
    print(f"Validation data is at: {VAL_FILE_PATH}")


if __name__ == '__main__':
    create_datasets()
