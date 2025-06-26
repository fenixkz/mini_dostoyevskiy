import re
import unicodedata
from corus import load_wiki
from tqdm import tqdm
import os

def clean_text(text: str) -> str:
    """
    A gentle cleaning function for text from sources like Wikipedia.
    """
    if not text:
        return ""
    # Normalize unicode characters to a standard form (NFC).
    text = unicodedata.normalize('NFC', text)
    # Replace the non-breaking space with a regular space.
    text = text.replace('\xa0', ' ')
    # Collapse sequences of 3 or more newlines into just two.
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into a single space.
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def create_master_file():
    # --- Configuration ---
    # Define where the downloaded files will be stored by corus
    # Make sure you have enough disk space for these!
    WIKI_PATH = 'ruwiki-latest-pages-articles.xml.bz2'
    MASTER_FILE_PATH = 'all_data.txt'

    # --- De-duplication Set ---
    # Use a set to store hashes of seen records to avoid duplicates
    seen_records = set()

    print("Starting data consolidation process...")
    with open(MASTER_FILE_PATH, 'w', encoding='utf-8') as master_file:
        # --- Process Russian Wikipedia ---
        print(f"Loading and processing Wikipedia from {WIKI_PATH}...")
        wiki_records = load_wiki(WIKI_PATH)
        for record in tqdm(wiki_records, desc="Wikipedia"):
            cleaned = clean_text(record.text)
            record_hash = hash(cleaned)
            if cleaned and record_hash not in seen_records:
                master_file.write(cleaned + '\n')
                seen_records.add(record_hash)

    print(f"\nProcess complete. All data consolidated into {MASTER_FILE_PATH}")
    print(f"Total unique records processed: {len(seen_records)}")

if __name__ == '__main__':
    create_master_file()