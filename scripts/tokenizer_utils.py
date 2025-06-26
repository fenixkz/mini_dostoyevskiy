import os
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer, Tokenizer


class RuTokenizer:

    def __init__(self, save_path: str, vocab_size: int = 15000):
        self.save_path = save_path
        self.vocab_size = vocab_size

    def train_tokenizer(self, train_data_path: str):

        # --- Шаг 1: Инициализация и обучение ByteLevelBPETokenizer ---

        # Инициализируем ByteLevelBPETokenizer
        # add_prefix_space=True - важный параметр, помогает в реконструкции предложений
        #                       (добавляет пробел в начало строки перед токенизацией)
        byte_level_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, lowercase=False) # lowercase=False, если вы хотите сохранить регистр. Можно поставить True.

        # Обучаем токенизатор
        # vocab_size - желаемый размер словаря
        # min_frequency - минимальная частота встречаемости
        # special_tokens - список специальных токенов
        print(f"Начинаем обучение ByteLevelBPETokenizer")
        byte_level_tokenizer.train(
            files=[train_data_path],
            vocab_size=self.vocab_size, 
            min_frequency=3,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
        print("Обучение ByteLevelBPETokenizer завершено.")
        
        byte_level_tokenizer.save(self.save_path)
        print(f"Токенизатор сохранен в: {self.save_path}")

    def encode(self, tokenizer: ByteLevelBPETokenizer, text):
        return tokenizer.encode(text).ids

    def decode(self, tokenizer: ByteLevelBPETokenizer, ids):
        return tokenizer.decode(ids)

    def tokenize_and_save_in_batches(self, tokenizer: Tokenizer, input_txt_path: str, output_bin_path: str):
        """
        Reads a large text file line-by-line, tokenizes it, and saves the token IDs
        to a binary file in a memory-efficient way.
        """
        print(f"Tokenizing {input_txt_path} and saving to {output_bin_path}...")
        
        # Use a specific dtype to save space. uint16 is good for vocabs up to 65,535.
        DTYPE = np.uint16 

        # Count total lines for a nice progress bar
        try:
            with open(input_txt_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_txt_path}")
            return

        # Open the output file in binary 'append' mode ('ab').
        # This will create the file if it doesn't exist.
        with open(output_bin_path, 'ab') as output_file:
            with open(input_txt_path, 'r', encoding='utf-8') as input_file:
                for line in tqdm(input_file, total=num_lines, desc=f"Processing {os.path.basename(input_txt_path)}"):
                    if line.strip():
                        # Tokenize one line at a time
                        token_ids = self.encode(tokenizer=tokenizer, text=line)
                        
                        # Convert this small list of IDs to a numpy array with our chosen dtype
                        chunk_array = np.array(token_ids, dtype=DTYPE)
                        
                        # Write the raw bytes of this array to the file.
                        # This is very efficient.
                        chunk_array.tofile(output_file)
        
        # Verify the number of tokens saved
        final_token_count = os.path.getsize(output_bin_path) // np.dtype(DTYPE).itemsize
        print(f"Tokenization and saving complete. Total tokens saved: {final_token_count:,}")
