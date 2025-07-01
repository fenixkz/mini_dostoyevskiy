import os
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer, Tokenizer


class RuTokenizer:

    def __init__(self, save_path: str = "data/tokenizer", vocab_size: int = 15000):
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

    @staticmethod
    def encode(tokenizer: ByteLevelBPETokenizer, text):
        return tokenizer.encode(text).ids
    
    @staticmethod
    def decode(tokenizer: ByteLevelBPETokenizer, ids):
        return tokenizer.decode(ids)

    def tokenize_and_save_in_batches(self, tokenizer: Tokenizer, input_txt_path: str, output_bin_path: str):
        """
        Reads a large text file line-by-line, tokenizes it, and saves the token IDs
        to a binary file in a memory-efficient way.
        """
        print(f"Tokenizing {input_txt_path} and saving to {output_bin_path}...")
        
        # Use a specific dtype to save space. uint16 is good for vocabs up to 65,535.
        DTYPE = np.uint16 
        BATCH_SIZE_LINES = 100_000

        # Open the output file in binary 'append' mode ('ab').
        with open(output_bin_path, 'ab') as output_file:
            with open(input_txt_path, 'r', encoding='utf-8') as input_file:
                # Use tqdm to track progress by file size for better estimation
                file_size = os.path.getsize(input_txt_path)
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Processing {os.path.basename(input_txt_path)}") as pbar:
                    while True:
                        # 1. Read a large batch of lines into memory
                        lines_batch = [next(input_file, '').strip() for _ in range(BATCH_SIZE_LINES)]
                        # Filter out any empty strings that may have been read
                        lines_batch = [line for line in lines_batch if line]
                        
                        # If the batch is empty, we've reached the end of the file
                        if not lines_batch:
                            break

                        # 2. Tokenize the entire batch at once.
                        # The `encode_batch` method is highly optimized for this.
                        encoded_batch = tokenizer.encode_batch(lines_batch)
                        
                        # 3. Concatenate all token IDs from the batch into a single list
                        all_token_ids = []
                        for encoding in encoded_batch:
                            all_token_ids.extend(encoding.ids)
                        
                        # 4. Convert the large list of IDs to a numpy array and write to disk once
                        if all_token_ids:
                            chunk_array = np.array(all_token_ids, dtype=DTYPE)
                            chunk_array.tofile(output_file)
                        
                        # Update progress bar based on bytes read
                        pbar.update(sum(len(line.encode('utf-8')) for line in lines_batch))

        final_token_count = os.path.getsize(output_bin_path) // np.dtype(DTYPE).itemsize
        print(f"\nTokenization and saving complete. Total tokens saved: {final_token_count:,}")