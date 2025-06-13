import os
import random
import re
import html
# Теперь очистим 
import re
import html
from torch.utils.data import Dataset

def clean_text(text):
    """
    Очищает текст, удаляя HTML-экранирование, номера страниц, глав, сноски и спецсимволы.

    Args:
        text (str): Исходный текст для очистки.

    Returns:
        str: Очищенный текст.
    """
    
    # 1. Декодирование HTML-сущностей (например, &amp; -> &)
    # Комментарий: Это должно быть одним из первых шагов.
    text = html.unescape(text)

    advertisement_patterns = [
        r'^\s*Спасибо, что скачали книгу в бесплатной электронной библиотеке.*?Royallib\.ru.*?\n',
        r'^\s*Все книги автора:.*?Royallib\.ru.*?\n',
        r'^\s*Эта же книга в других форматах:.*?Royallib\.ru.*?\n',
        r'^\s*Приятного чтения!\s*\n',
        r'^\s*Royallib Publishing\s*\n', # Возможная дополнительная строка
        r'^\s*http://royallib\.ru.*?\n', # Более общее правило для URL Royallib
        r'^\s*http://royallib\.com.*?\n', # Если домен .com
        # Добавьте другие шаблоны, если найдете рекламу от других источников
    ]
    for pattern in advertisement_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 2. Удаление специфических заголовков и разделителей (строки целиком)
    # Комментарий: Удаляем известные заголовки и маркеры перед основной обработкой глав/страниц.
    # Эти шаблоны применяются ко всему тексту с флагом MULTILINE.
    full_line_removals = [
        r'^\s*Annotation\s*$',      # Удаление строки "Annotation"
        r'^\s*notes\s*\d*\s*$',     # Удаление строк типа "notes" или "notes123"
        r'^\s*сноски\s*\d*\s*$',    # Русскоязычный вариант "notes"
        r'^\s*\*\s*\*\s*\*\s*$',    # Удаление разделителей "***"
    ]
    for pattern in full_line_removals:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 3. Удаление заголовков глав
    # Комментарий: Ищем строки, начинающиеся с типичных обозначений глав/частей.
    # Обновленный шаблон для "Часть перваяI" и подобных.
    # Также удаляем строки, состоящие только из римских или арабских цифр (часто номера глав/разделов).
    chapter_patterns = [
        # Шаблон для "Глава X", "Part 1", "Часть первая", "Книга Первая", "Часть перваяI"
        r'^\s*(Глава|CHAPTER|Chapter|Part|PART|Часть|КНИГА|BOOK)\s+([IVXLCDM\d]+|[А-Яа-яЁё\w\s]+)([IVXLCDM\d]*)\.?\s*$',
        # Шаблон для строк, содержащих только римские или арабские цифры (номера глав/частей)
        r'^\s*([IVXLCDM]+|[0-9]+)\.?\s*$' 
    ]
    for pattern in chapter_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 4. Удаление номеров страниц (строки, состоящие только из цифр, или цифры по краям)
    # Комментарий: Эвристика для номеров страниц.
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # Строка только из цифр
    text = re.sub(r'^\s*\d+\s*\n', '', text, flags=re.MULTILINE) # Цифры в начале строки (редко, но возможно)
    text = re.sub(r'\n\s*\d+\s*$', '', text, flags=re.MULTILINE) # Цифры в конце строки (классические номера страниц)
    
    # Удаление inline ссылок на страницы типа "Стр. 23"
    text = re.sub(r'(Стр\.|Page|Стр)\s*\.?\s*\d+', '', text, flags=re.IGNORECASE)

    # 5. Удаление маркеров сносок (внутри текста)
    # Комментарий: Удаляем распространенные маркеры сносок типа [1], (2), *.
    footnote_marker_patterns = [
        r'\[\d+\]',              # [1], [23]
        r'\(\d+\)',              # (1), (23)
        r'\{\d+\}',              # {1}, {23}
        r'\s+\*\s+',             # Одиночная звездочка как маркер сноски (окруженная пробелами)
                                 # Осторожно: может удалить стилистически важные звездочки, если они не сноски.
                                 # Лучше сделать более специфичным, если это проблема.
        r'\(\*\)',               # (*)
        r'(?<!\w)\[\*\](?!\w)'   # [*] как отдельный маркер
    ]
    for pattern in footnote_marker_patterns:
        text = re.sub(pattern, ' ', text) # Заменяем на пробел, чтобы не склеить слова

    # 6. Удаление нежелательных специальных символов
    # Комментарий: Сохраняем кириллицу, латиницу, цифры, основную пунктуацию и пробельные символы (включая \n).
    # \w включает буквы, цифры и '_'.
    text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?:;"\'«»(\)„“”—\-]', '', text, flags=re.UNICODE)

    # 7. Нормализация пробельных символов (этот шаг выполняется последним)
    # Комментарий: Сначала обрабатываем горизонтальные пробелы на каждой строке,
    # затем удаляем пустые строки и нормализуем переносы строк.

    # Заменяем множественные горизонтальные пробелы (пробел, таб) на один пробел внутри строк
    text = re.sub(r'[ \t\f\v]+', ' ', text)

    # Разделяем на строки, очищаем каждую строку от начальных/конечных пробелов,
    # и собираем обратно только непустые строки.
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        stripped_line = line.strip() # Удаляем пробелы в начале и конце каждой строки
        if stripped_line: # Добавляем строку, только если она не пустая
            processed_lines.append(stripped_line)
    
    # Соединяем непустые строки одним символом новой строки.
    # Это создаст текст, где каждый "абзац" или значащая строка отделена одним \n.
    text = "\n".join(processed_lines)

    return text


def get_dataset():
    books = []
    for filename in os.listdir('books'):
        if filename.endswith('.txt'):
            with open(f'books/{filename}', 'r') as file:
                books.append(file.read())
    data = " ".join(books)
    data = clean_text(data)
    val_books = []
    for filename in os.listdir('val'):
        if filename.endswith('.txt'):
            with open(f'val/{filename}', 'r') as file:
                val_books.append(file.read())
    val_data = " ".join(val_books)
    val_data = clean_text(val_data)
    return data, val_data
    

class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.context_length]
        y = self.data[idx+1:idx+self.context_length+1]
        return x, y

