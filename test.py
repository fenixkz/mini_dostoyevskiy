import os

book = []
for filename in os.listdir():
    if filename.endswith('.txt'):
        with open(filename, 'r') as file:
            book.extend(file.read())

print(book[:1500])