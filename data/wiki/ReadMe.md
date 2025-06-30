# Dataset of open source wikipedia articles in Russian language


## Prerequisites

- [corus](https://github.com/natasha/corus) 
- [gdown](https://github.com/wkentaro/gdown)


## How to use

First you need to download the archived version using the following command:

```
wget https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2
```

After that you can use the script `parse.py` to parse the data into .txt file. This helps to divide the data into training and validation sets.

```
python parse.py
```

This script will create `train.txt` and `val.txt` files, which we will use for training and validating our GPT model.

Or for simplicity you can download already parsed train and val text files using `gdown` command:

```
gdown --id 1Zk_XgnNE8iP-7aMO4NyQRkF433FZtp37 # To download train.txt
gdown --id 1Kz2IQ0oSANYWKZAs1II-Ih9m8V5fz8ty # To download val.txt
```
