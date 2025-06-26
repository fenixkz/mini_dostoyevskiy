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

This script will create a `all_data.txt` file, which we can shuffle and then divide into two sets using 5% for validation set

```
shuf all_data.txt -o all_data.shuffled.txt
wc -l all_data.shuffled.txt # To count the total number of lines
head -n 7045040 all_data.shuffled.txt > train.txt # 7045040 is 95% of total number of lines in the all_data.txt file
tail -n 370792 all_data.shuffled.txt > val.txt # 370792 is 5% of total number of lines in the all_data.txt file
```

Or for simplicity you can download already parsed train and val text files using `gdown` command:

```
gdown --id 1oCTCCUGxJH3RCIYEqtGa8k_SwZuwcvXo # To download train.txt
gdown --id 1kGRxry-VZw8DR60ycucoiL5FN24s-6QJ # To download val.txt
```
