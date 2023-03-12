import json
from pprint import pprint
from alive_progress import alive_bar
import re
import nltk
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np


def pre_process(in_text: str) -> str:
    # lower case it
    in_text = in_text.lower()
    # tokenize hashtags
    in_text = re.sub(r"#(\w+)", r"<HASHTAG> ", in_text)
    in_text = re.sub(r'\d+(,(\d+))*(\.(\d+))?%?\s', '<NUMBER> ', in_text)
    # tokenize mentions
    in_text = re.sub(r"@(\w+)", r"<MENTION> ", in_text)
    # tokenize urls
    in_text = re.sub(r"http\S+", r"<URL> ", in_text)
    # starting with www
    in_text = re.sub(r"www\S+", r"<URL> ", in_text)

    special_chars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-', '_', '—'
                                                                                                                     '=',
                     '+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                     '9']

    # pad the special characters with spaces
    for char in special_chars:
        in_text = in_text.replace(char, ' ')

    # pad < and > with spaces
    in_text = in_text.replace('<', ' <')
    in_text = in_text.replace('>', '> ')

    return in_text


nltk.download('punkt')

file = open('reviews_Movies_and_TV.json', 'r', encoding='utf-8')
# count lines without loading the whole file into memory

print("Counting lines...")
lines = 0

for line in file:
    lines += 1

print("Lines: ", lines)
vocab = {}
jsonLoadTime = 0
processTime = 0
dictTime = 0
if os.path.exists('vocab.pkl'):
    print("Loading Vocab...")
    vocab = pickle.load(open('vocab.pkl', 'rb'))
else:
    print("Calculating Vocab Size...")
    file.seek(0)
    with alive_bar(lines) as bar:
        for line in file:
            start = time.perf_counter()
            jD = json.loads(line)
            jsonLoadTime += time.perf_counter() - start
            start = time.perf_counter()
            sentences = nltk.sent_tokenize(jD['reviewText'])
            for i in range(len(sentences)):
                sentences[i] = pre_process(sentences[i])

            token2d = [nltk.word_tokenize(sent) for sent in sentences]
            processTime += time.perf_counter() - start
            start = time.perf_counter()
            for token in token2d:
                for word in token:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1

            dictTime += time.perf_counter() - start

            bar()
    print("Saving Vocab...")
    pickle.dump(vocab, open('vocab.pkl', 'wb'))
    print("JSON Load Time: ", jsonLoadTime)
    print("Process Time: ", processTime)
    print("Dict Time: ", dictTime)
    # Ratios
    print("JSON Load Time Ratio: ", jsonLoadTime / (jsonLoadTime + processTime + dictTime))
    print("Process Time Ratio: ", processTime / (jsonLoadTime + processTime + dictTime))
    print("Dict Time Ratio: ", dictTime / (jsonLoadTime + processTime + dictTime))

print("Vocab Size: ", len(vocab))

thresh = 500

new_vocab = {}
with alive_bar(len(vocab)) as bar:
    for word in vocab:
        if vocab[word] >= thresh:
            new_vocab[word] = vocab[word]
        bar()

print("Vocab Size: ", len(new_vocab))

co_occurrence = [0 for i in range(len(new_vocab)) for j in range(len(new_vocab))]
