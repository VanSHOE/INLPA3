import json
from pprint import pprint
from alive_progress import alive_bar
import re
import nltk
import os
import pickle
import time


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

exit(0)
# co occurence matrix
w2Idx = {}
curIdx = 0
if os.path.exists('co_occurence.pkl'):
    print("Loading Co-Occurence Matrix...")
    co_occurence = pickle.load(open('co_occurence.pkl', 'rb'))
else:
    print("Calculating Co-Occurence Matrix...")
    co_occurence = [[0 for i in range(len(vocab))] for j in range(len(vocab))]
    print("Co-Occurence Matrix Initialized...")
    file.seek(0)
    with alive_bar(lines) as bar:
        for line in file:
            jD = json.loads(line)
            sentences = nltk.sent_tokenize(jD['reviewText'])
            for i in range(len(sentences)):
                sentences[i] = pre_process(sentences[i])

            token2d = [nltk.word_tokenize(sent) for sent in sentences]
            for token in token2d:
                for word in token:
                    if word not in w2Idx:
                        w2Idx[word] = curIdx
                        curIdx += 1

                    for word2 in token:
                        if word2 not in w2Idx:
                            w2Idx[word2] = curIdx
                            curIdx += 1
                        co_occurence[w2Idx[word]][w2Idx[word2]] += 1
            bar()
    print("Saving Co-Occurence Matrix...")
    pickle.dump(co_occurence, open('co_occurence.pkl', 'wb'))

print("Co-Occurence Matrix Size: ", len(co_occurence))
