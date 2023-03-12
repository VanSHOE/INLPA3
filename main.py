import json
from pprint import pprint
from alive_progress import alive_bar
import re
import nltk
import os
import pickle


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
vocab = set()
if os.path.exists('vocab.pkl'):
    print("Loading Vocab...")
    vocab = pickle.load(open('vocab.pkl', 'rb'))
else:
    print("Calculating Vocab Size...")
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
                    vocab.add(word)
            bar()
    print("Saving Vocab...")
    pickle.dump(vocab, open('vocab.pkl', 'wb'))

print("Vocab Size: ", len(vocab))
