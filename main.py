import json
from pprint import pprint
from alive_progress import alive_bar
import re
import nltk
import os
import pickle
import time
import numpy as np
import plotly.express as px


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

if not os.path.exists('reviews_Movies_and_TV_processed.json'):
    newJson = open("reviews_Movies_and_TV_processed.json", "a", encoding='utf-8')
    file.seek(0)
    with alive_bar(lines) as bar:
        for line in file:
            jD = json.loads(line)
            sentences = nltk.sent_tokenize(jD['reviewText'])
            for i in range(len(sentences)):
                sentences[i] = pre_process(sentences[i])

            token2d = [nltk.word_tokenize(sent) for sent in sentences]
            # save to json
            jD['reviewText'] = token2d
            newJson.write(json.dumps(jD) + '\n')
            bar()
    newJson.close()

processed_file = open('reviews_Movies_and_TV_processed.json', 'r', encoding='utf-8')
vocab = {}
if os.path.exists('vocab.pkl'):
    print("Loading Vocab...")
    vocab = pickle.load(open('vocab.pkl', 'rb'))
else:
    print("Calculating Vocab Size...")
    file.seek(0)
    with alive_bar(lines) as bar:
        for line in processed_file:
            jD = json.loads(line)
            token2d = jD['reviewText']

            for token in token2d:
                for word in token:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1

            bar()
    print("Saving Vocab...")
    pickle.dump(vocab, open('vocab.pkl', 'wb'))

print("Vocab Size: ", len(vocab))

thresh = 500

new_vocab = {}
with alive_bar(len(vocab)) as bar:
    for word in vocab:
        if vocab[word] >= thresh:
            new_vocab[word] = vocab[word]
        bar()

print("Vocab Size: ", len(new_vocab))
new_vocabIdx = {}

co_occurrence = np.zeros((len(new_vocab), len(new_vocab)))
if os.path.exists('co_occurrence.pkl'):
    print("Loading Co-Occurrence...")
    new_vocabIdx, co_occurrence = pickle.load(open('co_occurrence.pkl', 'rb'))
else:
    idx = 0
    for word in new_vocab:
        new_vocabIdx[word] = idx
        idx += 1

    print("Calculating Co-Occurrence...")
    processed_file.seek(0)
    with alive_bar(lines) as bar:
        for line in processed_file:
            jD = json.loads(line)
            token2d = jD['reviewText']

            for token in token2d:
                indices = [new_vocabIdx.get(word, None) for word in token]
                indices = [i for i in indices if i is not None]
                co_occurrence[np.ix_(indices, indices)] += 1

            bar()
    print("Saving Co-Occurrence...")
    pickle.dump((new_vocabIdx, co_occurrence), open('co_occurrence.pkl', 'wb'))
print(co_occurrence)
print("Co-Occurrence Shape: ", co_occurrence.shape)
# Take the SVD of co-occurrence matrix
if os.path.exists('svd.pkl'):
    print("Loading SVD...")
    u, s, vh = pickle.load(open('svd.pkl', 'rb'))
else:
    u, s, vh = np.linalg.svd(co_occurrence)
    print("Saving SVD...")
    pickle.dump((u, s, vh), open('svd.pkl', 'wb'))

# Print shapes of the matrices
print("U Shape: ", u.shape)
print("S Shape: ", s.shape)
print("Vh Shape: ", vh.shape)

# Plot the singular values dark mode
fig = px.line(x=np.arange(1, len(s) + 1), y=s, title='Singular Values', template='plotly_dark')
fig.show()
