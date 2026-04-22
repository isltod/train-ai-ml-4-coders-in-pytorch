from bs4 import BeautifulSoup
from collections import Counter
import string
import sys

sys.path.append(".")
from utils import stopwords


def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    # 이 부분은 책의 helper에 없고 sarcasm 예제에 따로 붙어있던 부분인데...
    # - , . 등을 단어에서 없애는 역할...maketrans(기존, 변경, 제외)
    table = str.maketrans("", "", string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return filtered_tokens


def build_vocab(texts):
    counter = Counter()
    for sentence in texts:
        tokens = tokenize(sentence)
        counter.update(tokens)

    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i + 1 for i, (word, _) in enumerate(sorted_words)}
    vocab["<pad>"] = 0
    return vocab


def texts_to_sequences(texts, word_index):
    sequences = []
    for sentence in texts:
        sequence = []
        for word in sentence.split():
            if word in word_index:
                sequence.append(word_index[word])
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded_seq = sequence + [0] * (max_length - len(sequence))
        else:
            padded_seq = sequence[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences


def word_frequency(texts, word_dict):
    frequency = {word: 0 for word in word_dict}
    for sentence in texts:
        words = sentence.lower().split()
        for word in words:
            if word in frequency:
                frequency[word] += 1
    return frequency
