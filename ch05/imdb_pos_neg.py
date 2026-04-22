import os
import urllib.request
import tarfile
from collections import Counter
from bs4 import BeautifulSoup
from itertools import islice
import sys

sys.path.append(".")
from utils import stopwords


def download_and_extract_imdb(url, destination):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    file_path = os.path.join(destination, "aclImdb_v1.tar.gz")

    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")

    if not os.path.exists(os.path.join(destination, "aclImdb")):
        print("Extracting files...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(destination)
        print("Extraction complete.")


def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return [word.lower() for word in text.split() if word.lower() not in stopwords]


def build_vocab(path):
    # 집합처럼 유일하게 키를 만들고, 갯수를 값을 dict
    counter = Counter()
    for foler_name in ["pos", "neg"]:
        folder_path = os.path.join(path, foler_name)
        for file_name in os.listdir(folder_path):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                tokens = tokenize(text)
                counter.update(tokens)
    # 빈도에 따라 내림차순으로 정렬...근데 왜?
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # 인덱스를 1부터 시작해야...패딩에 0, 그럼 없는 단어는?
    vocab = {word: i + 1 for i, (word, _) in enumerate(sorted_words)}
    # 패딩에 0 토큰
    vocab["<pad>"] = 0
    return vocab


def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]


def pad_sequence(sequences, max_len):
    return [
        seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ]


if __name__ == "__main__":
    # 데이터 받기
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    # download_and_extract_imdb(dataset_url, "./data")

    # 어휘 사전 만들기
    # test에는 train에 없는 단어(OOV로 처리)들이 있다...
    vocab = build_vocab("./data/aclImdb/train")
    print(dict(islice(vocab.items(), 20)))
    print(len(vocab))

    # sequence 만들기
    texts = ["Today is a sunny day", "Today is a rainy day", "Is it sunny today?"]
    sequences = [text_to_sequence(text, vocab) for text in texts]
    padded_seqs = [pad_sequence([seq], 10) for seq in sequences]
    print(sequences)
    print(padded_seqs)

    # sequence를 문장으로 바꾸기
    revers_word_index = {index: word for word, index in vocab.items()}
    decoded_texts = [
        " ".join([revers_word_index.get(index, "<UNK>") for index in sequence])
        for sequence in sequences
    ]
    print(decoded_texts)
