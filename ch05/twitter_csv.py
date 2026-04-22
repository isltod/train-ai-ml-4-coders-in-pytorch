from bs4 import BeautifulSoup
from collections import Counter
import csv
import sys

sys.path.append(".")
from utils import stopwords
from imdb_pos_neg import tokenize, text_to_sequence, pad_sequence


def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenize(sentence)
        counter.update(tokens)

    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i + 1 for i, (word, _) in enumerate(sorted_words)}
    vocab["<pad>"] = 0
    return vocab


if __name__ == "__main__":
    sentences = []
    labels = []

    with open("data/binary-emotion.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1].lower()
            sentence = sentence.replace(",", " , ")
            sentence = sentence.replace(".", " . ")
            sentence = sentence.replace("-", " - ")
            sentence = sentence.replace("/", " / ")
            soup = BeautifulSoup(sentence, "html.parser")
            sentence = soup.get_text()
            words = sentence.split()
            filtered_sentence = ""
            for word in words:
                if word not in stopwords:
                    filtered_sentence += word + " "
            sentences.append(filtered_sentence.strip())
    print(len(sentences))

    # 훈련 세트와 테스트 세트 만들기...
    train_size = int(len(sentences) * 0.8)
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    test_sentences = sentences[train_size:]
    test_labels = labels[train_size:]

    # 어휘 사전 만들기
    vocab = build_vocab(train_sentences)
    print(len(vocab))

    # sequence 만들기
    sequences = [text_to_sequence(text, vocab) for text in test_sentences]
    print(test_sentences[1])
    print(sequences[1])
    # padded_seqs = [pad_sequence([seq], 10) for seq in sequences]
