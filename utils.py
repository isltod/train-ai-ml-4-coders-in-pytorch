import io
import json
import matplotlib.pyplot as plt
import os
import re
import string
import tarfile
import torch
import urllib.request
import zipfile
from bs4 import BeautifulSoup
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader


def print_model_summary(model):
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")


def download_and_extract(url, destination, file_name, dir_name):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    file_path = os.path.join(destination, file_name)

    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")

    if not os.path.exists(os.path.join(destination, dir_name)):
        print("Extracting files...")
        ext = os.path.splitext(file_name)[1]
        if ext == ".zip":
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(destination)
        elif ext == ".gz":
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(destination)
        print("Extraction complete.")


def load_sarcasm_dataset():
    with open("data/sarcasm.json", "r") as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []
    table = str.maketrans("", "", string.punctuation)
    for item in datastore:
        sentence = item["headline"].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
        labels.append(item["is_sarcastic"])
        urls.append(item["article_link"])

    return sentences, labels, urls


def load_datasets_sarcasm(
    sentences, labels, vocab_size, max_length, training_size, batch_size
):
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    word_index = build_vocab(training_sentences, max_vocab_size=vocab_size)
    training_sequences = texts_to_sequences(training_sentences, word_index)
    training_padded = pad_sequences(training_sequences, max_len=max_length)
    testing_sequences = texts_to_sequences(testing_sentences, word_index)
    testing_padded = pad_sequences(testing_sequences, max_len=max_length)

    # word_freq = word_frequency(sentences, word_index)

    training_padded = torch.tensor(training_padded, dtype=torch.long)
    testing_padded = torch.tensor(testing_padded, dtype=torch.long)
    training_labels = torch.tensor(training_labels, dtype=torch.float32)
    testing_labels = torch.tensor(testing_labels, dtype=torch.float32)

    train_dataset = TensorDataset(training_padded, training_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(testing_padded, testing_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, word_index


def train_sarcasm_model(
    model, device, train_loader, test_loader, criterion, optimizer, num_epochs
):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += targets.size(0)
            train_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                val_loss += loss.item()
                val_total += targets.size(0)
                val_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        train_loss_history.append(train_loss / len(train_loader))
        train_acc_history.append(train_correct / train_total)
        val_loss_history.append(val_loss / len(test_loader))
        val_acc_history.append(val_correct / val_total)
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}"
        )
        print(
            f"Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_correct/val_total:.4f}"
        )

    plot_training_metrics(
        train_loss_history, train_acc_history, val_loss_history, val_acc_history
    )
    return model


def plot_training_metrics(train_loss, train_acc, val_loss, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, "b-", label="Training Loss")
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    # 이건 뭐야??
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.tight_layout()
    plt.show()


def test_sentence_sarcasm(model, sentences, vocab, max_len, device, threshold=0.5):
    test_sentences = [
        "granny starting to fear spiders in the garden might be real",
        "game of thrones season finale showing this sunday night",
        "PyTorch book will be a best seller",
    ]
    # Preprocess
    sequences = texts_to_sequences(sentences, vocab)
    padded = pad_sequences(sequences, max_len)
    # print(padded)

    # Convert to tensor
    input_ids = torch.tensor(padded, dtype=torch.long).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        print(outputs)
        probabilities = outputs.squeeze().cpu().numpy()
        predictions = (probabilities >= threshold).astype(int)

    # Print results
    for sentence, prob, pred in zip(sentences, probabilities, predictions):
        print(f"\nText: {sentence}")
        print(f"Probability: {prob:.4f}")
        print(f"Classification: {'Sarcastic' if pred == 1 else 'Not Sarcastic'}")
        print(f"Confidence: {max(prob, 1-prob):.4f}")
        print("-" * 80)


def write_tensor_projector(dir, name, word_index, embedding_weights):
    # 출력할 벡터, 메타 파일 준비...
    v_file_name = dir + "/vecs_" + name + ".tsv"
    m_file_name = dir + "/meta_" + name + ".tsv"
    out_v = io.open(v_file_name, "w", encoding="utf-8")
    out_m = io.open(m_file_name, "w", encoding="utf-8")

    # 디코딩 사전 준비...
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    vocab_size = len(reverse_word_index)
    # 사전을 다 돌면서
    for word_num in range(1, vocab_size):
        # 나온 단어를 임베딩과 매칭...
        word = reverse_word_index[word_num]
        embeddings = embedding_weights[word_num]
        # 메타에는 단어를, 벡터에는 임베딩을 쓴다..
        out_m.write(word + "\n")
        out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()


def tokenize(text):
    # Tokenization logic, removing HTML and stopwords as discussed earlier
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def build_vocab(sentences, max_vocab_size=10000):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

    # Take only the top max_vocab_size-1 most frequent words (leave room for special tokens)
    most_common = counter.most_common(max_vocab_size - 2)  # -2 for <pad> and <unk>

    # Create vocabulary with indices starting from 2
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0  # Add padding token
    vocab["<unk>"] = 1  # Add unknown token
    return vocab


def texts_to_sequences(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in tokenize(sentence):
            # Use unknown token (1) for words not in vocabulary
            sequence.append(word_index.get(word, 1))
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences


def word_frequency(sentences, word_dict):
    frequency = {word: 0 for word in word_dict}

    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word in frequency:
                frequency[word] += 1

    return frequency


# GloVe 스타일 텍스트 처리 함수들...
def tokenize_glove_style(text):
    text = BeautifulSoup(text, "html.parser").get_text().lower()

    # 숫자는 0으로 바꾸는게 GloVe 스타일이라고...re.sub('패턴', '바꿀문자열', '대상문자열')
    text = re.sub(r"\d", "0", text)

    # 공백과 구둣점 나누기...GloVe는 구둣점도 토큰으로 관리...
    # .,!?() 찾아서 그 앞뒤로 공백...
    text = re.sub(r"([.,!?()])", r" \1 ", text)
    # space 2개 이상을 찾아서 공백 하나로...
    text = re.sub(r"\s{2,}", " ", text)

    return text.split()


def build_vocab_glove(sentences, max_vocab_size=10000):
    # 액셀 피봇처럼 토큰 별 빈도수 사전으로 만들고...
    counter = Counter()
    for text in sentences:
        counter.update(tokenize_glove_style(text))

    # 빈도 순으로 어휘 수 제한 - 2는 pad, unk
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    return vocab


# 여러 (문장)들을 받아서 (단어 ID 리스트)의 리스트로 반환
def texts_to_sequences_glove(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in tokenize_glove_style(sentence):
            sequence.append(word_index.get(word, 1))
        sequences.append(sequence)
    return sequences


# 문장들에 패딩 넣기...
def pad_sequences_glove(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences


# 문장들 받아서 그 안에서 단어사전에 있는 단어들에 대한 단어별 빈도수 반환...
def word_frequency_glove(sentences, vocab=None):
    # 뭔가 이건 위에 build_vocab이랑 똑 같은 코든데...
    counter = Counter()
    for sentence in sentences:
        tokens = tokenize_glove_style(sentence)
        counter.update(tokens)

    # 어휘 사전이 주어지면, 그 안에 있는 단어들만 남긴다...
    if vocab:
        counter = Counter(
            {word: count for word, count in counter.items() if word in vocab}
        )

    # 빈도와 알파벳 순으로 정렬해서 반환 - x[1]이 사전의 빈도수고, -는 내림차순...
    sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return sorted_words


stopwords = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "hed",
    "hes",
    "her",
    "here",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "hows",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "lets",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "nor",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "theres",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "were",
    "what",
    "whats",
    "when",
    "whens",
    "where",
    "wheres",
    "which",
    "while",
    "who",
    "whos",
    "whom",
    "why",
    "whys",
    "with",
    "would",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
]
