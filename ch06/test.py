# 코드 잘못된 곳을 못찾겠는데, 책처럼 안되서 그냥 책 코드 배껴 넣음...
from collections import Counter
from collections import Counter
from typing import List, Dict, Tuple, Union


# Assuming the tokenize function is defined elsewhere
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


import json


from bs4 import BeautifulSoup
import string

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

table = str.maketrans("", "", string.punctuation)

with open("data/sarcasm.json", "r") as f:
    datastore = json.load(f)


sentences = []
labels = []
urls = []
for item in datastore:
    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
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

xs = []
ys = []
current_item = 1
for item in sentences:
    xs.append(current_item)
    current_item = current_item + 1
    ys.append(len(item))
newys = sorted(ys)

import matplotlib.pyplot as plt

# plt.plot(xs, newys)
# plt.axis([20000, 27000, 50, 250])
# plt.show()

print(newys[20000])

vocab_size = 2000
embedding_dim = 7
max_length = 85
training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Updated for vocab size limitation
word_index = build_vocab(training_sentences, max_vocab_size=vocab_size)
training_sequences = texts_to_sequences(training_sentences, word_index)
training_padded = pad_sequences(training_sequences, max_len=max_length)


testing_sequences = texts_to_sequences(testing_sentences, word_index)
testing_padded = pad_sequences(testing_sequences, max_len=max_length)

word_freq = word_frequency(training_sentences, word_index)
print(word_freq)

import torch
import torch.nn as nn
import torch.optim as optim


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=8, dropout_rate=0.25):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Change for pooling layer
        x = self.global_pool(x).squeeze(2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


# Assuming vocab_size and embedding_dim are defined
vocab_size = len(word_index)  # Example value

# Create the model
model = TextClassificationModel(vocab_size, embedding_dim)


# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=False)

# Print model summary
print(model)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Assuming your TextClassificationModel is already defined

# Convert your preprocessed data to PyTorch tensors
training_padded = torch.tensor(training_padded, dtype=torch.long)
testing_padded = torch.tensor(testing_padded, dtype=torch.long)
training_labels = torch.tensor(training_labels, dtype=torch.float32)
testing_labels = torch.tensor(testing_labels, dtype=torch.float32)

# Create DataLoader objects
batch_size = 32
train_dataset = TensorDataset(training_padded, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(testing_padded, testing_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Training loop
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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

    # Validation
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

# After training, you can save the model
torch.save(model.state_dict(), "text_classification_model.pth")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(train_loss, train_acc, val_loss, val_acc):
    """
    Plot training and validation metrics on two subplots.

    Args:
        train_loss: Array of training loss values
        train_acc: Array of training accuracy values
        val_loss: Array of validation loss values
        val_acc: Array of validation accuracy values
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot epochs on x-axis
    epochs = range(1, len(train_loss) + 1)

    # First subplot: Loss
    ax1.plot(epochs, train_loss, "b-", label="Training Loss")
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Accuracy
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Add accuracy percentage labels
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig


# Usage example:
plot_training_metrics(
    train_loss_history, train_acc_history, val_loss_history, val_acc_history
)
plt.show()


def predict_sentences(model, sentences, vocab, max_len, device="cuda", threshold=0.5):
    """
    Make predictions for new sentences and interpret results
    """
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


test_sentences = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
    "PyTorch book will be a best seller",
]

# Example usage:
model = model.to(device)
predict_sentences(
    model=model,
    sentences=test_sentences,
    vocab=word_index,
    max_len=85,
    threshold=0.5,  # Adjust this threshold if needed
)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Get the embedding weights as a numpy array
embedding_weights = model.embedding.weight.data.cpu().numpy()
print(embedding_weights.shape)


# If you want to see the embedding for a specific word:
def get_word_embedding(word, vocab, model):
    if word in vocab:
        word_idx = vocab[word]
        return model.embedding.weight.data[word_idx].cpu().numpy()
    else:
        return None


# Example usage:
word = "game"
embedding = get_word_embedding(word, word_index, model)
if embedding is not None:
    print(f"Embedding for '{word}': {embedding}")
    print(f"Embedding dimension: {len(embedding)}")
else:
    print(f"Word '{word}' not found in vocabulary")

# If you want to save all embeddings with their corresponding words:
word_embeddings = {}
for word, idx in word_index.items():
    word_embeddings[word] = model.embedding.weight.data[idx].cpu().numpy()

print(reverse_word_index[2])
print(embedding_weights[2])

import io

out_v = io.open("vecs.tsv", "w", encoding="utf-8")
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = embedding_weights[word_num]
    out_m.write(word + "\n")
    out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
