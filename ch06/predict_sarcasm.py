from helper import *
import json
import torch

with open("data/sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

table = str.maketrans("", "", string.punctuation)
for item in datastore:
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    # 책 소스에서는 따로 bs4, stopword 적용해서 하는데, 나는 책 helper에서 만들어 둔 함수 이용...
    soup = BeautifulSoup(sentence, "html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence += word + " "
    sentences.append(sentence)

# 훈련 테스트 데이터셋 만들기...
# max_length = 100
max_length = 85
training_size = 23000

# 훈련과 테스트 나누고
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

# 정답지 라벨은 바로 텐서로...
training_labels = labels[0:training_size]
training_labels = torch.tensor(training_labels, dtype=torch.float32)
testing_labels = labels[training_size:]
testing_labels = torch.tensor(testing_labels, dtype=torch.float32)

# 어휘 사전은 훈련에만 만들고 - helper의 함수를 수정해서 단어 수 제한해서 과대적합 줄이기...
vocab_size = 2000
word_index = build_vocab(training_sentences, vocab_size)
print(len(word_index))

test_sentences = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
    "PyTorch book will be a best seller",
]


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


# Example usage:
embedding_dim = int(vocab_size**0.25) + 1
hidden_dim = 8
model = TextClassificationModel(vocab_size, embedding_dim, hidden_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("sarcasm_model.pth", map_location=device))
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

out_v = io.open("my_vecs.tsv", "w", encoding="utf-8")
out_m = io.open("my_meta.tsv", "w", encoding="utf-8")
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = embedding_weights[word_num]
    out_m.write(word + "\n")
    out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
