import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import streamlit as st

# Step 1: Load Dataset and Prepare Vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Load IMDB dataset
train_iter = IMDB(split='train')

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Function to preprocess text
def preprocess_text(text, vocab, tokenizer, max_len=500):
    tokens = tokenizer(text.lower())
    indices = [vocab[token] for token in tokens]
    return torch.tensor(indices[-max_len:])  # Truncate to the last `max_len` tokens

# Step 2: Define the LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return self.sigmoid(output)

# Step 3: Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(len(vocab), embed_dim=128, hidden_dim=256, output_dim=1).to(device)

# Load model state
try:
    model.load_state_dict(torch.load("rnn_sentiment.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    st.warning("Pretrained model not found. Please train and save the model as 'rnn_sentiment.pth'.")

# Step 4: Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.error('Please enter a valid movie review.')
    else:
        # Preprocess input
        input_tensor = preprocess_text(user_input, vocab, tokenizer).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (shape: [1, seq_len])

        # Convert to LongTensor for embedding layer
        input_tensor = input_tensor.long()

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
        sentiment = 'Positive' if prediction.item() > 0.5 else 'Negative'

        # Display results
        st.success(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction.item():.4f}')
else:
    st.write('Please enter a movie review.')
