import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

# Vocabulary builder and other functions (reuse from main.py)
from main import vocab, tokenizer, preprocess_text, SentimentLSTM, device

# Load dataset
train_iter = IMDB(split='train')
train_data = [(label, preprocess_text(text, vocab, tokenizer)) for label, text in train_iter]

# Convert labels to tensor
def label_to_tensor(label):
    return torch.tensor(1.0 if label == "pos" else 0.0)

train_data = [(label_to_tensor(label), text) for label, text in train_data]

# Split data into training and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# DataLoader
def collate_fn(batch):
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[0] for item in batch])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, labels

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize model
model = SentimentLSTM(len(vocab), embed_dim=128, hidden_dim=256, output_dim=1).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
best_val_loss = float('inf')
for epoch in range(5):
    model.train()
    train_loss = 0.0
    for batch_text, batch_labels in train_loader:
        batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        predictions = model(batch_text).squeeze(1)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_text, batch_labels in val_loader:
            batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
            predictions = model(batch_text).squeeze(1)
            loss = criterion(predictions, batch_labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "rnn_sentiment.pth")
        print(f"Model saved at epoch {epoch+1} with Validation Loss: {val_loss:.4f}")

print("Training complete. Best model saved as rnn_sentiment.pth.")
