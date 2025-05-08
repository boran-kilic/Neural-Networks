import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# Preprocessing
with open("names.txt", "r") as f:
    names = f.read().lower().splitlines()

all_chars = ['<EON>'] + [chr(i) for i in range(97, 123)]  # EON + a-z
char_to_idx = {ch: idx for idx, ch in enumerate(all_chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

max_len = 11
num_chars = len(all_chars)

def one_hot_encode(seq, max_len=11):
    if isinstance(seq, str):
        seq = list(seq)
    seq += ['<EON>'] * (max_len - len(seq))
    encoding = np.zeros((max_len, num_chars), dtype=np.float32)
    for i, ch in enumerate(seq):
        encoding[i, char_to_idx[ch]] = 1
    return encoding


X = []
Y = []

for name in names:
    name = name.strip()
    encoded = one_hot_encode(name, max_len)
    X.append(encoded)

    shifted_seq = list(name[1:])
    shifted_seq.append('<EON>')
    shifted_encoded = one_hot_encode(shifted_seq, max_len)
    Y.append(shifted_encoded)


X = torch.tensor(X)
Y = torch.tensor(Y)

# LSTM Model
class NameLSTM(nn.Module):
    def __init__(self):
        super(NameLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_chars, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_chars)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = NameLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training
losses = []
epochs = 2000

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.view(-1, num_chars), Y.view(-1, num_chars).argmax(dim=1))
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), f"0702-22103444-KILIC.pt")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig("training_loss.png")
plt.show()

