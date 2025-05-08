import torch
import torch.nn as nn

class NameLSTM(nn.Module):
    def __init__(self):
        super(NameLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=27, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 27)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

all_chars = ['<EON>'] + [chr(i) for i in range(97, 123)]
char_to_idx = {ch: idx for idx, ch in enumerate(all_chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

class NameGenerator:
    def __init__(self, model, temperature=2.5):
        self.model = model
        self.temperature = temperature
        self.model.eval()
    def sample(self, logits):
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def generate(self, initial, n=20):
        names = []
        initial_idx = char_to_idx[initial]
        max_len = 11

        for _ in range(n):
            input_seq = torch.zeros(1, max_len, 27)
            input_seq[0, 0, initial_idx] = 1
            name = initial

            hidden = None
            for i in range(1, max_len):
                output, hidden = self.model.lstm(input_seq[:, :i, :], hidden)
                logits = self.model.fc(output[:, -1, :])
                logits[:, char_to_idx['<EON>']] -= 10.0  # Avoid too short names  
                next_idx = self.sample(logits)
                if idx_to_char[next_idx] == '<EON>':
                    break
                name += idx_to_char[next_idx]
                input_seq[0, i, next_idx] = 1
            names.append(name)
        return names
    
    

model = NameLSTM()
model.load_state_dict(torch.load(f"0702-22103444-KILIC.pt"))

gen = NameGenerator(model)
while True:
    initial = input("Enter initial letter (or 'quit' to exit): ").lower()
    if initial == 'quit':
        break
    if initial not in char_to_idx:
        print("Invalid input. Please enter a valid letter.")
        continue
    print(gen.generate(initial))
