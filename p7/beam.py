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
    def __init__(self, model, temperature=1.5):
        self.model = model
        self.temperature = temperature
        self.model.eval()
    def sample(self, logits):
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def generate(self, initial, n=20, beam_width=20):
        names = []
        initial_idx = char_to_idx[initial]
        max_len = 11

        for _ in range(n):
            # Initialize the beam with the initial character
            beam = [(initial, 0)]  # Each element is (sequence, score)

            for _ in range(max_len - 1):
                all_candidates = []
                for seq, score in beam:
                    # Stop expanding if the sequence already ends with <EON>
                    if seq.endswith('<EON>'):
                        all_candidates.append((seq, score))
                        continue
                    # Prepare input sequence
                    input_seq = torch.zeros(1, max_len, 27)
                    for i, ch in enumerate(seq):
                        input_seq[0, i, char_to_idx[ch]] = 1

                    # Get logits for the next character
                    with torch.no_grad():
                        output, _ = self.model.lstm(input_seq[:, :len(seq), :])
                        logits = self.model.fc(output[:, -1, :])

                    # Apply temperature and softmax
                    logits = logits / self.temperature
                    probs = torch.softmax(logits, dim=-1).squeeze()

                    # Expand the beam with all possible next characters
                    for idx, prob in enumerate(probs):
                        next_char = idx_to_char[idx]
                        new_seq = seq + next_char
                        new_score = score + torch.log(prob).item()  # Add log probability
                        all_candidates.append((new_seq, new_score))

                # Sample from the top `beam_width` candidates
                all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                probabilities = torch.softmax(torch.tensor([score for _, score in all_candidates]), dim=0)
                sampled_indices = torch.multinomial(probabilities, beam_width, replacement=True)
                beam = [all_candidates[i] for i in sampled_indices]

            # Add the best sequence from the beam to the names list
            best_name = max(beam, key=lambda x: x[1])[0]
            # Remove <EON> token if present
            best_name = best_name.replace('<EON>', '')
            names.append(best_name)

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
