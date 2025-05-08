import os
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

seed = 16
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

zip_path = './geometry_dataset.zip'
extract_dir = './geometry_dataset'
output_dir = os.path.join(extract_dir, 'output')
classes = ['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f'Extracting {zip_path} to {extract_dir}...')
        for file in tqdm(file_list, desc='Extracting files', ncols=100):
            zip_ref.extract(file, extract_dir)
        print(f'Extraction completed: {zip_path} to {extract_dir}')
else:
    print(f'{extract_dir} already exists. Skipping extraction.')

train_images, train_labels = [], []
test_images, test_labels = [], []

if not (os.path.exists('training.file') and os.path.exists('testing.file')):
    for cls in classes:
        images = sorted([img for img in os.listdir(output_dir) if img.startswith(cls) and img.endswith('.png')])

        train_files = images[:8000]
        test_files = images[8000:10000]

        for file in tqdm(train_files, desc=f'Training - {cls}', ncols=100):
            img_path = os.path.join(output_dir, file)
            img = Image.open(img_path)
            train_images.append(np.array(img))
            train_labels.append(cls)

        for file in tqdm(test_files, desc=f'Testing - {cls}', ncols=100):
            img_path = os.path.join(output_dir, file)
            img = Image.open(img_path)
            test_images.append(np.array(img))
            test_labels.append(cls)

    with open('training.file', 'wb') as f:
        pickle.dump((train_images, train_labels), f)

    with open('testing.file', 'wb') as f:
        pickle.dump((test_images, test_labels), f)

    print('Dataset split completed and saved successfully.')

elif os.path.exists('training.file') and os.path.exists('testing.file'):    
    print('training.file and testing.file already exist. Skipping dataset split.')
    train_images, train_labels = pickle.load(open('training.file', 'rb'))
    test_images, test_labels = pickle.load(open('testing.file', 'rb'))
    print('Loaded existing dataset.')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 25 * 25, 256) 
        self.fc2 = nn.Linear(256, 9)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)        
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

class GeometryDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = [class_to_idx[label] for label in labels] 
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = GeometryDataset(train_images, train_labels, transform=transform)
test_dataset = GeometryDataset(test_images, test_labels, transform=transform)

debug_portion = 1.0  

num_train_samples = int(len(train_dataset) * debug_portion)
train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
train_subset = Subset(train_dataset, train_indices)

num_test_samples = int(len(test_dataset) * debug_portion)
test_indices = torch.randperm(len(test_dataset))[:num_test_samples]
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss/(batch_idx+1):.6f}, Accuracy: {100. * correct / ((batch_idx+1) * len(data)):.2f}%')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f'Epoch {epoch} Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.2f}%')

def test():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_losses.append(avg_loss)
    test_accuracies.append(accuracy)
    print(f'Test Loss: {avg_loss:.6f}, Test Accuracy: {accuracy:.2f}%')

for epoch in range(1, 16):
    train(epoch)
    test()
    scheduler.step()

torch.save(model.state_dict(), "0602-22103444-KILIC.pt")

plt.figure(figsize=(12, 6))

epochs = range(1, len(train_losses) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Epochs vs Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
