import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, lemmatize
from model import NeuralNet

# Load and preprocess data
def load_data(filename):
    with open(filename, 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['queries']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!', 'is', 'the', 'are', 'what', 'can']  # Add common stopwords
    all_words = [lemmatize(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words, tags, xy

# Create training data
def create_training_data(all_words, tags, xy):
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    return np.array(X_train), np.array(y_train)

# Define custom dataset
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = torch.Tensor(X).float()
        self.y_data = torch.Tensor(y).long()
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
def set_hyperparameters(X_train, tags):
    input_size = len(X_train[0])
    hidden_size = 16  # Increased hidden size
    output_size = len(tags)
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001

    return input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate

# Train the model
def train_model(dataset, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')
    return model

# Save the model
def save_model(model, input_size, hidden_size, output_size, all_words, tags, filename):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, filename)
    print(f'Training complete. Model saved to {filename}')

if __name__ == "__main__":
    all_words, tags, xy = load_data('intents.json')
    
    # Print number of queries
    print(f"Number of queries: {len(xy)}")
    
    # Print number of tags
    print(f"Number of tags: {len(tags)}")
    
    # Print unique lemmatized words
    print("\nUnique lemmatized Words:")
    print(" ".join(all_words))
    
    X_train, y_train = create_training_data(all_words, tags, xy)
    input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate = set_hyperparameters(X_train, tags)
    dataset = ChatDataset(X_train, y_train)
    model = train_model(dataset, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate)
    save_model(model, input_size, hidden_size, output_size, all_words, tags, 'data.pth')
