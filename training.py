import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Load data tensors
features_tensor = torch.load('data/v0_features_tensor.pt')
labels_tensor = torch.load('data/v0_labels_tensor.pt')
print(features_tensor.size(), labels_tensor.size())




# Split the data into training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features_tensor, labels_tensor, test_size=0.4, random_state=42)
fe = [train_features, test_features, train_labels, test_labels ]
for f in fe:
    print(f.size())

# Create TensorDatasets
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model with more layers
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = 10
hidden_size1 = 64
hidden_size2 = 64
output_size = 1  # Single output for binary classification
model = DeepNN(input_size, hidden_size1, hidden_size2, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_dataloader:
        # Forward pass
        outputs = model(batch_features)
        outputs = outputs.squeeze()  # Remove extra dimension
        loss = criterion(outputs, batch_labels.float())
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_dataloader:
            outputs = model(batch_features)
            outputs = outputs.squeeze()  # Remove extra dimension
            loss = criterion(outputs, batch_labels.float())
            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plot loss versus epoch
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

# Print final model state (optional)
print("Training completed.")
