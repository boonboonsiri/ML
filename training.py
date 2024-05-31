import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim

# Load data tensors
features_tensor = torch.load('data/v0_features_tensor.pt')
labels_tensor = torch.load('data/v0_labels_tensor.pt')
print(features_tensor.size(), labels_tensor.size())


# Create a TensorDataset and DataLoader
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the model with more layers
class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = 10
hidden_size1 = 64
hidden_size2 = 128
hidden_size3 = 64
output_size = 1  # Assuming binary classification
model = DeepNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500

losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_features, batch_labels in dataloader:
        # Forward pass
        outputs = model(batch_features)

        loss = criterion(outputs.squeeze(), batch_labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #epoch_loss /= len(dataloader)
    #losses.append(epoch_loss)
    losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot loss versus epoch
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()

# Print final model state (optional)
print("Training completed.")
