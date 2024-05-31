import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim

# Load data tensors
features_tensor = torch.load('data/v0_features_tensor.pt')
labels_tensor = torch.load('data/v0_labels_tensor.pt')

# Define PyTorch dataset
dataset = TensorDataset(features_tensor, labels_tensor)

# Define your PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate model
model = MyModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define number of folds for cross-validation
num_folds = 5
num_epochs = 20
kf = KFold(n_splits=num_folds)

# Iterate over folds
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Create DataLoader for training and validation
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)


    # Train model
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

    print(f"Validation Loss: {val_loss.item()}")

# Compute average performance metrics across folds if needed
