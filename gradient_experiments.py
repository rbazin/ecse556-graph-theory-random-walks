import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from joblib import dump


class CustomNN(nn.Module):
    def __init__(self, layer_sizes, activation_func):
        super(CustomNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_func)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        test_losses.append(avg_loss)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}"
        )

    return train_losses, test_losses


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
epochs = 5

network_configs = [
    (784, *[64] * 50, 10),
    (784, *[64] * 20, 10),
    (784, *[64] * 10, 10),
    (784, *[64] * 5, 10),
]

activation_functions = [nn.LeakyReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Sigmoid()]

loss_histories = {}

for config, activation in zip(network_configs, activation_functions):
    model = CustomNN(config, activation)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_name = f"{activation.__class__.__name__}_{len(config)-2}_layers"
    print(f"\nTraining model: {model_name}")

    try:
        train_loss, test_loss = train_model(
            model, train_loader, test_loader, criterion, optimizer, epochs
        )
        loss_histories[model_name] = {"train_loss": train_loss, "test_loss": test_loss}
    except RuntimeError as e:
        print(f"There was an error during training of model {model_name}: {e}")
        loss_histories[model_name] = {"train_loss": "Error", "test_loss": "Error"}

dump(loss_histories, "loss_histories.joblib")
