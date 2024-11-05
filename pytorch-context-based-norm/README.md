# Context-based Normalization with Pytorch


## References


- **All versions:** *Enhancing Neural Network Representations with Prior Knowledge-Based Normalization*, FAYE et al., [ArXiv Link](https://arxiv.org/abs/2403.16798)


## Installation

To install the Context-Based Normalization package with **Pytorch** via pip, use the following command::

```bash
pip install pytorch-context-based-norm
```

## Usage

### Generate Data

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create data
data = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25],
                 [26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35],
                 [36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45],
                 [46, 47, 48, 49, 50]])

X = torch.tensor(data, dtype=torch.float32)

# Create target (5 classes)
labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
Y = torch.tensor(labels, dtype=torch.long)


# Establishing contexts (3 contexts): ContextNorm employs indices as input for normalizing.
context_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
context_indices = torch.tensor(context_indices)
```


### Context Normalization


```python
from pytorch_context_based_norm import ContextNorm

# Apply normalization layer
context_layer = ContextNorm(num_contexts=3)

# Define the rest of your model architecture
# For example:
hidden_layer = nn.Linear(5, 10)
output_layer = nn.Linear(10, 10)

# Define the model
model = nn.Sequential(
    context_layer,
    nn.ReLU(),
    hidden_layer,
    nn.ReLU(),
    output_layer
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model([X, context_indices])
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
```

### Context Normalization Extended


```python
from pytorch_context_based_norm import ContextExtendedNorm

# Apply normalization layer
context_layer = ContextExtendedNorm(num_contexts=3, input_dim=X.shape[-1])

# Define the rest of your model architecture
# For example:
hidden_layer = nn.Linear(5, 10)
output_layer = nn.Linear(10, 10)

# Define the model
model = nn.Sequential(
    context_layer,
    nn.ReLU(),
    hidden_layer,
    nn.ReLU(),
    output_layer
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model([X, context_indices])
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
```

### Adaptive Context Normalization

This version doesn't require explicit prior information and adapts based on the input data distribution.

```python
from pytorch_context_based_norm import AdaptiveContextNorm

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.context_norm = AdaptiveContextNorm(num_contexts=3, input_dim=5)
        self.hidden_layer = nn.Linear(5, 10)
        self.output_layer = nn.Linear(10, 5)

    def forward(self, x):
        x = self.context_norm(x)
        x = F.relu(self.hidden_layer(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x

# Instantiate the model
model = MyModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```


This README provides an overview of the Context-Based Normalization package along with examples demonstrating the usage of different normalization layers. You can modify and extend these examples according to your specific requirements.
