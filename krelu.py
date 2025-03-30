import torch
import torch.nn as nn

# Base KReLU class
class KReLU(nn.Module):
    """
    KReLU: A parameterized ReLU activation function.
    f(x) = x + k * (-min(0, x)), where k controls the negative slope.
    """
    def __init__(self, k=0.5):
        """
        Args:
            k (float or nn.Parameter): Slope for negative inputs. Default: 0.5
        """
        super(KReLU, self).__init__()
        # If k is a float, make it a parameter for learning
        self.k = nn.Parameter(torch.tensor(k)) if isinstance(k, (float, int)) else k

    def forward(self, x):
        return x + self.k * (-torch.min(x, torch.zeros_like(x)))


# KReLU with freezing mechanism for learnable k
class KReLUFreezable(nn.Module):
    """
    KReLU with a freezing mechanism: stops updating k when it stabilizes.
    Useful for dynamic k optimization.
    """
    def __init__(self, k_init=0.5, threshold=1e-4, max_steps=100):
        """
        Args:
            k_init (float): Initial k value. Default: 0.5
            threshold (float): Gradient threshold to freeze k. Default: 1e-4
            max_steps (int): Max steps with small gradient before freezing. Default: 100
        """
        super(KReLUFreezable, self).__init__()
        self.k = nn.Parameter(torch.tensor(k_init))
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps_unchanged = 0
        self.frozen = False

    def forward(self, x):
        return x + self.k * (-torch.min(x, torch.zeros_like(x)))

    def check_and_freeze(self):
        """Freeze k if its gradient is small for max_steps."""
        if self.frozen or self.k.grad is None:
            return
        if torch.abs(self.k.grad) < self.threshold:
            self.steps_unchanged += 1
            if self.steps_unchanged >= self.max_steps:
                self.frozen = True
                self.k.requires_grad = False
                print(f"KReLU frozen at k={self.k.item():.4f}")
        else:
            self.steps_unchanged = 0


# Example usage in a simple MLP
class SimpleMLP(nn.Module):
    """MLP with different KReLU usage examples."""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        # Example 1: Fixed k (global)
        self.act1 = KReLU(k=0.35)  # Static k for all layers
        self.fc2 = nn.Linear(512, 256)
        # Example 2: Learnable k with freezing
        self.act2 = KReLUFreezable(k_init=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# Example usage with grouped k in a CNN
class SimpleCNN(nn.Module):
    """CNN with grouped KReLU (different k for different layer groups)."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Group 1: Early layers (low k for sparsity)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.act_group1 = KReLU(k=0.35)
        # Group 2: Middle layers (moderate k)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.act_group2 = KReLU(k=0.5)
        # Group 3: Late layers (high k for gradients)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.act_group3 = KReLU(k=0.7)
        self.fc = nn.Linear(64 * 26 * 26, 10)  # Assuming 32x32 input

    def forward(self, x):
        x = self.act_group1(self.conv1(x))
        x = self.act_group2(self.conv2(x))
        x = self.act_group3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Example training loop to demonstrate usage
if __name__ == "__main__":
    # Simple MLP with fixed and learnable k
    model_mlp = SimpleMLP()
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Fake data (e.g., MNIST-like)
    inputs = torch.randn(64, 784)
    targets = torch.randint(0, 10, (64,))

    # Training loop
    for step in range(5):  # Short demo
        optimizer.zero_grad()
        outputs = model_mlp(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Check and freeze learnable k
        model_mlp.act2.check_and_freeze()
        
        optimizer.step()
        print(f"Step {step+1}, Loss: {loss.item():.4f}, k2: {model_mlp.act2.k.item():.4f}")

    # Simple CNN with grouped k
    model_cnn = SimpleCNN()
    print("\nGrouped KReLU in CNN:")
    print(f"Group 1 k: {model_cnn.act_group1.k.item():.4f}")
    print(f"Group 2 k: {model_cnn.act_group2.k.item():.4f}")
    print(f"Group 3 k: {model_cnn.act_group3.k.item():.4f}")
