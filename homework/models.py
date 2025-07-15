"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F



class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return F.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # Input image is 3 channels (RGB), HxW.
        # For a linear classifier, we need to flatten the input image into a 1D vector.
        # The size of this vector will be 3 * H * W.
        input_dim = 3 * h * w
        # The linear layer maps the flattened input to the number of classes.
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor from (B, 3, H, W) to (B, 3*H*W)
        # The -1 in view means that dimension will be inferred.
        # So, x.view(x.size(0), -1) flattens all dimensions after the batch dimension.
        x = x.view(x.size(0), -1)
        # Pass the flattened tensor through the linear layer to get the logits.
        logits = self.linear(x)
        return logits


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 256, # Added a hidden_dim argument for flexibility

    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
    
        input_dim = 3 * h * w
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Input layer to hidden layer
            nn.ReLU(),                        # Activation function
            nn.Linear(hidden_dim, num_classes) # Hidden layer to output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        # Pass through the sequential model
        logits = self.model(x)
        return logits


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4, # At least 4 layers as per instructions
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_dim = 3 * h * w
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1): # num_layers includes the output layer
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """

        x = x.view(x.size(0), -1)
        logits = self.model(x)
        return logits

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4, 
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_dim = 3 * h * w
        
        # Initial linear layer to project input to hidden_dim
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Create a ModuleList for residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 1): # num_layers includes the initial and final layers
            # Each residual block consists of two linear layers with ReLU in between
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.residual_blocks.append(block)
        
        # Final linear layer to project to num_classes
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = x.view(x.size(0), -1) # Flatten the input

        # Pass through the initial layer
        x = self.initial_layer(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            # Add the input to the block's output (residual connection)
            # Apply ReLU after the addition as is common in ResNets
            x = F.relu(x + block(x))
        
        # Pass through the final output layer
        logits = self.output_layer(x)
        return logits


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
