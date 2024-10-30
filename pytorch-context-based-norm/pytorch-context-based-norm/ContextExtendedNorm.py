import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextExtendedNorm(nn.Module):
    def __init__(self, num_contexts, input_dim, epsilon=1e-3):
        """
        Initialize the Context Extended Normalization layer.

        Parameters:
        :param num_contexts: The number of contexts (prior knowledge)
        :param input_dim: The input dimension
        :param epsilon: A small positive value to prevent division by zero during normalization.
        """
        super(ContextExtendedNorm, self).__init__()
        self.num_contexts = num_contexts
        self.epsilon = epsilon
        self.input_dim = input_dim

        # Define initial mean and standard deviation as learnable parameters
        self.initial_mean = nn.Parameter(torch.Tensor(num_contexts, self.input_dim))
        self.initial_std = nn.Parameter(torch.Tensor(num_contexts, self.input_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.initial_mean)
        nn.init.xavier_uniform_(self.initial_std)

    def forward(self, inputs):
        """
        Apply the Context Extended Normalization to the input data.

        :param inputs: A tuple of (x, context_id) where x is the data to be normalized, and context_id is the context identifier. Context identifier must be in int32 format.

        :return normalized_x: The normalized output data.
        """
        x, context_id = inputs

        # Extract context indices from context_id
        indices = context_id

        # Gather initial mean and standard deviation based on context indices
        mean = self.initial_mean[indices]
        std = self.initial_std[indices]

        # Ensure standard deviation is positive
        std = torch.exp(std)

        # Determine the number of dimensions to expand
        num_expand_dims = x.dim() - 2

        # Expand mean and std dimensions accordingly
        for _ in range(num_expand_dims):
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)

        # Perform normalization
        normalized_x = (x - mean) / (std + self.epsilon)

        return normalized_x

    def call(self, inputs):
        return self.forward(inputs)

