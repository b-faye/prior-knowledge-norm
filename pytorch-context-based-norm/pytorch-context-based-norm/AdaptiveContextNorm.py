import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveContextNorm(nn.Module):
    def __init__(self, num_contexts, epsilon=1e-3, momentum=0.9):
        """
        Initialize the Adaptive Context-Based Normalization layer.

        :param num_contexts: The number of contexts for normalization.
        :param epsilon: A small positive value to prevent division by zero during normalization.
        :param momentum: The momentum for updating mean, variance, and prior during training.
        """
        super(AdaptiveContextNorm, self).__init__()
        self.num_contexts = num_contexts
        self.epsilon = epsilon
        self.momentum = momentum

        # Initialize mean, variance, and prior weights
        self.mean = nn.Parameter(torch.empty(num_contexts, 1))
        self.variance = nn.Parameter(torch.empty(num_contexts, 1))
        self.prior = nn.Parameter(torch.empty(num_contexts, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mean, -1.0, 1.0)
        nn.init.uniform_(self.variance, 0.001, 0.01)
        initial_prior = torch.rand_like(self.prior)
        self.prior.data = initial_prior / initial_prior.sum(dim=0, keepdim=True)

    def forward(self, x):
        # Determine input shape
        if x.ndim == 4:  # (Batch, Channel, H, W)
            batch_size, channels, height, width = x.shape
            input_dim = channels
            spatial_dims = (height, width)
        elif x.ndim == 3:  # (Batch, timestamp, dim)
            batch_size, timestamp, input_dim = x.shape
            spatial_dims = None
        elif x.ndim == 2:  # (Batch, dim)
            batch_size, input_dim = x.shape
            spatial_dims = None
        else:
            raise ValueError("Input shape not supported")

        # Expand mean, variance, and prior for batch processing
        mean = self.mean.expand(self.num_contexts, input_dim)
        var = F.softplus(self.variance).expand(self.num_contexts, input_dim)
        prior = F.softmax(self.prior, dim=0)

        # Prepare output tensor
        normalized_x = torch.zeros_like(x)

        for k in range(self.num_contexts):
            mean_k = mean[k].view(1, -1, 1, 1) if spatial_dims else mean[k]  # Reshape for 4D input
            var_k = var[k].view(1, -1, 1, 1) if spatial_dims else var[k]
            prior_k = prior[k]

            # Compute probabilities
            p_x_given_k = prior_k * torch.exp(-0.5 * ((x - mean_k) / (var_k + self.epsilon))**2)
            p_x_given_i = sum(prior[i] * torch.exp(-0.5 * ((x - mean[i].view(1, -1, 1, 1) if spatial_dims else mean[i]) /
                                                           (var[i].view(1, -1, 1, 1) if spatial_dims else var[i] + self.epsilon))**2)
                              for i in range(self.num_contexts))
            tau_k = p_x_given_k / (p_x_given_i + self.epsilon)

            if self.training:
                hat_tau_k = tau_k / (tau_k.sum(dim=tuple(range(x.ndim))[:-1], keepdim=True) + self.epsilon)
                expectation = (hat_tau_k * x).mean(dim=tuple(range(x.ndim))[1:-1], keepdim=True)
                v_i_k = x - expectation
                variance = (hat_tau_k * (v_i_k**2)).mean(dim=tuple(range(x.ndim))[1:-1], keepdim=True)
                hat_x_i_k = v_i_k / torch.sqrt(variance + self.epsilon)
                hat_x_i = (tau_k / torch.sqrt(prior_k + self.epsilon)) * hat_x_i_k
                normalized_x += hat_x_i

                # Update mean, variance, and prior
                updated_mean = hat_x_i.mean(dim=tuple(range(x.ndim))[:-1])
                updated_var = hat_x_i.var(dim=tuple(range(x.ndim))[:-1])
                updated_prior = tau_k.mean()

                self.mean[k].data = self.momentum * self.mean[k] + (1 - self.momentum) * updated_mean
                self.variance[k].data = self.momentum * self.variance[k] + (1 - self.momentum) * updated_var
                self.prior[k].data = self.momentum * self.prior[k] + (1 - self.momentum) * updated_prior

            else:
                hat_x_i_k = (x - mean_k) / torch.sqrt(var_k + self.epsilon)
                hat_x_i = (tau_k / torch.sqrt(prior_k + self.epsilon)) * hat_x_i_k
                normalized_x += hat_x_i

        return normalized_x

    def call(self, x):
        return self.forward(x)
