import torch
import torch.nn as nn

class ContextNorm(nn.Module):
    def __init__(self,
                 num_contexts,
                 priors=None,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True):
        super(ContextNorm, self).__init__()
        self.num_contexts = num_contexts
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        # Initialize priors
        if priors is None:
            self.priors = torch.ones(num_contexts) / num_contexts
        else:
            assert len(priors) == num_contexts, "Number of priors must match the number of contexts."
            self.priors = torch.tensor(priors, dtype=torch.float32)

        # Placeholder for the context-specific batch norm layers
        self.context_norm_layers = nn.ModuleList()

    def _get_norm_layer(self, input_shape):
        """Helper to return the appropriate batch normalization layer based on input shape."""
        if len(input_shape) == 2 or len(input_shape) == 3:
            return nn.BatchNorm1d(input_shape[1], eps=self.epsilon, momentum=self.momentum, affine=self.center or self.scale)
        elif len(input_shape) == 4:
            return nn.BatchNorm2d(input_shape[1], eps=self.epsilon, momentum=self.momentum, affine=self.center or self.scale)
        elif len(input_shape) == 5:
            return nn.BatchNorm3d(input_shape[1], eps=self.epsilon, momentum=self.momentum, affine=self.center or self.scale)
        else:
            raise ValueError(f"Unsupported input shape {input_shape}. Expected 2D, 3D, 4D, or 5D tensor.")

    def build(self, input_shape):
        """Initialize context-specific normalization layers based on input shape."""
        for _ in range(self.num_contexts):
            norm_layer = self._get_norm_layer(input_shape)
            self.context_norm_layers.append(norm_layer)

    def forward(self, inputs):
        samples, contexts = inputs

        # If `build` hasn't been called yet, infer shape from inputs
        if not self.context_norm_layers:
            self.build(samples.shape)

        # Apply context-specific normalization based on priors
        for i in range(self.num_contexts):
            indices = (contexts == i).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue  # Skip if no samples belong to this context

            group_data = samples[indices]

            # Use self.training to control behavior in training or evaluation mode
            normalized_group_data = self.context_norm_layers[i](group_data)
            if self.training:
                normalized_group_data *= (1. / torch.sqrt(self.priors[i]))
            else:
                with torch.no_grad():
                    normalized_group_data *= (1. / torch.sqrt(self.priors[i]))

            # Scatter the normalized data back to the correct indices in the original tensor
            samples = samples.index_copy(0, indices, normalized_group_data)

        return samples

    def extra_repr(self):
        return (f'num_contexts={self.num_contexts}, axis={self.axis}, momentum={self.momentum}, '
                f'epsilon={self.epsilon}, center={self.center}, scale={self.scale}')

