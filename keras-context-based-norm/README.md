# Context-based Normalization with Keras


## References


- **All versions:** *Enhancing Neural Network Representations with Prior Knowledge-Based Normalization*, FAYE et al., [ArXiv Link](https://arxiv.org/abs/2403.16798)


## Installation

To install the Context-Based Normalization package with **Keras** via pip, use the following command::

```bash
pip install keras-context-based-norm
```

## Usage

### Generate Data

```python
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

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

X = data

# Create target (5 classes)
labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
Y = tf.constant(labels)

# Establishing clusters (3 contexts): ContextNorm employs indices as input for normalizing.
context_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
context_indices = tf.constant(context_indices, shape=(10, 1), dtype="int32")
```


### Context Normalization


```python
from keras_context_based_norm import ContextNorm

context_indices = tf.constant(context_indices, shape=(10,1), dtype=tf.int32)

# Define input shapes
X_shape = (10, 5)

# Define inputs
X_input = keras.Input(shape=X_shape[1:])
context_input = keras.Input(shape=(1, ), dtype=tf.int32)

# Define the rest of your model architecture
# For example:
hidden_layer = keras.layers.Dense(units=10, activation='relu')(X_input)

# Apply normalization layer
normalized_activation = ContextNorm(num_contexts=3)([hidden_layer, context_input])

output_layer = keras.layers.Dense(units=5, activation='softmax')(normalized_activation)

# Define the model
model = keras.Model(inputs=[X_input, context_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit([X, context_indices], Y, epochs=10)
```

### Context Normalization Extended


```python
from keras_context_based_norm import ContextExtendedNorm

# Define input shapes
X_shape = (10, 5)

# Define inputs
X_input = keras.Input(shape=X_shape[1:])
context_input = keras.Input(shape=(1, ), dtype=tf.int32)

# Define the rest of your model architecture
# For example:
hidden_layer = keras.layers.Dense(units=10, activation='relu')(X_input)

# Apply normalization layer
normalized_activation = ContextExtendedNorm(num_contexts=3)([hidden_layer, context_input])

output_layer = keras.layers.Dense(units=5, activation='softmax')(normalized_activation)

# Define the model
model = keras.Model(inputs=[X_input, context_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit([X, context_indices], Y, epochs=10)
```

### Adaptive Context Normalization

This version doesn't require explicit prior information and adapts based on the input data distribution.

```python
from keras_context_based_norm import AdaptiveContextNorm

# Define input shapes
X_shape = (10, 5)

# Define inputs
X_input = keras.Input(shape=X_shape[1:])

# Apply normalization layer
normalized_X = AdaptiveContextNorm(num_contexts=3)(X_input)

# Define the rest of your model architecture
# For example:
hidden_layer = keras.layers.Dense(units=10, activation='relu')(normalized_X)
output_layer = keras.layers.Dense(units=5, activation='softmax')(hidden_layer)

# Define the model
model = keras.Model(inputs=X_input, outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X, Y, epochs=10)
```


This README provides an overview of the Context-Based Normalization package along with examples demonstrating the usage of different normalization layers. You can modify and extend these examples according to your specific requirements.
