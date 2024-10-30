import tensorflow as tf
from tensorflow_context_based_norm import ContextNorm, ContextExtendedNorm, AdaptiveContextNorm


#### Create data
data = [[1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45],
        [46, 47, 48, 49, 50]]

X = tf.constant(data)

# Create target (5 classes)
labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  
Y = tf.constant(labels)

# Create cluster (3 clusters)
context_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] 


################################# Model with ContextNorm layer ##############################

context_indices = tf.constant(context_indices, shape=(10,1), dtype=tf.int32)
# Define input shapes
X_shape = (10, 5)  

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  
context_input = tf.keras.Input(shape=(1, ), dtype=tf.int32) 

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(X_input)

# Apply normalization layer
normalized_activation = ContextNorm(num_contexts=3)([hidden_layer, context_input])

output_layer = tf.keras.layers.Dense(units=5, activation='softmax')(normalized_activation)

# Define the model
model = tf.keras.Model(inputs=[X_input, context_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit([X, context_indices], Y, epochs=10)


######################### Model with ContextExtendedNorm layer   ############################################

context_indices = tf.constant(context_indices, shape=(10,1), dtype=tf.int32)

# Define input shapes
X_shape = (10, 5)  

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  
context_input = tf.keras.Input(shape=(1, ), dtype=tf.int32) 

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(X_input)

# Apply normalization layer
normalized_activation = ContextExtendedNorm(num_contexts=3)([hidden_layer, context_input])

output_layer = tf.keras.layers.Dense(units=5, activation='softmax')(normalized_activation)

# Define the model
model = tf.keras.Model(inputs=[X_input, context_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit([X, context_indices], Y, epochs=10)



################### Model with AdaptiveContextNorm layer (don't need prior knowledge)  #######################


# Define input shapes
X_shape = (10, 5)  

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  

# Apply normalization layer
normalized_X = AdaptiveContextNorm(num_contexts=3)(X_input)

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(normalized_X)
output_layer = tf.keras.layers.Dense(units=5, activation='softmax')(hidden_layer)

# Define the model
model = tf.keras.Model(inputs=X_input, outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X, Y, epochs=10)

