import keras
from keras import utils
import tensorflow as tf

@utils.register_keras_serializable()
class ContextNorm(keras.layers.Layer):
    def __init__(self,
                 num_contexts,
                 priors=None,
                 axis=-1,
                 momentum=0.99,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 synchronized=False,
                 **kwargs):
        super(ContextNorm, self).__init__(**kwargs)
        self.num_contexts = num_contexts
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.synchronized = synchronized
        self.priors = priors

    def build(self, input_shape):
        self.context_norm_layers = []
        for i in range(self.num_contexts):
            self.context_norm_layers.append(keras.layers.BatchNormalization(
                axis=self.axis,
                momentum=self.momentum,
                epsilon=self.epsilon,
                center=self.center,
                scale=self.scale,
                beta_initializer=self.beta_initializer,
                gamma_initializer=self.gamma_initializer,
                moving_mean_initializer=self.moving_mean_initializer,
                moving_variance_initializer=self.moving_variance_initializer,
                beta_regularizer=self.beta_regularizer,
                gamma_regularizer=self.gamma_regularizer,
                beta_constraint=self.beta_constraint,
                gamma_constraint=self.gamma_constraint,
                trainable=True,
                name=f"context_norm_layer_{i}"
            ))
        if self.priors is None:
            self.priors = [1. / self.num_contexts] * self.num_contexts
        else:
            assert len(self.priors) == self.num_contexts, "Number of priors must match the number of groups."
        self.priors = tf.convert_to_tensor(self.priors, dtype=tf.float32)
        super(ContextNorm, self).build(input_shape)

    def call(self, inputs, training=False):
        samples, contexts = inputs
        contexts = contexts[:, 0]

        # During training, apply prior on normalized data
        if training:
            for i in range(self.num_contexts):
                indices = tf.where(tf.equal(contexts, i))
                group_data = tf.gather(samples, indices[:, 0])

                def update_samples():
                    normalized_group_data = self.context_norm_layers[i](group_data, training=True)
                    normalized_group_data *= (1. / tf.sqrt(self.priors[i]))
                    return tf.tensor_scatter_nd_update(samples, indices, normalized_group_data)

                samples = tf.cond(
                    tf.shape(indices)[0] > 0,  # Check if there are elements in this context
                    update_samples,
                    lambda: samples  # No update if no elements match the context
                )

            return samples

        # Inference
        else:
            for i in range(self.num_contexts):
                indices = tf.where(tf.equal(contexts, i))
                group_data = tf.gather(samples, indices[:, 0])

                def update_samples_inference():
                    normalized_group_data = (1. / tf.sqrt(self.priors[i])) * self.context_norm_layers[i](group_data, training=False)
                    return tf.tensor_scatter_nd_update(samples, indices, normalized_group_data)

                samples = tf.cond(
                    tf.shape(indices)[0] > 0,
                    update_samples_inference,
                    lambda: samples
                )

            return samples

    def get_config(self):
        config = {
            'num_contexts': self.num_contexts,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': self.beta_initializer,
            'gamma_initializer': self.gamma_initializer,
            'moving_mean_initializer': self.moving_mean_initializer,
            'moving_variance_initializer': self.moving_variance_initializer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_constraint': self.beta_constraint,
            'gamma_constraint': self.gamma_constraint,
            'synchronized': self.synchronized,
        }
        base_config = super(ContextNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

