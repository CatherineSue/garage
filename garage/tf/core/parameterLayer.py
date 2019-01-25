"""Parameter layer in TensorFlow."""

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import broadcast_to
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.python.framework import tensor_shape

# flake8: noqa


class ParameterLayer(KerasLayer):
    def __init__(self, length, initializer='ones', trainable=True, **kwargs):
        self.length = length
        self.initializer = initializer
        self.trainable = trainable
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.length, ),
            initializer=self.initializer,
            trainable=self.trainable)
        super().build(input_shape)

    def call(self, x):
        broadcast_shape = tf.concat(
            axis=0, values=[tf.shape(x)[:-1], [self.length]])
        return broadcast_to(self.kernel, shape=broadcast_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.length)

    def get_config(self):
        config = {
            'length': self.length,
            'initializer': self.initializer,
            'trainable': self.trainable
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
