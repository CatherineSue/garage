"""MLP Layer based on tf.keras.layer."""
import numpy as np
import pickle
import tensorflow as tf
from garage.tf.models import GaussianMLPModel2
from garage.tf.core.mlp2 import mlp2
from tests.fixtures import TfGraphTestCase
from tensorflow.keras.layers import Input
import unittest
from garage.tf.models import PickableModel
from garage.tf.core.parameterLayer import ParameterLayer
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.python.framework import tensor_shape

# flake8: noqa
# pylint: noqa


class CustomLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[-1].value, self.output_dim],
            initializer='zero',
            trainable=True,
            dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return tf.keras.backend.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class box(PickableModel):
    def __init__(self, input_var):
        self.model = mlp2(
            input_var=input_var, output_dim=2, hidden_sizes=(4, 4))


class TestKerasModel(TfGraphTestCase):
    @unittest.skip
    def test_custom_layer_pickable(self):
        model = Sequential()
        model.add(Dense(5, input_dim=2))
        model.add(CustomLayer(5))

        print("\n### Keras custom model : {}\n".format(model.get_config()))

        print("\nJSON : {}\n".format(model.to_json()))
        fresh_model = tf.keras.models.model_from_json(
            model.to_json(), custom_objects={'CustomLayer': CustomLayer})
        print("\nRestore from JSON : {}\n".format(fresh_model.get_config()))

    # This works
    @unittest.skip
    def test_parameter_json(self):
        input_var = Input(shape=(5, ))
        parameter = ParameterLayer(length=2)(input_var)
        model = Model(inputs=input_var, outputs=parameter)

        self.sess.run(tf.global_variables_initializer())

        fresh_model = tf.keras.models.model_from_json(
            model.to_json(), custom_objects={'ParameterLayer': ParameterLayer})
        fresh_model.set_weights(model.get_weights())

        y = self.sess.run(
            model.output, feed_dict={model.input: np.random.random((2, 5))})
        yy = self.sess.run(
            fresh_model.output,
            feed_dict={fresh_model.input: np.random.random((2, 5))})

    # This doesn't work
    @unittest.skip
    def test_parameter_config(self):
        input_var = Input(shape=(5, ))
        parameter = ParameterLayer(length=2)(input_var)
        model = Model(inputs=input_var, outputs=parameter)

        self.sess.run(tf.global_variables_initializer())

        fresh_model = tf.keras.models.model_from_config(
            model.get_config(),
            custom_objects={'ParameterLayer': ParameterLayer})
        fresh_model.set_weights(model.get_weights())

        y = self.sess.run(
            model.output, feed_dict={model.input: np.random.random((2, 5))})
        yy = self.sess.run(
            fresh_model.output,
            feed_dict={fresh_model.input: np.random.random((2, 5))})

    @unittest.skip
    def test_mlp_pickling(self):
        input_var = Input(shape=(5, ))

        mlp = box(input_var)
        x = pickle.dumps(mlp)
        y = pickle.loads(x)

    # @unittest.skip
    def test_gaussian_mlp(self):

        model = GaussianMLPModel2(
            input_dim=5, output_dim=2, hidden_sizes=(4, 4))

        self.sess.run(tf.global_variables_initializer())
        data = np.random.random((2, 5))

        y = self.sess.run(model.outputs[0], feed_dict={model.input: data})

        x = pickle.dumps(model)
        model_pickled = pickle.loads(x)

        y2 = self.sess.run(
            model_pickled.outputs[0], feed_dict={model_pickled.input: data})
        print(y)
        print(y2)
