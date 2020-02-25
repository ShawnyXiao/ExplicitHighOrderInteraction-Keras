import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError('The rank of input of LayerNormalization can not be less than 2, but now is %d' % len(input_shape))
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer='zeros',
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer='ones',
                                     trainable=True)

    def call(self, inputs):
        if len(inputs.get_shape()) < 2:
            raise ValueError('The rank of input of LayerNormalization can not be less than 2, but now is %d' % len(inputs.get_shape()))
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs
