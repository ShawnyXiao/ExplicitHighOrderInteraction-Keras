import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2


class CrossNetwork(Layer):

    def __init__(self, layer_num=1, l2_reg=0.0, **kwargs):
        super(CrossNetwork, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.l2_reg = l2_reg

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('The rank of input of CrossNetworkLayer must be 2, but now is %d' % len(input_shape))
        self.ws = [self.add_weight(shape=(int(input_shape[1]), 1),
                                   initializer='random_normal',
                                   regularizer=l2(self.l2_reg),
                                   trainable=True) for _ in range(self.layer_num)]
        self.bs = [self.add_weight(shape=(int(input_shape[1]), 1),
                                   initializer='zeros',
                                   trainable=True) for _ in range(self.layer_num)]

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of input of CrossNetworkLayer must be 2, but now is %d' % len(inputs.get_shape()))
        x_l = x_0 = inputs[:, :, tf.newaxis]
        for i in range(self.layer_num):
            x_l = self._cross(x_l, self.ws[i], self.bs[i], x_0)
        x_l = x_l[:, :, 0]
        return x_l

    def _cross(self, x_l, w_l, b_l, x_0):
        return tf.matmul(x_0, tf.tensordot(x_l, w_l, (1, 0))) + b_l + x_l
