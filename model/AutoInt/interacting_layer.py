import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

from layer_normalization import LayerNormalization


class InteractingLayer(Layer):

    def __init__(self,
                 layer_num=1,
                 unit_num=128,
                 head_num=1,
                 use_dropout=False,
                 dropout_rate=0.3,
                 use_res=True,
                 **kwargs):
        super(InteractingLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.unit_num = unit_num
        self.head_num = head_num
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_res = use_res
        self.query_dense = Dense(self.unit_num, activation='relu')
        self.key_dense = Dense(self.unit_num, activation='relu')
        self.value_dense = Dense(self.unit_num, activation='relu')
        if self.use_dropout:
            self.weight_dropout = Dropout(self.dropout_rate)
        if self.use_res:
            self.res_dense = Dense(self.unit_num, activation='relu')
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('The rank of input of InteractingLayer must be 3, but now is %d' % len(input_shape))

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of input of InteractingLayer must be 3, but now is %d' % len(inputs.get_shape()))
        output = inputs
        for i in range(self.layer_num):
            query = self.query_dense(output)
            key = self.key_dense(output)
            value = self.value_dense(output)
            if self.use_res:
                res = self.res_dense(output)
            query = tf.concat(tf.split(query, self.head_num, axis=2), axis=0)
            key = tf.concat(tf.split(key, self.head_num, axis=2), axis=0)
            value = tf.concat(tf.split(value, self.head_num, axis=2), axis=0)
            weight = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
            weight = weight / (key.get_shape().as_list()[-1] ** 0.5)
            weight = tf.nn.softmax(weight)
            if self.use_dropout:
                weight = self.weight_dropout(weight)
            output = tf.matmul(weight, value)
            output = tf.concat(tf.split(output, self.head_num, axis=0), axis=2)
            if self.use_res:
                output += res
            output = tf.nn.relu(output)
            output = self.layer_norm(output)
        return output
