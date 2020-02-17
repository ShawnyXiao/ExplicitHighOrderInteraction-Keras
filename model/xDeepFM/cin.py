import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation


class CIN(Layer):

    def __init__(self,
                 layer_sizes=(128, 128),
                 is_direct=False,
                 use_bias=False,
                 use_res=False,
                 use_activation=False,
                 activation='relu',
                 reduce_filter=False,
                 filter_dim=2,
                 **kwargs):
        super(CIN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.is_direct = is_direct
        self.use_bias = use_bias
        self.use_res = use_res
        self.use_activation = use_activation
        self.activation = activation
        self.reduce_filter = reduce_filter
        self.filter_dim = filter_dim
        if self.use_res:
            self.res_dense_layer = Dense(128, use_bias=True)
        if self.use_activation:
            self.activation_layer = Activation(activation)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('The rank of input of CIN must be 3, but now is %d' % len(input_shape))
        self.field_nums = [input_shape.as_list()[1]]
        self.filters = []
        for i, layer_size in enumerate(self.layer_sizes):
            if self.is_direct:
                field_num = layer_size
            else:
                field_num = int(layer_size / 2)
            self.field_nums.append(field_num)
            if self.reduce_filter:
                filter_0 = self.add_weight(shape=(1, layer_size, self.field_nums[0], self.filter_dim),
                                           initializer='random_normal',
                                           trainable=True)
                filter_1 = self.add_weight(shape=(1, layer_size, self.filter_dim, self.field_nums[i]),
                                           initializer='random_normal',
                                           trainable=True)
                filter = tf.matmul(filter_0, filter_1)
                filter = tf.reshape(filter, shape=(1, layer_size, self.field_nums[0] * self.field_nums[i]))
                filter = tf.transpose(filter, perm=(0, 2, 1))
            else:
                filter = self.add_weight(shape=(1, self.field_nums[0] * self.field_nums[i], layer_size),
                                         initializer='random_normal',
                                         trainable=True)
            self.filters.append(filter)
        if self.use_bias:
            self.biases = [self.add_weight(shape=(layer_size),
                                           initializer='zeros',
                                           trainable=True) for layer_size in self.layer_sizes]

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of input of CIN must be 3, but now is %d' % len(inputs.get_shape()))
        emb_size = inputs.get_shape()[2]
        hidden_layers = [inputs]
        final_layers = []
        split_tensor0 = tf.split(hidden_layers[0], emb_size * [1], 2)
        for i, layer_size in enumerate(self.layer_sizes):
            split_tensor = tf.split(hidden_layers[i], emb_size * [1], 2)
            outer_product = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            outer_product = tf.reshape(outer_product, shape=(emb_size, -1, self.field_nums[0] * self.field_nums[i]))
            outer_product = tf.transpose(outer_product, perm=(1, 0, 2))
            conv_tensor = tf.nn.conv1d(outer_product, filters=self.filters[i], stride=1, padding='VALID')
            if self.use_bias:
                conv_tensor = tf.nn.bias_add(conv_tensor, self.biases[i])
            if self.use_activation:
                conv_tensor = self.activation_layer(conv_tensor)
            conv_tensor = tf.transpose(conv_tensor, perm=(0, 2, 1))
            if self.is_direct:
                hidden_layer = conv_tensor
                final_layer = conv_tensor
            else:
                if i == len(self.layer_sizes) - 1:
                    hidden_layer = 0
                    final_layer = conv_tensor
                else:
                    hidden_layer, final_layer = tf.split(conv_tensor, 2 * [int(layer_size / 2)], 1)
            hidden_layers.append(hidden_layer)
            final_layers.append(final_layer)
        final_tensor = tf.concat(final_layers, axis=1)
        final_tensor = tf.reduce_sum(final_tensor, axis=2)
        if self.use_res:
            res_tensor = self.res_dense_layer(final_tensor)
            final_tensor = tf.concat([final_tensor, res_tensor], axis=1)
        return final_tensor
