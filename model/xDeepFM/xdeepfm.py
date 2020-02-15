from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.regularizers import l2

from cin import CIN


class xDeepFM(Model):

    def __init__(self,
                 deep_layer_sizes=(256, 128),
                 deep_use_bias=True,
                 deep_activation='relu',
                 deep_l2_reg=0.0,
                 deep_use_dropout=False,
                 deep_dropout_rate=0.3,
                 cin_layer_sizes=(128, 128),
                 cin_is_direct=False,
                 cin_use_bias=False,
                 cin_use_res=False,
                 cin_use_activation=False,
                 cin_activation='relu',
                 cin_reduce_filter=False,
                 cin_filter_dim=2):
        super(xDeepFM, self).__init__()
        self.deep_layer_sizes = deep_layer_sizes
        self.deep_use_bias = deep_use_bias
        self.deep_activation = deep_activation
        self.deep_l2_reg = deep_l2_reg
        self.deep_use_dropout = deep_use_dropout
        self.deep_dropout_rate = deep_dropout_rate
        self.cin_layer_sizes = cin_layer_sizes
        self.cin_is_direct = cin_is_direct
        self.cin_use_bias = cin_use_bias
        self.cin_use_res = cin_use_res
        self.cin_use_activation = cin_use_activation
        self.cin_activation = cin_activation
        self.cin_reduce_filter = cin_reduce_filter
        self.cin_filter_dim = cin_filter_dim
        self.deep_network = []
        for deep_layer_size in self.deep_layer_sizes:
            self.deep_network.append(Dense(deep_layer_size,
                                           self.deep_activation,
                                           self.deep_use_bias,
                                           kernel_regularizer=l2(self.deep_l2_reg)))
        if self.deep_use_dropout:
            self.deep_dropout = Dropout(self.deep_dropout_rate)
        self.cin = CIN(self.cin_layer_sizes,
                       self.cin_is_direct,
                       self.cin_use_bias,
                       self.cin_use_res,
                       self.cin_use_activation,
                       self.cin_activation,
                       self.cin_reduce_filter,
                       self.cin_filter_dim)
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of xDeepFM must be 3, but now is %d' % len(inputs.get_shape()))
        deep_output = Flatten()(inputs)
        for deep_layer in self.deep_network:
            deep_output = deep_layer(deep_output)
        if self.deep_use_dropout:
            deep_output = self.deep_dropout(deep_output)
        cin_output = self.cin(inputs)
        final_output = Concatenate()([deep_output, cin_output])
        final_output = self.classifier(final_output)
        return final_output
