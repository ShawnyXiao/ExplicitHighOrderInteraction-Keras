from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.regularizers import l2

from cross_network import CrossNetwork


class DCN(Model):

    def __init__(self,
                 deep_layer_sizes=(256, 128),
                 deep_use_bias=True,
                 deep_activation='relu',
                 deep_l2_reg=0.0,
                 deep_use_dropout=False,
                 deep_dropout_rate=0.3,
                 cross_layer_num=1,
                 cross_l2_reg=0.0):
        super(DCN, self).__init__()
        self.deep_layer_sizes = deep_layer_sizes
        self.deep_use_bias = deep_use_bias
        self.deep_activation = deep_activation
        self.deep_l2_reg = deep_l2_reg
        self.deep_use_dropout = deep_use_dropout
        self.deep_dropout_rate = deep_dropout_rate
        self.cross_layer_num = cross_layer_num
        self.cross_l2_reg = cross_l2_reg
        self.deep_network = []
        for deep_layer_size in self.deep_layer_sizes:
            self.deep_network.append(Dense(deep_layer_size,
                                           activation=self.deep_activation,
                                           use_bias=self.deep_use_bias,
                                           kernel_regularizer=l2(self.deep_l2_reg)))
        if self.deep_use_dropout:
            self.deep_dropout = Dropout(self.deep_dropout_rate)
        self.cross_network = CrossNetwork(self.cross_layer_num, self.cross_l2_reg)
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of DCN must be 2, but now is %d' % len(inputs.get_shape()))
        deep_output = inputs
        for deep_layer in self.deep_network:
            deep_output = deep_layer(deep_output)
        if self.deep_use_dropout:
            deep_output = self.deep_dropout(deep_output)
        cross_output = self.cross_network(inputs)
        final_output = Concatenate()([deep_output, cross_output])
        final_output = self.classifier(final_output)
        return final_output
