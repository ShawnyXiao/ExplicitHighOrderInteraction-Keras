from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Embedding
from tensorflow.keras.regularizers import l2

from cross_network import CrossNetwork


class DCN(Model):

    def __init__(self,
                 feat_dense_num=0,
                 feat_sparse_num=0,
                 feat_sparse_vocab_sizes=(),
                 feat_sparse_embedding_sizes=(),
                 deep_layer_sizes=(256, 128),
                 deep_use_bias=True,
                 deep_activation='relu',
                 deep_l2_reg=0.0,
                 deep_use_dropout=False,
                 deep_dropout_rate=0.3,
                 cross_layer_num=1,
                 cross_l2_reg=0.0):
        super(DCN, self).__init__()
        if (feat_dense_num == 0) and (feat_sparse_num == 0):
            raise ValueError('The feat_dense_num and feat_sparse_num can not all be 0')
        if len(feat_sparse_vocab_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_vocab_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_vocab_sizes)))
        if len(feat_sparse_embedding_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_embedding_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_embedding_sizes)))
        self.feat_dense_num = feat_dense_num
        self.feat_sparse_num = feat_sparse_num
        self.feat_sparse_vocab_sizes = feat_sparse_vocab_sizes
        self.feat_sparse_embedding_sizes = feat_sparse_embedding_sizes
        self.deep_layer_sizes = deep_layer_sizes
        self.deep_use_bias = deep_use_bias
        self.deep_activation = deep_activation
        self.deep_l2_reg = deep_l2_reg
        self.deep_use_dropout = deep_use_dropout
        self.deep_dropout_rate = deep_dropout_rate
        self.cross_layer_num = cross_layer_num
        self.cross_l2_reg = cross_l2_reg
        if self.feat_sparse_num > 0:
            self.embeddings = []
            for i in range(self.feat_sparse_num):
                self.embeddings.append(Embedding(self.feat_sparse_vocab_sizes[i], self.feat_sparse_embedding_sizes[i]))
        self.deep_network = []
        for deep_layer_size in self.deep_layer_sizes:
            self.deep_network.append(Dense(deep_layer_size,
                                           activation=self.deep_activation,
                                           use_bias=self.deep_use_bias,
                                           kernel_regularizer=l2(self.deep_l2_reg)))
        if self.deep_use_dropout:
            self.deep_dropout = Dropout(self.deep_dropout_rate)
        self.cross_network = CrossNetwork(self.cross_layer_num, self.cross_l2_reg)
        self.concatenate = Concatenate()
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of DCN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.feat_dense_num + self.feat_sparse_num:
            raise ValueError('The 2nd dim of inputs of DCN must be %d, but now is %d' % (self.feat_dense_num + self.feat_sparse_num, inputs.get_shape()[1]))
        dense_input = inputs[:, :self.feat_dense_num]
        sparse_input = inputs[:, self.feat_dense_num:]
        sparse_embeddings = self._get_sparse_embeddings(sparse_input)
        all_input = self._get_all_input(dense_input, sparse_embeddings)
        deep_output = all_input
        for deep_layer in self.deep_network:
            deep_output = deep_layer(deep_output)
        if self.deep_use_dropout:
            deep_output = self.deep_dropout(deep_output)
        cross_output = self.cross_network(all_input)
        final_output = self.concatenate([deep_output, cross_output])
        final_output = self.classifier(final_output)
        return final_output

    def _get_sparse_embeddings(self, sparse_input):
        sparse_embeddings = []
        for i in range(self.feat_sparse_num):
            sparse_embedding = self.embeddings[i](sparse_input[:, i])
            sparse_embeddings.append(sparse_embedding)
        return sparse_embeddings

    def _get_all_input(self, dense_input, sparse_embeddings):
        if self.feat_dense_num > 0 and self.feat_sparse_num > 0:
            all_input = self.concatenate([dense_input] + sparse_embeddings)
        elif self.feat_sparse_num > 0:
            all_input = self.concatenate(sparse_embeddings)
        else:
            all_input = dense_input
        return all_input
