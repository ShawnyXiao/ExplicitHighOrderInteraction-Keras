import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Flatten, Embedding
from tensorflow.keras.regularizers import l2

from cin import CIN


class xDeepFM(Model):

    def __init__(self,
                 feat_dense_num=0,
                 feat_sparse_num=0,
                 feat_sparse_vocab_sizes=(),
                 feat_sparse_embedding_sizes=(),
                 dnn_layer_sizes=(256, 128),
                 dnn_use_bias=True,
                 dnn_activation='relu',
                 dnn_l2_reg=0.0,
                 dnn_use_dropout=False,
                 dnn_dropout_rate=0.3,
                 cin_layer_sizes=(128, 128),
                 cin_is_direct=False,
                 cin_use_bias=False,
                 cin_use_res=False,
                 cin_use_activation=False,
                 cin_activation='relu',
                 cin_reduce_filter=False,
                 cin_filter_dim=2):
        super(xDeepFM, self).__init__()
        if feat_sparse_num == 0:
            raise ValueError('The feat_sparse_num can not be 0')
        if len(feat_sparse_vocab_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_vocab_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_vocab_sizes)))
        if len(feat_sparse_embedding_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_embedding_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_embedding_sizes)))
        self.feat_dense_num = feat_dense_num
        self.feat_sparse_num = feat_sparse_num
        self.feat_sparse_vocab_sizes = feat_sparse_vocab_sizes
        self.feat_sparse_embedding_sizes = feat_sparse_embedding_sizes
        self.dnn_layer_sizes = dnn_layer_sizes
        self.dnn_use_bias = dnn_use_bias
        self.dnn_activation = dnn_activation
        self.dnn_l2_reg = dnn_l2_reg
        self.dnn_use_dropout = dnn_use_dropout
        self.dnn_dropout_rate = dnn_dropout_rate
        self.cin_layer_sizes = cin_layer_sizes
        self.cin_is_direct = cin_is_direct
        self.cin_use_bias = cin_use_bias
        self.cin_use_res = cin_use_res
        self.cin_use_activation = cin_use_activation
        self.cin_activation = cin_activation
        self.cin_reduce_filter = cin_reduce_filter
        self.cin_filter_dim = cin_filter_dim
        if self.feat_sparse_num > 0:
            self.embeddings = []
            for i in range(self.feat_sparse_num):
                self.embeddings.append(Embedding(self.feat_sparse_vocab_sizes[i], self.feat_sparse_embedding_sizes[i]))
        self.linear = Embedding(sum(self.feat_sparse_vocab_sizes), 1)
        self.dnn = []
        for dnn_layer_size in self.dnn_layer_sizes:
            self.dnn.append(Dense(dnn_layer_size,
                                  self.dnn_activation,
                                  self.dnn_use_bias,
                                  kernel_regularizer=l2(self.dnn_l2_reg)))
        if self.dnn_use_dropout:
            self.dnn_dropout = Dropout(self.dnn_dropout_rate)
        self.cin = CIN(self.cin_layer_sizes,
                       self.cin_is_direct,
                       self.cin_use_bias,
                       self.cin_use_res,
                       self.cin_use_activation,
                       self.cin_activation,
                       self.cin_reduce_filter,
                       self.cin_filter_dim)
        self.concatenate = Concatenate()
        self.cin_input_concatenate = Concatenate(1)
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of xDeepFM must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.feat_dense_num + self.feat_sparse_num:
            raise ValueError('The 2nd dim of inputs of xDeepFM must be %d, but now is %d' % (self.feat_dense_num + self.feat_sparse_num, inputs.get_shape()[1]))
        dense_input = inputs[:, :self.feat_dense_num]
        sparse_input = inputs[:, self.feat_dense_num:]
        sparse_embeddings = self._get_sparse_embeddings(sparse_input)
        dnn_input = self._get_dnn_input(dense_input, sparse_embeddings)
        cin_input = self.cin_input_concatenate([embedding[:, tf.newaxis, :] for embedding in sparse_embeddings])
        linear_output = self.linear(sparse_input)[:, :, 0]
        dnn_output = dnn_input
        for dnn_layer in self.dnn:
            dnn_output = dnn_layer(dnn_output)
        if self.dnn_use_dropout:
            dnn_output = self.dnn_dropout(dnn_output)
        cin_output = self.cin(cin_input)
        final_output = self.concatenate([linear_output, dnn_output, cin_output])
        final_output = self.classifier(final_output)
        return final_output

    def _get_sparse_embeddings(self, sparse_input):
        sparse_embeddings = []
        for i in range(self.feat_sparse_num):
            sparse_embedding = self.embeddings[i](sparse_input[:, i])
            sparse_embeddings.append(sparse_embedding)
        return sparse_embeddings

    def _get_dnn_input(self, dense_input, sparse_embeddings):
        if self.feat_dense_num > 0 and self.feat_sparse_num > 0:
            dnn_input = self.concatenate([dense_input] + sparse_embeddings)
        elif self.feat_sparse_num > 0:
            dnn_input = self.concatenate(sparse_embeddings)
        else:
            dnn_input = dense_input
        return dnn_input
