from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Embedding, BatchNormalization, Activation, Flatten
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from interacting_layer import InteractingLayer


class AutoInt(Model):

    def __init__(self,
                 feat_dense_num=0,
                 feat_dense_embedding_size=32,
                 feat_sparse_num=0,
                 feat_sparse_vocab_sizes=(),
                 feat_sparse_embedding_sizes=(),
                 feat_use_dropout=False,
                 feat_dropout_rate=0.3,
                 deep_layer_sizes=(256, 128),
                 deep_use_bias=True,
                 deep_l2_reg=0.0,
                 deep_use_batch_norm=False,
                 deep_activation='relu',
                 deep_use_dropout=False,
                 deep_dropout_rate=0.3,
                 interact_layer_num=1,
                 interact_unit_num=128,
                 interact_head_num=1,
                 interact_use_dropout=False,
                 interact_dropout_rate=0.3,
                 interact_use_res=True):
        super(AutoInt, self).__init__()
        if (feat_dense_num == 0) and (feat_sparse_num == 0):
            raise ValueError('The feat_dense_num and feat_sparse_num can not all be 0')
        if len(feat_sparse_vocab_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_vocab_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_vocab_sizes)))
        if len(feat_sparse_embedding_sizes) != feat_sparse_num:
            raise ValueError('The length of feat_sparse_embedding_sizes must be equal with feat_sparse_num %d, but now is %d' % (feat_sparse_num, len(feat_sparse_embedding_sizes)))
        self.feat_dense_num = feat_dense_num
        self.feat_dense_embedding_size = feat_dense_embedding_size
        self.feat_sparse_num = feat_sparse_num
        self.feat_sparse_vocab_sizes = feat_sparse_vocab_sizes
        self.feat_sparse_embedding_sizes = feat_sparse_embedding_sizes
        self.feat_use_dropout = feat_use_dropout
        self.feat_dropout_rate = feat_dropout_rate
        self.deep_layer_sizes = deep_layer_sizes
        self.deep_use_bias = deep_use_bias
        self.deep_l2_reg = deep_l2_reg
        self.deep_use_batch_norm = deep_use_batch_norm
        self.deep_activation = deep_activation
        self.deep_use_dropout = deep_use_dropout
        self.deep_dropout_rate = deep_dropout_rate
        self.interact_layer_num = interact_layer_num
        self.interact_unit_num = interact_unit_num
        self.interact_head_num = interact_head_num
        self.interact_use_dropout = interact_use_dropout
        self.interact_dropout_rate = interact_dropout_rate
        self.interact_use_res = interact_use_res
        if self.feat_dense_num > 0:
            self.denses = []
            for i in range(self.feat_dense_num):
                self.denses.append(Dense(self.feat_dense_embedding_size, use_bias=False))
        if self.feat_sparse_num > 0:
            self.embeddings = []
            for i in range(self.feat_sparse_num):
                self.embeddings.append(Embedding(self.feat_sparse_vocab_sizes[i], self.feat_sparse_embedding_sizes[i]))
        if self.feat_use_dropout:
            self.feat_dropout = Dropout(self.feat_dropout_rate)
        self.deep_network = []
        for deep_layer_size in self.deep_layer_sizes:
            self.deep_network.append(Dense(deep_layer_size,
                                           use_bias=self.deep_use_bias,
                                           kernel_regularizer=l2(self.deep_l2_reg)))
        if self.deep_use_batch_norm:
            self.deep_batch_norm = BatchNormalization()
        self.deep_activation_func = Activation(self.deep_activation)
        if self.deep_use_dropout:
            self.deep_dropout = Dropout(self.deep_dropout_rate)
        self.interacting_layer = InteractingLayer(self.interact_layer_num,
                                                  self.interact_unit_num,
                                                  self.interact_head_num,
                                                  self.interact_use_dropout,
                                                  self.interact_dropout_rate,
                                                  self.interact_use_res)
        self.all_input_concatenate = Concatenate(1)
        self.concatenate = Concatenate()
        self.flatten = Flatten()
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of AutoInt must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.feat_dense_num + self.feat_sparse_num:
            raise ValueError('The 2nd dim of inputs of AutoInt must be %d, but now is %d' % (self.feat_dense_num + self.feat_sparse_num, inputs.get_shape()[1]))
        dense_input = inputs[:, :self.feat_dense_num]
        sparse_input = inputs[:, self.feat_dense_num:]
        dense_embeddings = self._get_dense_embeddings(dense_input)
        sparse_embeddings = self._get_sparse_embeddings(sparse_input)
        all_input = self.all_input_concatenate([embedding[:, tf.newaxis, :] for embedding in dense_embeddings + sparse_embeddings])
        if self.feat_use_dropout:
            all_input = self.feat_dropout(all_input)
        deep_output = self.flatten(all_input)
        for deep_layer in self.deep_network:
            deep_output = deep_layer(deep_output)
            if self.deep_use_batch_norm:
                deep_output = self.deep_batch_norm(deep_output)
            deep_output = self.deep_activation_func(deep_output)
            if self.deep_use_dropout:
                deep_output = self.deep_dropout(deep_output)
        interact_output = self.interacting_layer(all_input)
        interact_output = self.flatten(interact_output)
        final_output = self.concatenate([deep_output, interact_output])
        final_output = self.classifier(final_output)
        return final_output

    def _get_dense_embeddings(self, dense_input):
        dense_embeddings = []
        for i in range(self.feat_dense_num):
            dense_embedding = self.denses[i](dense_input[:, i][:, tf.newaxis])
            dense_embeddings.append(dense_embedding)
        return dense_embeddings

    def _get_sparse_embeddings(self, sparse_input):
        sparse_embeddings = []
        for i in range(self.feat_sparse_num):
            sparse_embedding = self.embeddings[i](sparse_input[:, i])
            sparse_embeddings.append(sparse_embedding)
        return sparse_embeddings
