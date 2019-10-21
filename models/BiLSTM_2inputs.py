# -*- coding: utf-8 -*-

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU, Bidirectional,Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.constraints import maxnorm
from models.RNNBasic import RNNBasic

class BiLSTM_2inputs(RNNBasic):
    
    def get_model(self,opt):
        claim = Input(shape=(opt.max_sequence_length,opt.embedding_dim), dtype='int32', name='main_input')
        supprt = Input(shape(opt.max_sequence_length_long,embedding_dim),dtype='int32',name='support_doc')
        # LSTM claim encoder
        x1 = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=False)(main_input)
        x1 = Bidirectional(self.rnncell(units=self.opt.hidden_unit_num_second,return_sequences=True))(x1)  # 300
        LSTM_claim = Dropout(self.opt.dropout_rate)(x1)
        encode_claim = LSTM_claim(claim)
        # LSTM support
        x2 = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length_long,trainable=False)(main_input)
        x2 = Bidirectional(self.rnncell(units=self.opt.hidden_unit_num_second,return_sequences=True))(x2)  # 300
        LSTM_support = Dropout(self.opt.dropout_rate)(x2)
        encode_support = LSTM_support(support)
        # merge two lstm encoder
        merged_vector = keras.layers.concatenate([encode_claim, encode_support], axis=-1)
        # predict
        prediction = Dense(self.opt.nb_classes, activation='softmax')(merged_vector)
        # modeling
        model = Model(inputs=[claim, support], outputs=prediction)

        return model



    