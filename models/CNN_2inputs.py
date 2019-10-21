# -*- coding: utf-8 -*-
from keras.layers import Conv1D, MaxPooling1D,Dense,  LSTM, GRU, Bidirectional,Dropout,Input,GlobalMaxPooling1D, Embedding,Concatenate
from models.BasicModel import BasicModel
from keras.models import Model
class CNN(BasicModel):
    def get_model(self,opt):
        claim = Input(shape=(opt.max_sequence_length,opt.embedding_dim), dtype='int32', name='main_input')
        supprt = Input(shape(opt.max_sequence_length_long,embedding_dim),dtype='int32',name='support_doc')

        # claim encoder
        x1 = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        embedded_sequences = embedding_layer(sequence_input)
        representions=[]
        for i in [1,2,3,4]:
            x1 = Conv1D(filters=opt.filter_size, kernel_size=i, activation='relu')(embedded_sequences)
            x1 = GlobalMaxPooling1D()(x1)
            x1 = Dropout(self.opt.dropout_rate)(x1)
            representions.append(x1)
        x1 = Concatenate()(representions)
        LSTM_claim = Dense(100,activation='relu')(x1)
        encode_claim = LSTM_claim(claim)

        # support encoder
        x2 = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        embedded_sequences = embedding_layer(sequence_input)
        representions=[]
        for i in [1,2,3,4]:
            x2 = Conv1D(filters=opt.filter_size, kernel_size=i, activation='relu')(embedded_sequences)
            x2 = GlobalMaxPooling1D()(x2)
            x2 = Dropout(self.opt.dropout_rate)(x2)
            representions.append(x2)
        x2 = Concatenate()(representions)
        LSTM_support = Dense(100,activation='relu')(x2)
        encode_claim = LSTM_support(support)

        # concatenate
        merged_vector = keras.layers.concatenate([encode_claim, encode_support], axis=-1)
        
        prediction = Dense(self.opt.nb_classes, activation='softmax')(x)   # 3 catetory

        return Model(inputs=[claim, support], outputs=prediction)