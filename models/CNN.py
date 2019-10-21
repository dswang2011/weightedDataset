# -*- coding: utf-8 -*-
from keras.layers import Conv1D, MaxPooling1D,Dense,  LSTM, GRU, Bidirectional,Dropout,Input,GlobalMaxPooling1D, Embedding,Concatenate
from models.BasicModel import BasicModel
from keras.models import Model
from keras_pos_embd import TrigPosEmbedding


class CNN(BasicModel):
    def get_model(self,opt,embedding_type='word'):
        sequence_input = Input(shape=(opt.max_sequence_length,), dtype='int32')
        if embedding_type=='pos':
            embedding_layer = \
            Embedding(int(opt.pos_dim) + 1,opt.pos_dim,weights=[opt.pos_embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        elif embedding_type=='dep':
            embedding_layer = \
            Embedding(int(opt.dep_dim) + 1,opt.dep_dim,weights=[opt.dep_embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        else:
            embedding_layer = \
            Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        embedded_sequences = embedding_layer(sequence_input)

        representions=[]
        for i in [1,2,3,4]:
            x = Conv1D(filters=opt.filter_size, kernel_size=i, activation='relu')(embedded_sequences)
            x = GlobalMaxPooling1D()(x)
            x = Dropout(self.opt.dropout_rate)(x)
            representions.append(x)
        x = Concatenate()(representions)
        x = Dense(100,activation='relu',name='previous_layer')(x)
        preds = Dense(self.opt.nb_classes, activation='softmax')(x)   # 3 catetory

        return Model(sequence_input, preds)
