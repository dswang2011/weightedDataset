# -*- coding: utf-8 -*-


from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU, Bidirectional,Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.constraints import maxnorm
from models.RNNBasic import RNNBasic

class LSTM1(RNNBasic):
    
    def get_model(self,opt):
        text_branch = Sequential()
        embedding_layer = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=False)
        text_branch.add(embedding_layer)
        # text_branch.add(TrigPosEmbedding(
        #     input_shape=(None,),
        #     output_dim=EMBEDDING_DIM,     # The dimension of embeddings.
        #     mode=TrigPosEmbedding.MODE_ADD,  # Use `add` mode; MODE_CONCAT
        #     name='Pos-Embd',
        # ))
        # text_branch.add(Bidirectional(LSTM(units=100,return_sequences=False)))
        text_branch.add(LSTM(units=self.opt.hidden_unit_num,return_sequences=True, activation='relu', kernel_constraint=maxnorm(3)))
        text_branch.add(Dropout(self.opt.dropout_rate))
        text_branch.add(LSTM(units=self.opt.hidden_unit_num,return_sequences=False, activation='relu', kernel_constraint=maxnorm(3)))
        text_branch.add(Dropout(self.opt.dropout_rate))
        # text_branch.add(SeqSelfAttention(attention_width=5,attention_activation='sigmoid'))
        text_branch.add(Dense(self.opt.nb_classes,activation="softmax"))        
        
        return text_branch
