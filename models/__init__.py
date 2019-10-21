
# -*- coding: utf-8 -*-

from models.LSTM2L import LSTM2L
from models.CNN import CNN
from models.BiLSTM import BiLSTM
from models.BiLSTM2L import BiLSTM2L



def setup(opt):
    
    if opt.contatenate==1:
            opt.max_sequence_length = opt.max_sequence_length_long  
            
    if opt.model == "lstm_2L":
        model = LSTM2L(opt)
    elif opt.model == "cnn":
        model = CNN(opt)
    elif opt.model == "bilstm":
        model = BiLSTM(opt)
    elif opt.model == "bilstm_2L":
        model = BiLSTM2L(opt)
    elif opt.model == "bilstm_2inputs":
        model = BiLSTM_2inputs(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))

    return model