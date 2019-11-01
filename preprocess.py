import pickle  
from  Params import Params
import argparse
import data_reader
import os
import keras
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
from keras.preprocessing import text as text4text

class Preprocess(object):
	def __init__(self,opt):
		self.opt=opt   

	def build_word_embedding_matrix(self,word_index):
		# word embedding lodading
		embeddings_index = data_reader.get_embedding_dict(self.opt.glove_dir)
		print('Total %s word vectors.' % len(embeddings_index))

		# initial: random initial (not zero initial)
		embedding_matrix = np.random.random((len(word_index) + 1,self.opt.embedding_dim  ))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		return embedding_matrix

	def get_train(self,dataset,strategy="fulltext",selected_ratio=0.9,cut=1,POS_category="Noun",sig_num=3):
		print('=====strategy:',strategy,'pos_cat:',POS_category,' cut:',cut,'======')
		texts_train_test = []
		labels_train_test = []
		for file_name in ["train.csv","test.csv"]:
			texts,labels = data_reader.load_data_overall(dataset,file_name)
			texts_train_test.append(texts)
			labels_train_test.append(labels)
		self.opt.nb_classes = len(set(labels))
		print('labels:',set(labels))
		# max_num_words = self.opt.max_num_words
		all_sents= [sentence for dataset in texts_train_test for sentence in dataset[0]]
		
		tokenizer = text4text.Tokenizer(num_words=self.opt.max_nb_words) 
		
		tokenizer.fit_on_texts(all_sents) 
		word_index = word_index = tokenizer.word_index
		self.opt.word_index = word_index
		print('word_index len:',len(word_index))
		# create embedding (usually not put here)
		self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)

		le = preprocessing.LabelEncoder()
		# labels = le.fit_transform(labels)
		# print(labels)
		# padding
		train_test = []
		for texts,labels in zip(texts_train_test,labels_train_test):
			x = []
			if dataset in self.opt.pair_set.split(","):
				x1 = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts[0]), maxlen=self.opt.max_sequence_length)
				x2 = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts[1]), maxlen=self.opt.max_sequence_length)
				x = [x1,x2]
			else:
				x1 = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts[0]), maxlen=self.opt.max_sequence_length)
				x2 = np.array(texts[1])
				x = [x1,x2]
			y = le.fit_transform(labels)
			y = to_categorical(np.asarray(y)) # one-hot encoding y_train = labels # one-hot label encoding
			train_test.append([x,y])
			print('[train] Shape of label tensor:', y.shape)
		return train_test