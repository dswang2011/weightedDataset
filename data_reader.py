
import os
#task: 1. get tokenized diction; 2. 

#import stanfordnlp
import numpy as np
import codecs

# global tool
nlp = None



punctuation_list = [',',':',';','.','!','?','...','…','。']
# punctuation_list = ['.']


def get_embedding_dict(GLOVE_DIR):
	embeddings_index = {}
	f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	# customized dict
	f =  codecs.open(os.path.join(GLOVE_DIR, 'customized.100d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index

def get_pos_embedding_dict(GLOVE_DIR):
	embeddings_index = {}
	# customized dict
	f =  codecs.open(os.path.join(GLOVE_DIR, 'POSs_13d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index

def get_dep_embedding_dict(GLOVE_DIR):
	embeddings_index = {}
	# customized dict
	f =  codecs.open(os.path.join(GLOVE_DIR, 'deps_42d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index

#/home/dongsheng/data/resources/wordnet/wordnet.20d
def get_wn_embedding_dict(wordnet_dir):
	embeddings_index = {}
	# customized dict
	f =  codecs.open(os.path.join(WORDNET_DIR, 'hypernym_noun.20d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


def tokenizer(tokens_list,MAX_NB_WORDS):
	index = 1
	word_index = {}
	for tokens in tokens_list:
		for token in tokens:
			# add to word_index
			if len(word_index)<MAX_NB_WORDS:
				token=token.lower()
				if token in word_index.keys():
					continue
				else:
					word_index[token] = index
					index+=1
	return word_index


def feature_to_sequences(texts,word_index,MAX_SEQUENCE_LENGTH):
	sequences = []
	for text in texts:
		sequence = []
		tokens = text.split(' ')
		for token in tokens:
			if token.strip() in word_index.keys():
				sequence.append(word_index[token.strip()])
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# input is the generalized text; 
def text_to_sequences(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	if nlp is None:
		nlp=stanfordnlp.Pipeline(use_gpu=False)
	# nlp = stanfordnlp.Pipeline()
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding (predicates of mentions)
				local_encoding = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						local_encoding = word_index[pred_token]
				# global encoding (punctuations or predicates)
				global_encoding = 0
				if token in punctuation_list:
					global_encoding = token_index
				else:
					if global_pred[i]['head']==j:
						global_encoding = token_index
				# concatenate
				concate = [token_index,local_encoding,global_encoding]
				sequence+=concate # add to the list
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# input is the generalized text; 
def tokens_list_to_sequences(tokens_lists,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for tokens in tokens_lists:
		sequence = []
		for token in tokens:
			token = token.lower()
			if token in word_index.keys():
				token_index = word_index[token]
				sequence.append(token_index)
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)

# input is the generalized text; 
 

def docs_to_sequences_suffix(docs,word_index, MAX_SEQUENCE_LENGTH, contatenate=0):

	sequences = []
	a = 1
	for doc in docs:
		# print("Doc in docs:", a)
		a+=1
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		# txt_matrix = np.asarray(txt_matrix)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []

		# == process this text matrix
		attentions = []
		attentions+=[word_index['.']]
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			if i==0:
				sequence+=[word_index['.']]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0
				if token in word_index.keys():
					token_index = word_index[token]
				if contatenate==1:

					sequence += [token_index]
				# local encoding
				pred_index = 0
				if token in ['aaac','bbbc','pppc','pppcs']:
					possessive = 0
					if len(sent_arr)>(j+1) and sent_arr[j+1].lower()=="'s":
						possessive = 1
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate 
		sequence += attentions
		# print('seq/att:',len(sequence),len(attentions))
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)

# input is the generalized text; 
def text_to_sequences_suffix(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		attentions = []
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]	
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding
				pred_index = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate
		sequence += attentions
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


def load_apple_tweet_sent(file_path,hasHead=1):
	texts,confidences,labels=[],[],[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split(',')
			if len(strs)<11:
				continue
			text = strs[11].strip()
			label = strs[5].strip().lower()
			confid = strs[6].strip().lower()
			labels.append(label)
			texts.append(text)
			confidences.append(confid)
			
	return [texts,confidences],labels

def load_tweet_glob_warm(file_path,hasHead=1):
	texts,confidences,labels=[],[],[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			if len(strs)<3:
				continue
			text = strs[0].strip()
			label = strs[1].strip().lower()
			if label not in ['0','1']:
				continue
			confid = strs[2].strip().lower()
			labels.append(label)
			texts.append(text)
			confidences.append(confid)
			
	return [texts,confidences],labels

def load_data_overall(dataset,file_name="train.csv"):
	texts,entity1,entity2,labels=[],[],[],[]
	output_root = "datasets/"+dataset+"/"
	if dataset in ['apple_tweet']:
		return load_apple_tweet_sent(file_path=output_root+file_name)
	elif dataset in ['tweet_global_warm']:
		return load_tweet_glob_warm(file_path=output_root+file_name)


def write_line(file_path,content):
	with open(file_path,'a',encoding='utf8') as fw:
		fw.write(content)
		fw.write('\n')


texts,labels = load_data_overall('tweet_global_warm','test.csv')
print(set(labels))
print(len(labels))
# print(texts[7][:5])
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
# le = LabelEncoder()
# pos_list = list(pos_set)
# encoded = le.fit_transform(np.asarray(pos_list))
# encoded = to_categorical(encoded)
# for i in range(len(pos_list)):
# 	vect = encoded[i]
# 	# vect = vect.replace(']','')
# 	# vect = vect.replace('\n','')
# 	# write_line('POSs.txt',pos_list[i].strip()+' '+vect.strip())
# 	print(pos_list[i],' ',len(vect))
# # print(encoded)
# # print('labels:',len(set(pos_tags)),set(pos_tags))
