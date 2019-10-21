
import os
#task: 1. get tokenized diction; 2. 

#import stanfordnlp
import NLP
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


def load_data(tsv_file_path,mode= "train"):
    with open(tsv_file_path, encoding='utf8') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    header = content[0]
    res = []
    for line in content[1:]:
        data = DatasetSchema(line)
        orig_txt = data.get_text()
        generalized_txt = data.get_generalized_text()
        # below is to get exact sentences
        sentences = generalized_txt.split('.')
        exact_sents = []
        for sent in sentences:
            if 'AAAC' in sent or 'BBBC' in sent or 'PPPC' in sent or 'PPPCS' in sent:
                exact_sents.append(sent)
        exact_txt = '.'.join(exact_sents)
        # end of previous below

        if mode == "train":
            label_A = data.get_A_coref()
            label_B = data.get_B_coref()
            if label_A in ['TRUE','True','true'] and label_B in ['FALSE','False','false']:
                label = 0
            elif label_B in ['TRUE','True','true'] and label_A in ['FALSE','False','false']:
                label = 1
            else:
                label = 2
            res.append([orig_txt,exact_txt,label])
        else:
            samp_id = data.get_id()
            res.append([orig_txt,exact_txt,samp_id])
    return np.array(res)

import csv
### uncomment this to use for the IMDB/MR dataset ##########
def load_bi_class_data(file_path,hasHead=0):
	texts=[]
	labels=[]
	with open(file_path, encoding='utf8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for row in csv_reader:
			texts.append(row[0].strip())
			label = '0'
			for i in range(1,len(row)):
				if row[i].strip() in ['0','1']:
					label = row[i].strip()
			labels.append(label)
	# print('labels:',labels)

	return [texts,labels]

#### THIS IS TO RUN FOR GAP ######
def load_classification_data(file_path,hasHead=0):
	texts=[]
	labels=[]
	with open(file_path, encoding='utf8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for row in csv_reader:
			texts.append(row[0].strip())
			# label = '0'
			for i in range(1,len(row)):
			# 	if row[i].strip() in ['0','1',0,1]:
				label = row[i].strip()
				# print(label)
			labels.append(label)
	# print('labels:',labels)
	return [texts,labels]


def load_pair_data(file_path,hasHead=0):
	texts1,texts2=[],[]
	labels=[]
	with open(file_path, encoding='utf8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for row in csv_reader:
			claim_id = row[0].strip()
			if 'pomt' not in claim_id:
				continue
			label = row[2].strip().lower()
			if label not in ['true','false','no flip','half-true','pants on fire!','half flip','mostly true','full flop','mostly false']:
				continue
			max_snippets = np.maximum(5,len(row)-3)
			texts1.append(row[1].strip())
			text2 = ' '.join(row[3:max_snippets])
			texts2.append(text2.strip())
			labels.append(label)
	return texts1,texts2,labels

def load_triple_data(file_path):
	triples=[]
	labels=[]
	with open(file_path,'r',encoding='utf8') as f:
		for line in f:
			strs = line.split('\t')
			triples.append(strs[0].strip())
			label = '0'
			for i in range(1,len(strs)):
				if strs[i].strip() in ['0','1']:
					label = strs[i].strip()
			labels.append(label)
	return [triples,labels]


def get_texts_from_folder(directory):
	texts = []
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			file_path = os.path.join(directory, filename)
			file = open(file_path,'r')
			lines = file.readlines()
			texts.append(' '.join(lines).replace('\n',''))
	return texts
# load MR
from sklearn.utils import shuffle
def load_mr_data(folder):
	pos_texts = get_texts_from_folder(folder+'/'+'pos/')
	neg_texts = get_texts_from_folder(folder+'/'+'neg/')
	pos_labels = np.ones(len(pos_texts),dtype=int)
	neg_labels = np.zeros(len(neg_texts),dtype=int)
	texts = pos_texts+neg_texts
	labels = pos_labels.tolist()+neg_labels.tolist()
	X,y = shuffle(texts, labels, random_state=0)
	return [X,y]

def load_RTE_data(file_path,hasHead=0):
	texts1,texts2=[],[]
	labels=[]
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			strs = row.split('\t')
			label = strs[3].strip().lower()
			if label not in ['not_entailment','entailment','0','1']:
				print('strange label:',label)
				continue
			texts1.append(strs[1].strip())
			texts2.append(strs[2].strip())
			labels.append(label)
	return texts1,texts2,labels

def load_MRPC_data(file_path,hasHead=1):
	texts1,texts2=[],[]
	labels=[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			label = strs[0].strip().lower()
			if label not in [0,1,'0','1']:
				print('strange label:',label)
				continue
			texts1.append(strs[3].strip())
			texts2.append(strs[4].strip())
			labels.append(label)
	return texts1,texts2,labels


def load_relation_data(file_path,hasHead=0):
	texts,entity1,entity2,labels=[],[],[],[]
	labels=[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			label = strs[3].strip().lower()
			
			texts.append(strs[0].strip())
			entity1.append(strs[1].strip())
			entity2.append(strs[2].strip())
			labels.append(label)
	return [texts,entity1,entity2],labels	

def load_KBP_data(file_path,hasHead=0):
	texts,entity1,entity2,labels=[],[],[],[]
	labels=[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			label = strs[4].strip().lower()
			
			texts.append(strs[1].strip())
			entity1.append(strs[2].strip())
			entity2.append(strs[3].strip())
			labels.append(label)
	return [texts,entity1,entity2],labels	

def load_rel_block_data(file_path,hasHead=0):
	texts,blocks,entity1,entity2,labels=[],[],[],[],[]
	labels=[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			label = strs[4].strip().lower()
			label = label.replace('(e1,e2)','')
			label = label.replace('(e2,e1)','')
			texts.append(strs[0].strip())
			# blocks.append(strs[1].strip())
			block = strs[1].strip()
			e1 = strs[2].strip()
			e2 = strs[3].strip()
			# process
			entity1.append(e1)
			entity2.append(e2)
			# block = block.replace(e1,'aaac')
			# block = block.replace(e2,'bbbc')
			blocks.append(block)
			labels.append(label)
	if 'SemEval2' in file_path:
		return [blocks,entity1,entity2],labels
	return [texts,blocks,entity1,entity2],labels

def load_rel_feature_data(file_path,hasHead=0):
	blocks,POSs,deps,entity1,entity2,labels=[],[],[],[],[],[]
	texts,heads,cats=[],[],[]
	count_line=0
	with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
		for row in f:
			count_line+=1
			if count_line==1 and hasHead==1:
				continue
			strs = row.split('\t')
			label = strs[6].strip().lower()
			label = label.replace('(e1,e2)','')
			label = label.replace('(e2,e1)','')
			# blocks.append(strs[1].strip())
			text = strs[0].strip()
			block = strs[1].strip()
			pos = strs[2].strip()
			dep = strs[3].strip()
			# head = strs[4].strip()
			e1 = strs[4].strip()
			e2 = strs[5].strip()
			# cat
			# cat = strs[8].strip()
			# cats.append(cat)
			# process
			texts.append(text)
			entity1.append(e1)
			entity2.append(e2)
			blocks.append(block)
			# heads.append(head)
			POSs.append(pos)
			deps.append(dep)
			labels.append(label)
			
	return [texts,blocks,heads,POSs,deps,entity1,entity2],labels


def load_data_overall(dataset,file_name="train.csv"):
	texts,entity1,entity2,labels=[],[],[],[]
	output_root = "prepared/"+dataset+"/"
	if dataset in ['SemEval']:
		return load_relation_data(file_path=output_root+file_name)
	elif dataset in ['SemEval_block','SemEval2']:
		return load_rel_block_data(file_path=output_root+file_name)
	elif dataset in ['SemEval_feature','SemEval_longblock','KBP37_longblock','KBP37_shortblock']:
		return load_rel_feature_data(file_path=output_root+file_name)
	elif dataset in ['KBP','KBP37']:
		return load_KBP_data(file_path=output_root+file_name)

def write_line(file_path,content):
	with open(file_path,'a',encoding='utf8') as fw:
		fw.write(content)
		fw.write('\n')


# texts,labels = load_data_overall('SemEval_feature','test.csv')
# print(set(labels))
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
