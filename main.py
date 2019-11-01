
# -*- coding: utf-8 -*-
from  Params import Params
import argparse
import models
from keras.models import load_model
import os
import numpy as np
import itertools
from preprocess import Preprocess

from sklearn.utils import shuffle
import pprint
# import myutil
import matplotlib.pyplot as plt

params = Params()
parser = argparse.ArgumentParser(description='Running Gap.')
parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
args = parser.parse_args()
params.parse_config(args.config)
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,log_loss
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
import time

    

def emsemble(vals,dir_name="saved_model"):
    predicts= [] 
    
    for filename in os.listdir( dir_name ):
        if '0.6' in filename and 'longblock' in filename:
            model_file = os.path.join(dir_name,filename)
            model = load_model(model_file)
            predicted = model.predict(vals[0])
            # print(filename + ": " + str(log_loss(val[1], predicted)))
            predicts.append(predicted)
    return np.mean(predicts,axis=0)

def grid_search_parameters(grid_parameters, train, test, dict_results, dataset,strategy):
    parameters = [arg for index, arg in enumerate(itertools.product(*grid_parameters.values())) if
                  index % args.gpu_num == args.gpu]
    val_acc = 0
    max_acc = 0
    local_time=0.0
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(), parameter))
        model = models.setup(params)
        print('======train relation model=========')
        val_acc, time_spent, model = model.train_weighted(train, dev=test,dataset=dataset)
        if float(val_acc) > max_acc:
            max_acc=float(val_acc)
            local_time=time_spent
        if dataset not in dict_results:
            dict_results[dataset] = {}
        if model not in dict_results[dataset]:
            dict_results[dataset][model] = {}
        if strategy not in dict_results[dataset][model]:
            dict_results[dataset][model][strategy] = {"val_acc":val_acc, "time":time_spent}

        if val_acc > dict_results[dataset][model][strategy]['val_acc']:
            dict_results[dataset][model][strategy]['val_acc'] = val_acc
            dict_results[dataset][model][strategy]['time'] = time_spent
    return model,max_acc,local_time


def train_for_document():
    models = [
    	#CNN parameters
    	{
        "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["cnn"],
        # "filter_size":[30],
        "filter_size":[30,60],
        "lr":[0.1,0.01],
        # "batch_size":[32],
        "batch_size":[32,64],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
        },
    ]

    file_summary_results = open("summary_results_3.txt", "a")
    file_local = "local_results_3.txt"

    dict_results = {}
    datasets = ["tweet_global_warm"]
    for dataset in datasets:
        print('running dataset:', dataset)
        for stragety in ['orig']:#,'phrase','tree']:
            for grid_parameters in models:
                preprocess = Preprocess(params)

                train,test = preprocess.get_train(dataset=dataset)

                model,max_acc,loc_time = grid_search_parameters(grid_parameters, train, test, dict_results, dataset,strategy=stragety)
                # myutil.write_line(dataset+'-'+model+'-'+'->'+str(max_acc)+'\t'+str(loc_time)+'s',file_local)

    pprint.pprint(dict_results)
    pprint.pprint(dict_results, file_summary_results)
    file_summary_results.close()


if __name__ == '__main__':
	# train or test
    train_for_document()
    # test_model(dataset='KBP37_longblock')

    
