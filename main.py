
# -*- coding: utf-8 -*-
from  Params import Params
import argparse
from preprocessing import Process
import models
from keras.models import load_model
import os
import numpy as np
import itertools
from token_selection import TokenSelection

from sklearn.utils import shuffle
import pprint
import util
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

dict_semeval={'cause-effect': 0, 'component-whole': 1, 'content-container': 2, 'entity-destination': 3, 'entity-origin': 4, 'instrument-agency': 5, 'member-collection': 6, 'message-topic': 7, 'other': 8, 'product-producer': 9}
dict_kbp={'no_relation': 0, 'org:alternate_names': 1, 'org:city_of_headquarters': 2, 'org:country_of_headquarters': 3, 'org:founded': 4, 'org:founded_by': 5, 'org:members': 6, 'org:stateorprovince_of_headquarters': 7, 'org:subsidiaries': 8, 'org:top_members': 9, 'per:alternate_names': 10, 'per:cities_of_residence': 11, 'per:countries_of_residence': 12, 'per:country_of_birth': 13, 'per:employee_of': 14, 'per:origin': 15, 'per:spouse': 16, 'per:stateorprovinces_of_residence': 17, 'per:title': 18}
# {0: 'cause-effect', 1: 'component-whole', 2: 'content-container', 3: 'entity-destination', 4: 'entity-origin', 5: 'instrument-agency', 6: 'member-collection', 7: 'message-topic', 8: 'other', 9: 'product-producer'}
# ['cause-effect' 'component-whole' 'content-container' 'entity-destination' 'entity-origin' 'instrument-agency' 'member-collection' 'message-topic' 'other' 'product-producer']


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # fig= plt.figure(figsize=(7,7))
    fig, ax = plt.subplots(figsize=(9,9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def draw_result(predicted, val):
    # print('loss:',log_loss(val,predicted)) 
    ground_label = np.array(val).argmax(axis=1)
    predicted_label = np.array(predicted).argmax(axis=1)
    print('F1 macro:',f1_score(predicted_label ,ground_label,average='macro'))
    print('F1 micro:',f1_score(predicted_label ,ground_label,average='micro'))

    print('accuracy:',accuracy_score(predicted_label ,ground_label))
    print(confusion_matrix(predicted_label ,ground_label))
    plot_confusion_matrix(ground_label, predicted_label, classes=list(dict_kbp.values()), normalize=True,
                      title='Normalized confusion matrix')
    plt.show()
    plt.savefig('foo.png')

    print(classification_report(ground_label, predicted_label))

def test_model(dataset):
    token_select = TokenSelection(params)
    train,test = token_select.get_train_from_text(dataset=dataset)
    predicted = emsemble(test)    
    draw_result(predicted,test[1])
    


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
        val_acc, time_spent, model = model.train_relation(train, dev=test,dataset=dataset)
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
        "lr":[0.1],
        # "batch_size":[32],
        "batch_size":[32,64],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
        },
    	# RNN parameters
        # {
        #     "cell_type": ["gru"],
        #     "hidden_unit_num": [100],
        #     "dropout_rate": [0.2],  # ,0.5,0.75,0.8,1]    ,
        #     "model": ["bilstm_2L"],
        #     # "contatenate":[0],
        #     "lr": [0.1],
        #     "batch_size": [32,64],
        #     # "validation_split":[0.05,0.1,0.15,0.2],
        #     "validation_split": [0.1],
        # }
    ]

    file_summary_results = open("summary_results_3.txt", "a")
    file_local = "local_results_3.txt"

    dict_results = {}
    datasets = ["KBP37_longblock"]
    for dataset in datasets:
        print('running dataset:', dataset)
        for stragety in ['orig']:#,'phrase','tree']:
            for grid_parameters in models:
                token_select = TokenSelection(params)
                train,test = token_select.get_train_from_text(dataset=dataset)

                model,max_acc,loc_time = grid_search_parameters(grid_parameters, train, test, dict_results, dataset,strategy=stragety)
                util.write_line(dataset+'-'+model+'-'+'->'+str(max_acc)+'\t'+str(loc_time)+'s',file_local)

    pprint.pprint(dict_results)
    pprint.pprint(dict_results, file_summary_results)
    file_summary_results.close()


if __name__ == '__main__':
	# train or test
    # train_for_document()
    test_model(dataset='KBP37_longblock')

    
