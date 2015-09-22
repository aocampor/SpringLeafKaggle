#from ROOT import *
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
import random
import math
from datetime import datetime

def SettingsForXGBoost( eta = 0.2, gamma = 1, min_child_weight = 6, 
                        max_depth = 30, max_delta_step = 2):
    params = {}
    params["booster"] = "gbtree"
    #params["booster"] = "gblinear"
    params["objective"] = "binary:logistic"
    params["bst:eta"] = eta
    params["bst:gamma"] = gamma
    #params["lambda"] = lambda1    
    params["bst:min_child_weight"] = min_child_weight
    #params["subsample"] = subsample
    #params["colsample_bytree"] = colsample_bytree
    #params["scale_pos_weight"] = scale_pos_weight
    params["silent"] = 1
    params["bst:max_depth"] = max_depth
    params["bst:max_delta_step"] = max_delta_step
    params["nthread"] = 16
    plst = list(params.items())
    return params, plst


if __name__ == "__main__":
    startTime = datetime.now()
    predict = True
    csvmap = {}
    print 'loading'
    csvmap['train'] = pd.read_csv('Data/train.csv', error_bad_lines = False , index_col=False, dtype='unicode')
    print 'loaded train'
    if(predict):
        csvmap['test'] = pd.read_csv('Data/test.csv', error_bad_lines = False , index_col=False, dtype='unicode')
    print 'loaded csvs'
    train_names = csvmap['train'].columns.values.tolist()
    types = csvmap['train'].dtypes
    print types
    test_names = 0
    types_test = 0
    if(predict):
        test_names = csvmap['test'].columns.values.tolist()
        types_test = csvmap['test'].dtypes
    print types_test

    csvmap['train'].replace( np.nan, -999999, regex=True, inplace=True)    
    if(predict):
        csvmap['test'].replace( np.nan, -999999, regex=True, inplace=True)    

    dates = []
    i = 0
    for item in types:
        #print item
        if(item == 'object'):
            dates.append(i)
        i = i + 1
    for item in dates:
        lbl =  preprocessing.LabelEncoder()
        lbl.fit( list( csvmap['train'][train_names[item]] ) )
        csvmap['train'][train_names[item]] = lbl.transform( csvmap['train'][train_names[item]] )

    
    if(predict):
        dates = []
        i=0
        for item in types_test:
            #print item
            if(item == 'object'):
                dates.append(i)
            i = i + 1
        for item in dates:
            lbl =  preprocessing.LabelEncoder()
            lbl.fit( list( csvmap['test'][test_names[item]] ) )
            csvmap['test'][test_names[item]] = lbl.transform( csvmap['test'][test_names[item]] )


    train = csvmap['train']
    tags = csvmap['train']['target']
    totag = 0
    if(predict):
        totag = csvmap['test'] 
    params, plst = SettingsForXGBoost( 0.02, 0.02, 25, 200, 0.9 )
    num_rounds = 200
    rounds  = 40
    rows = random.sample( train.index, len(tags)/2)
    df_train = train.ix[rows]
    df_test = train.drop(rows)
    #ids_train = df_train['ID']
    tags_train = df_train['target']
    df_train = df_train.drop(['target'], axis = 1)
    #ids_test = df_test['ID']
    tags_test = df_test['target']
    if(predict):
        idtotag = totag['ID']
    df_test = df_test.drop(['target'], axis = 1)
    df_train = np.array(df_train)
    tags_train = np.array(tags_train)
    df_test = np.array(df_test)
    tags_test = np.array(tags_test)
    idtotag = 0
    if(predict):
        totag = np.array(totag)
        idtotag = np.array(idtotag)
    xgtrain = xgb.DMatrix(df_train, label=tags_train, missing = np.nan)
    xgcontr = xgb.DMatrix(df_test, missing = np.nan)
    xtotag = 0
    if(predict):
        xtotag = xgb.DMatrix(totag, missing = np.nan)
    model = xgb.train(plst, xgtrain, num_rounds)
    preds = model.predict(xgcontr) 
    tagging = 0
    if(predict):
        tagging = model.predict(xtotag)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    sub = np.subtract( tags_test, preds )
    sub = np.square( sub )
    sumerr = np.sum(sub)
    print 'Error on prediction ' , math.sqrt(sumerr)
    plt.scatter(tags_test, preds , marker="o")
    plt.xlabel('Real Cost', fontsize=18)
    plt.ylabel('Predicted cost', fontsize=16)
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    plt.savefig('scatterplot_bp.png')
    if(predict):
        outpreds = pd.DataFrame({"ID": idtotag, "target": tagging})
        outpreds.to_csv('benchmark.csv', index=False)

    #print tags_test, preds
    print 'It took ' , datetime.now() - startTime

