from lib.KDTreeEncoding import *

import xgboost as xgb
from lib.XGBHelper import *
from lib.XGBoost_params import *
from lib.score_analysis import *

from lib.logger import logger

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from glob import glob
import pandas as pd
import pickle as pkl
import sys
from time import time
class timer:
    def __init__(self):
        self.t0=time()
        self.ts=[]
    def mark(self,message):
        self.ts.append((time()-self.t0,message))
        print('%6.2f %s'%self.ts[-1])
    def _print(self):
        for i in range(len(self.ts)):
            print('%6.2f %s'%self.ts[i])


T=timer()
            
poverty_dir=sys.argv[1]
image_dir=poverty_dir+'anon_images/'
depth=8   #for KDTree

# Loading model
all_pkl = '../data/Checkpoint.pk'
D = pkl.load(open(all_pkl, 'rb'))

D_urban = D[0]
D_rural = D[1]

scaling_mean_urban = D_urban['mean']
scaling_std_urban = D_urban['std']
scaling_mean_rural = D_rural['mean']
scaling_std_rural = D_rural['std']

bst_list_urban = [x['bst'] for x in D_urban['styled_logs'][1]['log']]
bst_list_rural = [x['bst'] for x in D_rural['styled_logs'][1]['log']]

tree_urban = D_urban['tree']
tree_rural = D_rural['tree']

T.mark('read pickle file')


# ## Iterate over test sets
folds = [{'in':'country_test_reduct.csv','out':'results_country.csv'},
    {'in':'random_test_reduct.csv','out':'results.csv'}]

for fold in folds:
    # Loading test set
    test_csv = f'../public_tables/{fold["in"]}'
    test = pd.read_csv(test_csv, index_col=0)

    test_urban = test[test['urban'] == True]
    test_rural = test[test['urban'] == False]

    out_urban = pd.DataFrame()
    out_urban['filename'] = test_urban['filename']
    out_urban['urban'] = test_urban['urban']
    out_urban['pred_wo_abstention'] = 0
    out_urban.set_index('filename', inplace=True)

    out_rural = pd.DataFrame()
    out_rural['filename'] = test_rural['filename']
    out_rural['urban'] = test_rural['urban']
    out_rural['pred_wo_abstention'] = 0
    out_rural.set_index('filename', inplace=True)


    # Encode data
    enc_data_urban = encoded_dataset(image_dir, out_urban, tree_urban, label_col='pred_wo_abstention')
    enc_data_rural = encoded_dataset(image_dir, out_rural, tree_rural, label_col='pred_wo_abstention')

    data_urban = to_DMatrix(enc_data_urban.data)
    data_rural = to_DMatrix(enc_data_rural.data)


    # Predict
    preds_urban = zeros([enc_data_urban.data.shape[0], len(bst_list_urban)])
    preds_rural = zeros([enc_data_rural.data.shape[0], len(bst_list_rural)])


    for i in range(len(bst_list_urban)):
        preds_urban[:, i] = bst_list_urban[i].predict(data_urban, output_margin=True)
    preds_urban = (preds_urban - scaling_mean_urban) / scaling_std_urban

    for i in range(len(bst_list_rural)):
        preds_rural[:, i] = bst_list_rural[i].predict(data_rural, output_margin=True)
    preds_rural = (preds_rural - scaling_mean_rural) / scaling_std_rural


    urban_mean = np.mean(preds_urban, axis=1)
    urban_std = np.std(preds_urban, axis=1)
    pred_wo_abstention_urban = (2*(urban_mean>0))-1
    pred_with_abstention_urban = copy(pred_wo_abstention_urban)
    pred_with_abstention_urban[urban_std > abs(urban_mean)] = 0

    rural_mean = np.mean(preds_rural, axis=1)
    rural_std = np.std(preds_rural, axis=1)
    pred_wo_abstention_rural = (2*(rural_mean>0))-1
    pred_with_abstention_rural = copy(pred_wo_abstention_rural)
    pred_with_abstention_rural[rural_std > abs(rural_mean)] = 0


    out_urban['pred_with_abstention'] = pred_with_abstention_urban
    out_urban['pred_wo_abstention'] = pred_wo_abstention_urban

    out_rural['pred_with_abstention'] = pred_with_abstention_rural
    out_rural['pred_wo_abstention'] = pred_wo_abstention_rural

    
    # Merge & save output
    out = pd.concat([out_urban, out_rural])
    outFile=f'../data/{fold["out"]}'
    out.to_csv(outFile)
    print('\n\n'+'-'*60)
    print(outFile)
    T.mark('generated '+outFile)


