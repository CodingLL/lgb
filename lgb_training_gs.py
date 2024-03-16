import os, json

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')
from utils import setup_seed
from feature_engineer import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.linear_model import SGDClassifier

def lgb_score(y_hat, data, raw_data):
    # y_true = data.get_label()
    temp = raw_data[['label']].copy()
    temp['pred']=y_hat

    temp["pred"] = temp["pred"].apply(lambda x: 1 if x> 0.5 else 0)
    length = len(temp)
    labels = list(temp["label"].values)
    preds = list(temp["pred"].values)
    temp = temp[temp["pred"] == temp["label"]]
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    r_labels = [int(not i) for i in labels]
    r_preds = [int(not i) for i in preds]
    r_precision = precision_score(r_labels, r_preds)
    r_recall = recall_score(r_labels, r_preds)
    r_f1 = f1_score(r_labels, r_preds)
    
    accuracy = 0
    fake_preds, real_preds = [], []
    for i in range(len(labels)):
        if labels[i] == 0:
            fake_preds.append(not preds[i])
        else:
            real_preds.append(preds[i])
        if labels[i] == preds[i]:
            accuracy += 1
    accuracy /= len(labels)
    # fake_acc = sum(fake_preds) / len(fake_preds)
    # real_acc = sum(real_preds) / len(real_preds)
    
    # return 'acc', len(temp) / length, True
    # return 'acc', fake_acc, True
    return 'f1', r_f1, True

def f1_eval_metric(y_true, y_pred):
    # 在这里定义你的自定义评估指标计算逻辑
    # 这里使用 F1 分数作为示例
    y_pred = [int(i>0.5) for i in y_pred]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(precision, recall, f1_score)
    # exit(1)
    r_labels = [int(not i) for i in y_true]
    r_preds = [int(not i) for i in y_pred]
    r_precision = precision_score(r_labels, r_preds)
    r_recall = recall_score(r_labels, r_preds)
    r_f1 = f1_score(r_labels, r_preds)
    return 'custom_f1_score', r_f1, True

def f1_eval_metric2(y_true, y_pred):
    # 在这里定义你的自定义评估指标计算逻辑
    # 这里使用 F1 分数作为示例
    y_pred = [int(i>0.5) for i in y_pred]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(precision, recall, f1_score)
    # exit(1)
    r_labels = [int(not i) for i in y_true]
    r_preds = [int(not i) for i in y_pred]
    r_precision = precision_score(r_labels, r_preds)
    r_recall = recall_score(r_labels, r_preds)
    r_f1 = f1_score(r_labels, r_preds)
    return 'custom_f1_score', f1, True

class Catf1(object):
    def __init__(self, f1_type='neg'):
        self.f1_type = f1_type
        
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        if self.f1_type == 'neg':
            return f1_eval_metric(target, preds)[1], 0
        elif self.f1_type == 'pos':
            return f1_eval_metric2(target, preds)[1], 0


if __name__ == '__main__':
    parser = ArgumentParser()
    setup_seed(1992)
    
    parser.add_argument('--filepath', type=str, default="data/buzzfeed182_gpt_v2.csv")
    parser.add_argument('--filepath2', type=str, default=None)
    parser.add_argument('--gpt_rationale', type=int, default=0)
    parser.add_argument('--outpath', type=str, default="exp/buzzfeed182_gpt_5fold")
    args = parser.parse_args()
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    folds_num = 5
    if 'jsonl' in args.filepath:
        data = []
        with open(args.filepath) as f:
            for line in f:
                dic = json.loads(line)
                if isinstance(dic['gpt_rating']['rating'], str):
                    dic['gpt_rating']['rating'] = 5
                data.append(dic)
        df = pd.DataFrame(data)
    else: 
        df = pd.read_csv(args.filepath)
    if args.filepath2:
        df2 = pd.read_csv(args.filepath)
        df2 = df2.loc[df2['label']==0, :]
        df2['fold'] = -1
    #
    # check_code(df)
    # exit(1)
    df = feature_engineer(args, df)

    # train_columns = ["mv", "media_quality_max", "media_quality_score3_mean", "media_quality_cscore_mean",
    #                  "media_quality_cscore_min", "factual_cscore_mean", "factual_score3_mean", "factual_max",
    #                  "support_quality_mean", "support_quality_min", "support_quality_4",
    #                 "hnegate_quality_max", "hnegate_quality_4", "media_quality_sum", "media_quality_cscore_sum",
    #                 "factual_cscore_sum", "factual_sum", "squalityabove3", "full_code_score3_mean", "full_code_score3_mean_sum",
    #                 "full_code_score_mean", "full_code_score_mean_sum", 
    #                  ]
    
    train_columns = ["media_quality_score3_mean", "media_quality_cscore_mean",
                     "factual_cscore_mean", "factual_score3_mean",
                     "support_quality_mean",
                     "media_quality_sum", "media_quality_cscore_sum",
                     "factual_cscore_sum", "factual_sum", "squalityabove3", 
                     # "full_code_score3_mean", "full_code_score3_mean_sum",
                     # "full_code_score_mean", "full_code_score_mean_sum", 
                     "support_mean", "baseless_mean", "negate_mean", 
                     # "support_num", "baseless_num", "negate_num", 
                     # "gpt_rationale_score",
                     ]
    if args.gpt_rationale:
        train_columns.append("gpt_rationale_score")

    # with open(str(args.outpath + "/" + ('features.pkl')), 'wb') as f:
    #     pickle.dump(train_columns, f)
    print(df)
    feature_importance_df = pd.DataFrame()
    fold_feats = pd.DataFrame()
    # param = {
    #     'objective': 'binary', # 'num_leaves': 62, # 'max_depth': 12,
    #     "learning_rate": 0.05, "num_iterations": 1500, # "n_estimators": 2500, 
    #     'metric': 'custom',
    #     # 'metric': 'auc',
    #     'scale_pos_weight': 1, "verbosity": -1,
    # }
    param2 = {
        'objective': 'binary', 'boosting_type': 'gbdt',# 'num_leaves': 62, # 'max_depth': 12,
        "learning_rate": 0.05, "num_iterations": 1000, # "n_estimators": 2500, 
        'metric': 'custom',
        # 'metric': 'auc',
        'scale_pos_weight': 1, "verbosity": -1,
        'colsample_bytree': 0.78, 'colsample_bynode': 0.8,
    }
    
    
    paramrf = {
        'objective': 'binary', 'boosting_type': 'rf',# 'num_leaves': 62, # 'max_depth': 12,
        "learning_rate": 0.05, "num_iterations": 1000, # "n_estimators": 2500, 
        'metric': 'custom',
        # 'metric': 'auc',
        'scale_pos_weight': 1, "verbosity": -1,
        'colsample_bytree': 0.78, 'colsample_bynode': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 10
    }
    
    # paramdart = {
    #     'objective': 'binary', 'boosting_type': 'dart',# 'num_leaves': 62, # 'max_depth': 12,
    #     "learning_rate": 0.05, "num_iterations": 1000, # "n_estimators": 2500, 
    #     'metric': 'custom',
    #     # 'metric': 'auc',
    #     'scale_pos_weight': 1, "verbosity": -1,
    #     'colsample_bytree': 0.78, 'colsample_bynode': 0.8,
    # }
    
    # param_dist = {
    #     'num_leaves': [50, 60, 70, 80, 90, 100],
    #     'max_depth': [6, 7, 8, 9, 10, 11, 12],
    #     'learning_rate': [0.04, 0.05, 0.06],
    #     'n_estimators': sp_randint(50, 500)
    # }
    param_test ={# 'num_leaves': [6, 8, 10, 12],
            'min_child_samples': [5, 10, 20],  # 20
            'min_child_weight': [1e-3, 1e-1, 1, 1e1, 1e3], # 1
            'colsample_bytree': [0.78],
            'colsample_bynode': [0.78],
        #  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        #  'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
            "learning_rate": [0.05],
            }
    
    lgbcls = LGBMClassifier(**param2, random_state=1992)    
    # politifact: {'min_child_samples': 20, 'min_child_weight': 1}
    # buzzfeed: {'min_child_samples': 10, 'min_child_weight': 0.001}
    # gs = RandomizedSearchCV(estimator=lgbcls, param_distributions=param_test, n_iter=10, cv=5)
    gs = GridSearchCV(estimator=lgbcls, param_grid=param_test, cv=3)
    
    # fold 0 的数据来调参
    fold = 0
    train_feat = df.loc[df["fold"]!=fold]
    valid_feat = df.loc[df["fold"]==fold]
    gs.fit(train_feat[train_columns], train_feat["label"], 
            eval_set=[(valid_feat[train_columns], valid_feat["label"])], 
            eval_metric=f1_eval_metric, 
            early_stopping_rounds=100,
            verbose=10)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
    
    for fold in range(folds_num):
        # if fold != 2:
        #     continue
        print("****fold_%s****"%fold)

        train_feat = df.loc[df["fold"]!=fold]
        valid_feat = df.loc[df["fold"]==fold]
        if args.filepath2:
            train_feat = pd.concat([train_feat, df2])
        
        # 将标签为 0 的数据单独提取出来
        # label_0_data = train_feat[train_feat["label"] == 0]
        # resampled_label_0_data = label_0_data.sample(n=3*len(label_0_data), replace=True)
        # train_feat = pd.concat([resampled_label_0_data, train_feat[train_feat["label"] == 1]])

        # train_label = df.loc[df["fold"]!=fold]
        # valid_label = df.loc[df["fold"]==fold, "label"]

        # print(train_feat.head())
        # train_feat = train_feat.fillna(0)
        # valid_feat = valid_feat.fillna(0)
    #
    #     total = 0
    #     recall = 0

        # create dataset
        cat_features = []
        if 'mv' in train_columns:
            
            cat_features = [
                "mv"
            ]

            train_feat[cat_features] = train_feat[cat_features].astype('category')
            valid_feat[cat_features] = valid_feat[cat_features].astype('category')

        train_data = lgb.Dataset(train_feat[train_columns], label=train_feat["label"],
                                 feature_name=train_columns,
                                 categorical_feature=cat_features
                                 )

        valid_data = lgb.Dataset(valid_feat[train_columns], label=valid_feat["label"],
                                 feature_name=train_columns,
                                 categorical_feature=cat_features,
                                 reference=train_data
                                 )

        clf_final = lgb.LGBMClassifier(**lgbcls.get_params())
        clf_final.fit(train_feat[train_columns], train_feat["label"], 
            eval_set=[(valid_feat[train_columns], valid_feat["label"])], 
            eval_metric=f1_eval_metric2, 
            early_stopping_rounds=100,
            verbose=10)
        
        
        pred = clf_final.predict(valid_feat[train_columns])
        

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_columns
       #  fold_importance_df["importance"] = bst.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        valid_feat['preds'] = pred
        valid_feat["fold"] = fold
                                                                                                                
        if 'gpt' in list(valid_feat.columns):                                                                                               
            fold_feats = pd.concat([fold_feats, valid_feat[["id", "url", "title", "text", "gpt", "claim_estimations", "support_mean", \
                                                            "baseless_mean", "negate_mean", "mv",  "preds", "fold", "label"]]], axis=0)
        else:
            fold_feats = pd.concat([fold_feats, valid_feat[["id", "url", "title", "text", "nbmodel", "claim_estimations", "support_mean", \
                                                            "baseless_mean", "negate_mean", "mv",  "preds", "fold", "label"]]], axis=0)
        print(fold_feats.shape)

    # mean_gain = feature_importance_df[['importance', 'Feature']].groupby('Feature').mean()
    # mean_gain.sort_values("importance", ascending=False).to_csv(str(args.outpath + "/" + 'feature_importance.csv'))
    fold_feats["pred_label"] = fold_feats["preds"].apply(lambda x: 1 if x>0.5 else 0)
    # accuracy = fold_feats[fold_feats["label"] == fold_feats["pred_label"]]
    
    labels = list(fold_feats["label"].values)
    preds = list(fold_feats["pred_label"].values)
    fake_preds = []
    real_preds = []
    accuracy = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            fake_preds.append(not preds[i])
        else:
            real_preds.append(preds[i])
        if labels[i] == preds[i]:
            accuracy += 1
    accuracy /= len(labels)
    fake_acc = sum(fake_preds) / len(fake_preds)
    real_acc = sum(real_preds) / len(real_preds)
    
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    
    print('micro_precision: {:.4f}, micro_recall: {:.4f}, micro_f1: {:.4f}'.format(micro_precision, micro_recall, micro_f1))
    print('pos_precision: {:.4f}, pos_recall: {:.4f}, pos_f1: {:.4f}'.format(precision, recall, f1))
    
    r_labels = [int(not i) for i in labels]
    r_preds = [int(not i) for i in preds]
    r_precision = precision_score(r_labels, r_preds)
    r_recall = recall_score(r_labels, r_preds)
    r_f1 = f1_score(r_labels, r_preds)
    print('neg_precision: {:.4f}, neg_recall: {:.4f}, neg_f1: {:.4f}'.format(r_precision, r_recall, r_f1))
    print('accuracy: {:.4f}, fake_acc: {:.4f}, real_acc: {:.4f}'.format(accuracy, fake_acc, real_acc))

    fold_feats.to_csv(str(args.outpath + "/" + 'result.csv'), index=False)
    with open(str(args.outpath + "/" + 'metrics.txt'), 'a') as f:
        f.write('pos_precision: {:.4f}, pos_recall: {:.4f}, pos_f1: {:.4f}'.format(precision, recall, f1) + '\n')
        f.write('neg_precision: {:.4f}, neg_recall: {:.4f}, neg_f1: {:.4f}'.format(r_precision, r_recall, r_f1) + '\n')
        f.write('accuracy: {:.4f}, fake_acc: {:.4f}, real_acc: {:.4f}'.format(accuracy, fake_acc, real_acc) + '\n')
    # print(valid_feat.columns)
    
    