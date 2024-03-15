import os, json
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
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


if __name__ == '__main__':
    parser = ArgumentParser()
    setup_seed(1992)
    
    parser.add_argument('--filepath', type=str, default="data/overall_training_0229_fold.csv")
    parser.add_argument('--filepath2', type=str, default=None)
    parser.add_argument('--testpath', type=str, default="data/buzzfeed182_gpt")
    parser.add_argument('--outpath', type=str, default="exp/buzzfeed182_gpt")
    args = parser.parse_args()
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    folds_num = 1
    df = pd.read_csv(args.filepath)
    if args.filepath2:
        df2 = pd.read_csv(args.filepath)
        df2 = df2.loc[df2['label']==0, :]
        df2['fold'] = -1
    else:
        # testpath = '/home/services/siliang/lgb/data/buzzfeed_1267_gpt.csv'
        testpath = 'data/buzzfeed182_gpt.csv'
        df2 = pd.read_csv(testpath)
    #
    # check_code(df)
    # exit(1)

    df = feature_engineer(df)
    df2 = feature_engineer(df2)

    train_columns = ["mv", "media_quality_max", "media_quality_score3_mean", "media_quality_cscore_mean",
                     "media_quality_cscore_min", "factual_cscore_mean", "factual_score3_mean", "factual_max",
                     "support_quality_mean", "support_quality_min", "support_quality_4",
                    "hnegate_quality_max", "hnegate_quality_4", "media_quality_sum", "media_quality_cscore_sum",
                    "factual_cscore_sum", "factual_sum", "squalityabove3", "full_code_score3_mean", "full_code_score3_mean_sum",
                    "full_code_score_mean", "full_code_score_mean_sum", 
                     ]
    
    train_columns = ["mv", "media_quality_score3_mean", "media_quality_cscore_mean",
                     "factual_cscore_mean", "factual_score3_mean", "factual_max",
                     "support_quality_mean", "support_quality_4",
                     "media_quality_sum", "media_quality_cscore_sum",
                     "factual_cscore_sum", "factual_sum", "squalityabove3", "full_code_score3_mean", "full_code_score3_mean_sum",
                     "full_code_score_mean", "full_code_score_mean_sum", "support_mean", "baseless_mean", "negate_mean"
                     ]

    # with open(str(args.outpath + "/" + ('features.pkl')), 'wb') as f:
    #     pickle.dump(train_columns, f)
    print(df)
    feature_importance_df = pd.DataFrame()
    fold_feats = pd.DataFrame()
    for fold in range(folds_num):
        print("****fold_%s****"%fold)

        # train_feat = df.loc[df["fold"]!=fold]
        # valid_feat = df.loc[df["fold"]==fold]
        
        train_feat = df
        valid_feat = df2
        
        # num_samples_label_1 = (df2['label'] == 1).sum()
        # sampled_df2 = df2[df2['label'] == 0].sample(n=num_samples_label_1, replace=True)
        # valid_feat = pd.concat([df2[df2['label'] == 1], sampled_df2])
        print(valid_feat.label.value_counts())
        
        if args.filepath2:
            train_feat = pd.concat([train_feat, df2])
        
        # 将标签为 0 的数据单独提取出来
        # label_0_data = train_feat[train_feat["label"] == 0]
        # resampled_label_0_data = label_0_data.sample(n=3*len(label_0_data), replace=True)
        # train_feat = pd.concat([resampled_label_0_data, train_feat[train_feat["label"] == 1]])

        # train_label = df.loc[df["fold"]!=fold]
        # valid_label = df.loc[df["fold"]==fold, "label"]

        # print(train_feat.head())
    #     train_feat = train_feat.fillna(0)
    #     valid_feat = valid_feat.fillna(0)
    #
    #     total = 0
    #     recall = 0

        # create dataset

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

        param = {
            'objective': 'binary',
            # 'num_leaves': 62,
            # 'max_depth': 6,
            "learning_rate": 0.01,
            "num_iterations": 1500,
            # 'metric': 'custom',
            'metric': 'auc',
            # 'colsample_bytree': 0.78,
            # 'colsample_bynode': 0.8,
            # 'feature_fraction': 0.9,
            # 'lambda_l2': 0.1,
            # 'subsample': 0.35,
            'scale_pos_weight': 1,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 1,
            "verbosity": -1,
            # "seed": 1992, d
            # "lambda_l1": 0.4,
            # "lambda_l2": 0.4
            # "early_stopping_round": 50
        }
        
        # lgb = LGBMClassifier(**param, random_state=1992, scale_pos_weight=0.01)
        # lgb2 = LGBMClassifier(**param, random_state=1992, scale_pos_weight=1)
        # cat = CatBoostClassifier(iterations=1500,
        #                         verbose=0,
        #                         random_seed=1992,
        #                         learning_rate=0.05,
        #                         subsample=0.35,
        #                         scale_pos_weight=0.01,
        #                         # allow_const_label=True,
        #                         # loss_function = 'CrossEntropy',
        #                         loss_function='Logloss',
        #                         cat_features=cat_features
        #                         )
        # weights = [0.0, 1.0]
        # weights = [0.5, 0.5]
        # voting_clf = VotingClassifier(estimators=[('lgb', lgb),
        #                                           ('lgb2', lgb2),
        #                                           ],
        #                               weights=weights, 
        #                               voting='soft',
        #                               n_jobs=-1)
        
        # voting_clf.fit(train_feat[train_columns], train_feat["label"])
        # bst = voting_clf
        
        bst = lgb.train(param, train_data, valid_sets=[valid_data],
                        early_stopping_rounds=100, verbose_eval=10,
                        feval=lambda p, d: [lgb_score(p, d, valid_feat),],)
        
        bst.save_model(str(args.outpath+"/"+('model_%s'%fold)))
        pred = bst.predict(valid_feat[train_columns])

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_columns
        fold_importance_df["importance"] = bst.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        valid_feat['preds'] = pred
        valid_feat["fold"] = fold
                                                                                                                
        fold_feats = pd.concat([fold_feats, valid_feat[["mv",  "preds", "fold", "label"]]], axis=0)
        print(fold_feats.shape)

    mean_gain = feature_importance_df[['importance', 'Feature']].groupby('Feature').mean()
    mean_gain.sort_values("importance", ascending=False).to_csv(str(args.outpath + "/" + 'feature_importance.csv'))
    fold_feats["pred_label"] = fold_feats["preds"].apply(lambda x: 1 if x>0.5 else 0)
    accuracy = fold_feats[fold_feats["label"] == fold_feats["pred_label"]]
    
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
    print('pos_precision: {:.4f}, pos_recall: {:.4f}, pos_f1: {:.4f}'.format(precision, recall, f1))
    
    r_labels = [int(not i) for i in labels]
    r_preds = [int(not i) for i in preds]
    r_precision = precision_score(r_labels, r_preds)
    r_recall = recall_score(r_labels, r_preds)
    r_f1 = f1_score(r_labels, r_preds)
    print('neg_precision: {:.4f}, neg_recall: {:.4f}, neg_f1: {:.4f}'.format(r_precision, r_recall, r_f1))
    print('accuracy: {:.4f}, fake_acc: {:.4f}, real_acc: {:.4f}'.format(accuracy, fake_acc, real_acc))

    fold_feats.to_csv(str(args.outpath + "/" + 'result.csv'), index=False)