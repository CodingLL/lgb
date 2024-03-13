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

def check_code(df):
    data_list = []
    for i in range(len(df)):
        domain_list = literal_eval(df.loc[i, "claim_estimations"])
        for d in domain_list:
            fc = d["mbfc_credibility_rating"]
            data_list.append(fc)
        # label = df.loc[i, "label"]
        # data_list.append({
        #     "label"
        # })
    print(set(data_list))
    # print(df["label"].value_counts())



def check_mv(df):
    # temp = df[df["mv"]!="baseless"]
    # print(len(temp))
    # temp["mv_label"] = temp["mv"].apply(lambda x: 1 if x == "support" else 0)
    # temp = temp[temp["mv_label"]==temp["label"]]
    # print(len(temp))

    df["mv_label"] = df["mv"].apply(lambda x: 1 if x == "support" else 0)
    df = df[df["mv_label"] == df["label"]]
    print(len(df))


def lgb_score(y_hat, data, raw_data):
    # y_true = data.get_label()
    temp = raw_data[['label']].copy()
    temp['pred']=y_hat

    temp["pred"] = temp["pred"].apply(lambda x: 1 if x> 0.5 else 0)
    length = len(temp)
    labels = list(temp["label"].values)
    preds = list(temp["pred"].values)
    temp = temp[temp["pred"] == temp["label"]]
    precision = precision_score(temp["label"], temp["pred"])
    recall = recall_score(temp["label"], temp["pred"])
    f1 = f1_score(temp["label"], temp["pred"])
    
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
    fake_acc = sum(fake_preds) / len(fake_preds)
    real_acc = sum(real_preds) / len(real_preds)
    
    # return 'acc', len(temp) / length, True
    # return 'acc', fake_acc, True
    return 'f1', r_f1, True


def get_mj(domain_list):
    sn = [d["support_or_negate"] for d in domain_list]
    support_count = sn.count("support")
    negate_count = sn.count("negate")
    if support_count == negate_count:
        return "baseless"
    elif support_count > negate_count:
        return "support"
    else:
        return "negate"


def get_media_quality(domain_list):
    mqs = [d["media_quality"] for d in domain_list]
    mqs = [m for m in mqs if m]
    if not mqs:
        return 0, 0, 0
    else:
        return np.max(mqs), np.mean(mqs), np.min(mqs), np.sum(mqs)


def get_media_quality_score(domain_list):

    score_list_3 = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        score_list_3.append((media_quality ** 0.3) * sn)
    # if not score_list_1:
    #     return 0, 0, 0, 0, 0, 0
    return np.mean(score_list_3), np.sum(score_list_3)


def get_media_quality_cscore(domain_list):

    cscore_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        confidence = d["confidence"]
        confidence = 1 if confidence == "high" else 0.6 if confidence == "medium" else 0.2
        media_quality_score = (media_quality ** 0.3) * sn * confidence

        cscore_list.append(media_quality_score)
    if not cscore_list:
        return np.mean(cscore_list), 0, 0
    return np.mean(cscore_list), np.max(cscore_list), np.min(cscore_list), np.sum(cscore_list)


def get_support_medianum(domain_list):

    support_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        # confidence = d["confidence"]
        if sn == 1:
            support_list.append(media_quality)
    if not support_list:
        return np.mean(support_list), 0, 0, 0, 0, 0, 0, 0
    return np.mean(support_list), np.max(support_list), np.min(support_list), np.sum(support_list), support_list.count(5), support_list.count(4), support_list.count(3), support_list.count(2), support_list.count(1)


def get_negate_medianum(domain_list):

    data_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        confidence = d["confidence"]
        if sn == -1 and confidence == "high":
            data_list.append(media_quality)
    if not data_list:
        return np.mean(data_list), 0, 0, 0, 0
    return np.mean(data_list), np.max(data_list), np.min(data_list), np.sum(data_list), data_list.count(5), data_list.count(4)


def get_csupport_medianum(domain_list):
    hsupport_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        confidence = d["confidence"]
        if sn == 1 and confidence == "high":
            hsupport_list.append(media_quality)
    if not hsupport_list:
        return np.mean(hsupport_list), 0, 0, 0, 0, 0
    return np.mean(hsupport_list), np.max(hsupport_list), np.min(hsupport_list), np.sum(hsupport_list), hsupport_list.count(5), hsupport_list.count(4), hsupport_list.count(3)


def get_factual_cscore(domain_list):
    score_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        fc = d["factual_reporting"].lower()
        if "very" in fc and "high" in fc:
            fc = 10
        elif "high" in fc:
            fc = 8
        elif "mixed" in fc or "n/a" in fc or "unknown" in fc:
            fc = 5
        elif "mostly factual" in fc:
            fc = 9
        elif "low" in fc and "very" in fc:
            fc = 2
        elif "low" in fc:
            fc = 3
        else:
            print(fc)
            exit(1)
        confidence = d["confidence"]
        confidence = 1 if confidence == "high" else 0.6 if confidence == "medium" else 0.2
        factual_cscore = (fc ** 0.3) * sn * confidence
        score_list.append(factual_cscore)
    if not score_list:
        return np.mean(score_list), 0, 0
    return np.mean(score_list), np.max(score_list), np.min(score_list), np.sum(score_list)


def get_factual_score(domain_list):
    score_list_3 = []

    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        fc = d["factual_reporting"].lower()
        if "very" in fc and "high" in fc:
            fc = 10
        elif "high" in fc:
            fc = 8
        elif "mixed" in fc or "n/a" in fc or "unknown" in fc:
            fc = 5
        elif "mostly factual" in fc:
            fc = 9
        elif "low" in fc and "very" in fc:
            fc = 2
        elif "low" in fc:
            fc = 3
        else:
            print(fc)
            exit(1)
        # factual_score = (fc ** 0.3) * sn
        # score_list.append(factual_score)

        score_list_3.append((fc ** 0.3) * sn)
    return np.mean(score_list_3), np.sum(score_list_3)


def get_factual(domain_list):
    score_list = []
    for d in domain_list:
        fc = d["factual_reporting"].lower()
        if "very" in fc and "high" in fc:
            fc = 10
        elif "high" in fc:
            fc = 8
        elif "mixed" in fc or "n/a" in fc or "unknown" in fc:
            fc = 5
        elif "mostly factual" in fc:
            fc = 9
        elif "low" in fc and "very" in fc:
            fc = 2
        elif "low" in fc:
            fc = 3
        else:
            print(fc)
            exit(1)
        score_list.append(fc)

    return np.mean(score_list), np.max(score_list), np.min(score_list), np.sum(score_list)


def get_squalityabove3(domain_list):

    score_list = []
    hscore_list = []
    nscore_list = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality or media_quality < 3:
            media_quality = 3
        confidence = d["confidence"]
        if sn == 1:
            score_list.append(1)
        if sn == 1 and confidence == "high":
            hscore_list.append(1)
        if sn == -1:
            score_list.append(1)
    if not score_list:
        sum_score = 0
    else:
        sum_score = np.sum(score_list)
    if not hscore_list:
        hsum_score = 0
    else:
        hsum_score = np.sum(score_list)
    return sum_score, hsum_score

def get_media_quality_score(domain_list):

    score_list_3 = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        score_list_3.append((media_quality ** 0.3) * sn)
    # if not score_list_1:
    #     return 0, 0, 0, 0, 0, 0
    return np.mean(score_list_3), np.sum(score_list_3)


def get_full_code_score3(domain_list):

    score_list_3 = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        full_code = d['full_code']
        if '1' in full_code or '2' in full_code:
            full_code_r = 1
        elif 'BLOCK' in full_code or 'Foreign' in full_code:
            full_code_r = -1
        else:
            full_code_r = 0.5
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        score = (media_quality ** 0.3) * sn
        if score > 0 and full_code_r < 1:
            score *= full_code_r
            
        score_list_3.append(score)
    return np.mean(score_list_3), np.sum(score_list_3)

def get_full_code_score(domain_list):

    score_list_3 = []
    for d in domain_list:
        sn = d["support_or_negate"]
        sn = 1 if sn == "support" else -1 if sn == "negate" else 0
        full_code = d['full_code']
        if ('1' in full_code or '2' in full_code) and '10' not in full_code:
            full_code_r = 1
        elif 'BLOCK' in full_code in full_code:
            full_code_r = -1
        else:
            full_code_r = 0.5
        media_quality = d["media_quality"]
        if not media_quality:
            media_quality = 3
        score = sn
        if score > 0 and full_code_r < 1:
            score *= full_code_r
            
        score_list_3.append(score)
    return np.mean(score_list_3), np.sum(score_list_3)

def get_mv_mean(domain_list):
    support_or_negate_stat = {'support': 0, 'baseless': 0, 'negate': 0}
    for d in domain_list:
        sn = d["support_or_negate"]
        media_quality = d["media_quality"]
        # if not media_quality:
        #     continue
        support_or_negate_stat[f'{sn}'] += 1
    return np.mean(support_or_negate_stat['support']), np.mean(support_or_negate_stat['baseless']), np.mean(support_or_negate_stat['negate'])


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = ArgumentParser()
    setup_seed(1992)
    
    parser.add_argument('--filepath', type=str, default="data/buzzfeed_1267_nbmodel.csv")
    parser.add_argument('--filepath2', type=str, default=None)
    parser.add_argument('--outpath', type=str, default="exp/buzzfeed_1267_nbmodel")
    args = parser.parse_args()
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    folds_num = 5
    df = pd.read_csv(args.filepath)
    if args.filepath2:
        df2 = pd.read_csv(args.filepath)
        df2 = df2.loc[df2['label']==0, :]
        df2['fold'] = -1
    #
    # check_code(df)
    # exit(1)

    df["label"] = df["label"].apply(lambda x: 1 if x else 0)
    df["claim_estimations"] = df["claim_estimations"].apply(lambda x: str([d for d in literal_eval(x) if d not in ["newbreakapp.com"]]))
    df["mv"] = df["claim_estimations"].apply(lambda x: get_mj(literal_eval(x)))
    # check_mv(df)
    # exit(1)
    df[["media_quality_max", "media_quality_mean", "media_quality_min", "media_quality_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_media_quality(literal_eval(x))))
    df[["media_quality_score3_mean", "media_quality_score3_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_media_quality_score(literal_eval(x))))
    df[["media_quality_cscore_mean", "media_quality_cscore_max", "media_quality_cscore_min", "media_quality_cscore_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_media_quality_cscore(literal_eval(x))))
    df[["factual_score3_mean", "factual_score3_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_factual_score(literal_eval(x))))
    df[["factual_cscore_mean", "factual_cscore_max", "factual_cscore_min", "factual_cscore_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_factual_cscore(literal_eval(x))))
    df[["factual_mean", 'factual_max', "factual_min", "factual_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_factual(literal_eval(x))))
    df[["support_quality_mean", "support_quality_max", "support_quality_min", "support_quality_sum", "support_quality_5", "support_quality_4", "support_quality_3", "support_quality_2", "support_quality_1"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_support_medianum(literal_eval(x))))
    # df[["hsupport_quality_mean", "hsupport_quality_max", "hsupport_quality_min", "hsupport_quality_5", "hsupport_quality_4", "hsupport_quality_3"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_csupport_medianum(literal_eval(x))))
    df[["hnegate_quality_mean", "hnegate_quality_max", "hnegate_quality_min", "hnegate_quality_sum", "hnegate_quality_5", "hnegate_quality_4"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_negate_medianum(literal_eval(x))))
    df[["squalityabove3", "hsqualityabove3"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_squalityabove3(literal_eval(x))))
    df[["full_code_score3_mean", "full_code_score3_mean_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_full_code_score3(literal_eval(x))))
    df[["full_code_score_mean", "full_code_score_mean_sum"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_full_code_score(literal_eval(x))))
    df[["support_mean", "baseless_mean", "negate_mean"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_mv_mean(literal_eval(x))))
    # df["samenews"]

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
            # 'max_depth': 12,
            "learning_rate": 0.04,
            "num_iterations": 1500,
            # 'metric': 'custom',
            'metric': 'custom',
            # 'colsample_bytree': 0.78,
            # 'colsample_bynode': 0.8,
            # 'feature_fraction': 0.9,
            # 'lambda_l2': 0.1,
            # 'subsample': 0.35,
            'scale_pos_weight': 0.01,
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