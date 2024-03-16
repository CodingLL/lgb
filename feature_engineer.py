from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from ast import literal_eval

full_code_map = {'Block': 0.1, }

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
        return 0, 0, 0, 0
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
    # print(support_list)
    if not support_list:
        return np.mean(support_list), 0, 0, 0, 0, 0, 0, 0, 0
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
        
        # full_code = d["full_code"]
        # full_code = 0.1 if "BLOCK" in full_code else 1
        
        factual_cscore = (fc ** 1) * sn * confidence # (fc ** 0.3) * sn * confidence
        score_list.append(factual_cscore)
    if not score_list:
        return 0, 0, 0, 0
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

        # score_list_3.append((fc ** 0.3) * sn)
        score_list_3.append(fc* sn)
    if not score_list_3:
        return 0, 0
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
    if not score_list:
        return 0, 0, 0, 0
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
        # score_list_3.append((media_quality ** 0.3) * sn)
        score_list_3.append(media_quality * sn)
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
        # score = (media_quality ** 0.3) * sn
        score = media_quality * sn
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
    support_or_negate_stat = {'support': 0, 'full support': 0, 'partial support': 0, 'baseless': 0, 'negate': 0}
    n = len(domain_list)
    for d in domain_list:
        sn = d["support_or_negate"]
        media_quality = d["media_quality"]
        # if not media_quality:
        #     continue
        support_or_negate_stat[f'{sn}'] += 1
    support_or_negate_stat['support'] = support_or_negate_stat['full support'] + support_or_negate_stat['partial support']
    if n > 0:
        return support_or_negate_stat['support']/n, support_or_negate_stat['baseless']/n, support_or_negate_stat['negate']/n
    else:
        return 0, 0, 0

def get_mv_sum(domain_list):
    support_or_negate_stat = {'support': 0, 'full support': 0, 'partial support': 0, 'baseless': 0, 'negate': 0}
    n = len(domain_list)
    for d in domain_list:
        sn = d["support_or_negate"]
        media_quality = d["media_quality"]
        # if not media_quality:
        #     continue
        support_or_negate_stat[f'{sn}'] += 1
    support_or_negate_stat['support'] = support_or_negate_stat['full support'] + support_or_negate_stat['partial support']
    return support_or_negate_stat['support'], support_or_negate_stat['baseless'], support_or_negate_stat['negate']


def feature_engineer(args, df):
    
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
    df[["support_num", "baseless_num", "negate_num"]] = df["claim_estimations"].apply(lambda x: pd.Series(get_mv_sum(literal_eval(x))))
    if "gpt_rating" in list(df.columns) and args.gpt_rationale:
        df["gpt_rationale_score"] = df["gpt_rating"].apply(lambda x: x['rating'])

    return df