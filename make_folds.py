import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

# prefix = "../data/"
prefix = "data/"
# fname = "buzzfeed_1267_2model.jsonl"
fname = "buzzfeed182_gpt.jsonl"

with open(prefix + fname, "r") as fr:
    lines = fr.readlines()

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1992)

jline_list = []
stratify_col = "label"

for idx, l in enumerate(lines):
    dic = {}
    jline = json.loads(l)
    label = jline["label"]
    if label == 'mostly true':
        label = 1
    else:
        label = 0
    claim_estimations = jline['gpt']['features']['claim_estimations']
    if not claim_estimations:
        continue
    # print(label)
    dic["claim_estimations"] = claim_estimations
    dic['predicate_estimations'] = []
    dic["label"] = label
    jline_list.append(dic)


df = pd.DataFrame(jline_list)
print(len(df))

df = df.drop_duplicates(subset=["claim_estimations"]).reset_index(drop=True)
# print(len(df))
# print(df["label"].value_counts())
# exit(1)
# true_df = df[df["label"]==1]
# false_df = df[df["label"]==0]
# df = pd.concat([true_df.sample(649), false_df.sample(649)]).sample(frac=1.0).reset_index(drop=True)

data_list = []
for idx, (train_index, valid_index) in enumerate(skf.split(df.index, df[stratify_col])):
    count = 0
    # print(train_index)
    print(valid_index)

    df.loc[valid_index, "fold"] = idx

    # with open(prefix + train_fname, "w") as fw:
    #     for tidx in train_index:
    #         count += jline_list[tidx][stratify_col]
    #         fw.write(json.dumps(jline_list[tidx])+"\n")
    # print(count)
    # count = 0
    # with open(prefix + valid_fname, "w") as fw:
    #     for tidx in valid_index:
    #         count += jline_list[tidx][stratify_col]
    #         fw.write(json.dumps(jline_list[tidx])+"\n")
    # print(count)
    # exit()

print(df)
print(df["fold"].value_counts())
print(df["label"].value_counts())
df.to_csv("data/buzzfeed182_gpt.csv", index=False)
