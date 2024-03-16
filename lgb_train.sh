# political
# 0316 nbmodel
python lgb_training_3ensemble.py \
--filepath data/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315.jsonl \
--outpath exp/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315_5fold \
--gpt_rationale 1

python lgb_training.py \
--filepath data/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315.jsonl \
--outpath exp/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315_5fold \
--gpt_rationale 1
# -------------------
# python lgb_training.py \
# --filepath data/politifact_547_gpt_main_claim_only_rationale_0315.jsonl \
# --outpath exp/politifact_547_gpt_main_claim_only_rationale_0315_5fold \
# --gpt_rationale 1

# grid search
echo 'politifact_547_gpt_main_claim_only_0315'
python lgb_training_gs.py \
--filepath data/politifact_547_gpt_main_claim_only_rationale_0315.jsonl \
--outpath exp/politifact_547_gpt_main_claim_only_0315_gs \
--gpt_rationale 1 \
> logs/politifact_547_gpt_main_claim_only_0315_gs.log 2>&1

echo 'buzzfeed182_gpt_v2_rationale_0315'
python lgb_training_gs.py \
--filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
--outpath exp/buzzfeed182_gpt_v2_rationale_0315_gs \
--gpt_rationale 0 \
> logs/buzzfeed182_gpt_v2_rationale_0315.log 2>&1

python lgb_training_gs.py \
--filepath data/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315.jsonl \
--outpath exp/buzzfeed182_nbmodel_main_claim_only_rationale_v2_0315_5fold_gs \
--gpt_rationale 1

python lgb_training_gs.py \
--filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
--outpath exp/buzzfeed182_gpt_v2_rationale_0315_gs \
--gpt_rationale 0

# ---------------------
# '''
# 0315 3model ensemble
# 5 fold best pos_f1 0.8186 neg_f1 0.7789
# 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
python lgb_training_3ensemble.py \
--filepath data/politifact_547_gpt_main_claim_only_rationale_0315.jsonl \
--outpath exp/politifact_547_gpt_main_claim_only_rationale_0315 \
--gpt_rationale 0

# 5 fold best pos_f1 0.7914 neg_f1 0.7797
# 0.54 * pred1 + 0.28 * pred2 + 0.18 * pred3
python lgb_training_3ensemble.py \
--filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
--outpath exp/buzzfeed182_gpt_v2_0315_5fold \
--gpt_rationale 0
# '''

# '''
# python lgb_training.py \
# --filepath data/politifact_547_gpt_main_claim_only_0315.csv \
# --outpath exp/politifact_547_gpt_main_claim_only_0315_gs

# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
# --outpath exp/buzzfeed182_gpt_v2_rationale_0315


# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
# --outpath exp/buzzfeed182_gpt_v2_rationale_0315

# # buzzfeed ----------------------

# python lgb_training.py \
# --filepath data/buzzfeed_1267_nbmodel.csv \
# --outpath exp/buzzfeed_1267_nbmodel


# python lgb_training.py \
# --filepath data/buzzfeed_1267_gpt.csv \
# --outpath exp/buzzfeed_1267_gpt


# python lgb_training.py \
# --filepath data/overall_training_0229_fold.csv \
# --outpath exp/buzzfeed_1267_gpt_full


# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_v2.csv \
# --outpath exp/buzzfeed182_gpt_5fold


# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_v3.csv \
# --outpath exp/buzzfeed182_gpt_5fold_v3

# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_v2_rationale.jsonl \
# --outpath exp/buzzfeed182_gpt_v2_rationale_5fold

# python lgb_training.py \
# --filepath data/buzzfeed182_gpt_rerun_0314_rationale.jsonl \
# --outpath exp/buzzfeed182_gpt_rerun_0314_rationale_5fold
# '''