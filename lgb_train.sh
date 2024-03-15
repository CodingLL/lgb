# political
python lgb_training.py \
--filepath data/politifact_547_gpt_main_claim_only_0315.csv \
--outpath exp/politifact_547_gpt_main_claim_only_0315

# buzzfeed ----------------------

python lgb_training.py \
--filepath data/buzzfeed_1267_nbmodel.csv \
--outpath exp/buzzfeed_1267_nbmodel


python lgb_training.py \
--filepath data/buzzfeed_1267_gpt.csv \
--outpath exp/buzzfeed_1267_gpt


python lgb_training.py \
--filepath data/overall_training_0229_fold.csv \
--outpath exp/buzzfeed_1267_gpt_full


python lgb_training.py \
--filepath data/buzzfeed182_gpt_v2.csv \
--outpath exp/buzzfeed182_gpt_5fold


python lgb_training.py \
--filepath data/buzzfeed182_gpt_v3.csv \
--outpath exp/buzzfeed182_gpt_5fold_v3

python lgb_training.py \
--filepath data/buzzfeed182_gpt_v2_rationale.jsonl \
--outpath exp/buzzfeed182_gpt_v2_rationale_5fold

python lgb_training.py \
--filepath data/buzzfeed182_gpt_rerun_0314_rationale.jsonl \
--outpath exp/buzzfeed182_gpt_rerun_0314_rationale_5fold