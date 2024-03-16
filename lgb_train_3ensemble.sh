# 0315 3model ensemble
# 5 fold best pos_f1 0.8186 neg_f1 0.7789
# 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
python lgb_training_3ensemble.py \
--filepath data/politifact_547_gpt_main_claim_only_rationale_0315.jsonl \
--outpath exp/politifact_547_gpt_main_claim_only_rationale_0315 \
--gpt_rationale 1

# 5 fold best pos_f1 0.7914 neg_f1 0.7797
# 0.54 * pred1 + 0.28 * pred2 + 0.18 * pred3
python lgb_training_3ensemble.py \
--filepath data/buzzfeed182_gpt_v2_rationale_0315.jsonl \
--outpath exp/buzzfeed182_gpt_v2_0315_5fold \
--gpt_rationale 0