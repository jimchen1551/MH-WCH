#!/bin/bash

# Activate your environment
source activate chc

# Define arrays of options for each parameter
data=("demographic" "3bp" "2bp")
label=("14090" "13080")
resample=("None" "Tomek" "SMOTE" "SMOTETomek")
dim_red=("None" "PCA" "kPCA" "t-SNE")
model=("SVM" "MLP" "TabPFN")
loss=("BCELoss" "BalancedBCELoss" "FocalLoss")

# Use GNU Parallel to iterate over every combination
parallel --results results/logs -j 100% \
    python main.py --data {1} --label {2} --resample {3} --dim_red {4} --model {5} --loss {6} ::: \
    "${data[@]}" ::: "${label[@]}" ::: "${resample[@]}" ::: "${dim_red[@]}" ::: "${model[@]}" ::: "${loss[@]}"
