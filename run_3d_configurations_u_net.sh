#!/usr/bin/env bash

# bash run_3d_configurations_u_net.sh 



source /home/renku/work/scripts/start.sh

DATASET_ID=27
TRAINER="nnUNetTrainer_100epochs"

nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

echo "==== TRAINING WITH 3D FULL RES (${TRAINER}) ===="
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train "${DATASET_ID}" 3d_fullres "${FOLD}" -tr "${TRAINER}" --npz -device cuda
done

echo "==== Finding best configuration ===="
nnUNetv2_find_best_configuration "${DATASET_ID}" -c 3d_fullres --disable_ensembling
