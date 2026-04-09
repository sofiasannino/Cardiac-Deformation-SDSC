#!/usr/bin/env bash

# bash run_inference_u_net.sh 



source /home/renku/work/scripts/start.sh

# prediction
nnUNetv2_predict -i /home/renku/work/test_dir_interpolated_frames -o /home/renku/work/test_dir_interpolated_frames_out -d Dataset027_ACDC -f 0 1 2 3 4 -tr nnUNetTrainer_100epochs -c 3d_fullres -p nnUNetPlans

# post processing
nnUNetv2_apply_postprocessing -i /home/renku/work/test_dir_interpolated_frames_out -o /home/renku/work/test_dir_interpolated_frames_out_pp -pp_pkl_file /home/renku/work/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/renku/work/nnUNet_results/Dataset027_ACDC/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json