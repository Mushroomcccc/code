# Run User Model
python run/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --cuda 6  --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg"
python run/usermodel/run_DeepFM_ensemble.py --env MovieLensEnv-v0  --cuda 1     --epoch 5 --is_ab 1 --tau 100 --n_models 1 --leave_threshold 0 --loss "pointneg" --message "CIRS_UM" 

# Run Policy
python run/policy/run_DiscreteCRR.py --env MovieLensEnv-v0   --cuda 4 --which_tracker avg --reward_handle "cat" --window_size 3 --explore_eps 0.01 --read_message "pointneg"  --message "CRR"   --num_leave_compute 4 
python run/policy/run_DiscreteCQL.py --env MovieLensEnv-v0  --cuda 3 --explore_eps 0.5 --which_tracker avg --reward_handle "cat"  --num-quantiles 20 --min-q-weight 10 --window_size 3 --read_message "pointneg"  --message "CQL" --num_leave_compute 4  
python run/policy/run_DiscreteBCQ.py --env MovieLensEnv-v0 --cuda 3 --explore_eps 0.5 --which_tracker avg --reward_handle "cat"  --unlikely-action-threshold 0.6 --window_size 3 --read_message "pointneg"  --message "BCQ" --num_leave_compute 4 
python run/policy/run_SQN.py        --env MovieLensEnv-v0   --cuda 2 --which_tracker avg --reward_handle "cat"  --window_size 3 --read_message "pointneg"  --message "SQN"  --num_leave_compute 4 

python run/policy/run_C51.py     --env MovieLensEnv-v0   --cuda 1 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "C51"  --num_leave_compute 4 
python run/policy/run_DDPG.py    --env MovieLensEnv-v0   --cuda 2  --explore_eps 1.0 --remap_eps 0.001 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "DDPG" --num_leave_compute 6 
python run/policy/run_TD3.py     --env MovieLensEnv-v0  --cuda 2 --explore_eps 0.9 --remap_eps 0.9 --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg"  --message "TD3" --num_leave_compute 4 

python run/policy/run_SAC4IR.py     --env MovieLensEnv-v0  --cuda 5  --which_tracker avg --reward_handle "cat" --window_size 3 --remap 0.001  --target_entropy_ratio 0.9 --explore_eps 1.2 --read_message "pointneg" --lambda_temper 0.01 --num_leave_compute 4  --message "SAC4IR-8" 
python run/policy/run_CIRS.py --env MovieLensEnv-v0  --cuda 7 --tau 100 --window_size 3  --gamma_exposure 10  --read_message "CIRS_UM"  --message "CIRS" --num_leave_compute 4 
python run/policy/run_DNaIR.py  --env MovieLensEnv-v0  --cuda 4    --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --message "DNaIR" --lambda_novelty  0.01 --num_leave_compute 4 
python run/policy/run_DORL.py  --env MovieLensEnv-v0 --cuda 4 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg" --lambda_entropy 1 --message "DORL" --num_leave_compute 4 


python run/policy/run_IDRL.py     --env MovieLensEnv-v0  --cuda 7 --which_tracker sasrec --reward_handle "cat" --window_size 5  --read_message "pointneg" --lambda_diversity 2.0  --message "IDRL" --k_near 1500 --lambda_guide 0.05  --subgoal_interval 4 --num_leave_compute 4 



