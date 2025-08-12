<div align= "center">
    <h1>Code For Our Papaer</h1>
</div>


## 📦 Installation

We recommend creating a new [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) to manage dependencies:

1. **Create a new Conda environment** (Python 3.8.18):

   ```bash
   conda create --name idrl python=3.8.18 -y
   ```

2. **Activate the newly created environment:**

   ```bash
   conda activate idrl
   ```
   
3. **Install the required modules from pip:**
   ```bash
   sh install.sh
   ```

---

## 📁 Download the Data

You can manually download the compressed dataset from the following link:

🔗 [Download from Google Drive](https://drive.google.com/file/d/1WWXBsS1Lu_7jpW4-Sl_jZY_sPivBgGpe/view?usp=sharing)

> ⚠️ **Note:** The decompressed file size is approximately **6 GB** due to the large ground-truth user-item interaction matrix.

---

## 🚀 Running Commands

You can now run the following examples if everything is set up correctly!

- The `env` argument in all experiments should be set to one of the following environments:
  - `KuaiEnv-v0`
  - `MovieLensEnv-v0`

#### Run user model
```shell
python run/usermodel/run_DeepFM_ensemble.py --env KuaiEnv-v0  --cuda 3     --epoch 5 --n_models 5 --loss "pointneg" --message "pointneg" 

python run/usermodel/run_DeepFM_ensemble.py --env KuaiEnv-v0  --cuda 1     --epoch 5 --is_ab 1 --tau 100 --n_models 1 --leave_threshold 0 --loss "pointneg" --message "CIRS_UM"  
```

#### Run policies
```shell
python run/policy/run_DiscreteBCQ.py --env KuaiEnv-v0   --cuda 3 --which_tracker avg --reward_handle "cat" --window_size 3 --unlikely-action-threshold 0.2 --explore_eps 0.4 --read_message "pointneg"  --message "BCQ"  --num_leave_compute 4 

python run/policy/run_DiscreteCQL.py --env KuaiEnv-v0   --cuda 2 --which_tracker avg --reward_handle "cat" --window_size 3 --min-q-weight 0.3 --explore_eps 0.4 --read_message "pointneg"  --message "CQL" --num_leave_compute 4 

python run/policy/run_DiscreteCRR.py --env KuaiEnv-v0  --cuda 1 --which_tracker avg --reward_handle "cat" --window_size 3 --explore_eps 0.01 --read_message "pointneg"  --message "CRR"    --num_leave_compute 4 

python run/policy/run_SQN.py        --env KuaiEnv-v0  --cuda 4 --which_tracker avg --reward_handle "cat" --window_size 3 --unlikely-action-threshold 0.6 --explore_eps 0.4 --read_message "pointneg"  --message "SQN"  --num_leave_compute 4 

python run/policy/run_C51.py  --env KuaiEnv-v0  --cuda 1 --which_tracker avg --reward_handle "cat" --window_size 3 --v-min 0. --v-max 1. --explore_eps 0.005 --read_message "pointneg"  --message "C51"  --num_leave_compute 4 

python run/policy/run_DDPG.py --env KuaiEnv-v0 --cuda 3 --which_tracker avg --reward_handle "cat" --window_size 3 --remap 0.001 --explore_eps 1.2 --read_message "pointneg"  --message "DDPG"    --num_leave_compute 4 

python run/policy/run_TD3.py  --env KuaiEnv-v0  --cuda 3 --which_tracker avg --reward_handle "cat" --window_size 3 --remap 0.001 --explore_eps 1.5 --read_message "pointneg"  --message "TD3" --num_leave_compute 4 

python run/policy/run_SAC4IR.py     --env KuaiEnv-v0   --cuda 5  --which_tracker avg --reward_handle "cat" --window_size 3 --target_entropy_ratio 0.8 --explore_eps 0.1 --read_message "pointneg" --lambda_temper 0.01 --num_leave_compute 4  --message "SAC4IR" 

python run/policy/run_DNaIR.py  --env KuaiEnv-v0  --cuda 5    --which_tracker avg --reward_handle "cat" --window_size 3 --read_message "pointneg" --lambda_novelty  0.01 --num_leave_compute 4  --message "DNaIR" 

python run/policy/run_DORL.py        --env KuaiEnv-v0  --cuda 6 --which_tracker avg --reward_handle "cat" --window_size 3  --read_message "pointneg"  --message "DORL" --lambda_entropy 5 --num_leave_compute 4 

python run/policy/run_CIRS.py --env KuaiEnv-v0  --cuda 5 --which_tracker sasrec --reward_handle "cat2" --tau 100 --window_size 3 --gamma_exposure 10  --read_message "CIRS_UM"  --message "CIRS" --num_leave_compute 4 

python run/policy/run_IDRL.py       --env KuaiEnv-v0  --cuda 4 --which_tracker sasrec --reward_handle "cat" --window_size 3  --read_message "pointneg" --lambda_diversity 0.1 --message "IDRL"  --k_near 1000 --lambda_guide 0.02 --subgoal_interval 3 --num_leave_compute 4 
```
