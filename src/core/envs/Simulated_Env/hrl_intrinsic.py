import numpy as np
import torch
from torch import FloatTensor
from src.core.envs.Simulated_Env.base import BaseSimulatedEnv
import math

# from virtualTB.model.UserModel import UserModel
# from src.core.envs.VirtualTaobao.virtualTB.utils import *


class HRLIntrinsicSimulatedEnv(BaseSimulatedEnv):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None,
                 item_similarity=None,
                 item_popularity=None,
                 lambda_diversity=0.1,
                 subgoal_interval=4
                 ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.item_similarity = item_similarity
        self.item_popularity = item_popularity
        self.lambda_diversity = lambda_diversity
        self.subgoal_interval = subgoal_interval

    def reset(self):
        self.c_h_reward = 0
        return super().reset()

    def _cal_diversity(self, action):
        if hasattr(self.env_task, "lbe_item"):
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
        else:
            p_id = action

        l = len(self.history_action)
        div = 0.0
        if l <= 1:
            return 0.0
        nor = 0
        s = 4.
        for i in range(l - 1):
            if hasattr(self.env_task, "lbe_item"):
                q_id = self.env_task.lbe_item.inverse_transform([self.history_action[i]])[0]
            else:
                q_id = self.history_action[i]
            div += (1 - self.item_similarity[q_id, p_id]) * math.exp(-(l-1-i) / s)
            nor += math.exp(-(l-1-i) / s)
        div /= nor #(l - 1)
        return div

    def _cal_novelty(self, action):
        if hasattr(self.env_task, "lbe_item"):
            p_id = self.env_task.lbe_item.inverse_transform([action])[0]
        else:
            p_id = action
        item_pop = self.item_popularity[p_id]
        nov = - np.log(item_pop + 1e-10)  # nov \in xxxx   -log(0.0001) = 4  lambda 考虑~0.25
        return nov

    def _compute_pred_reward(self, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        else:  # elif self.env_name == "KuaiEnv-v0":
            # get prediction
            pred_reward = self.predicted_mat[self.cur_user, action]  # todo

        # get diversity
        div_reward = self._cal_diversity(action)
        # get novelty
        nov_reward = self._cal_novelty(action)
        intrinsic_reward = self.lambda_diversity * div_reward 
        pred_reward = pred_reward - self.MIN_R
        #print(pred_reward, div_reward, nov_reward)
        self.c_h_reward += pred_reward + intrinsic_reward
        return pred_reward

    def step(self, action: FloatTensor):
        # 1. Collect ground-truth transition info
        self.action = action
        # real_state, real_reward, real_done, real_info = self.env_task.step(action)
        real_state, real_reward, _, real_terminated, real_truncated, real_info = self.env_task.step(action)

        t = int(self.total_turn)

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action)

        # 2. Predict click score, i.e, reward
        l_reward = self._compute_pred_reward(action)

        self.cum_reward += l_reward
        self.total_turn = self.env_task.total_turn

        terminated = real_terminated
        # Rethink commented, do not use new user as new state
        # if terminated:
        #     self.state, self.info = self.env_task.reset()

        self.state = self._construct_state(l_reward)

        # info =  {'CTR': self.cum_reward / self.total_turn / 10}
        info = {'cum_reward': self.cum_reward}
        truncated = False
        if (self.total_turn) % self.subgoal_interval == 0 or terminated:
            h_reward = float(self.c_h_reward)#self.c_h_reward / self.subgoal_interval
            self.c_h_reward = 0.
        else:
            h_reward = 0.#self.c_h_reward / self.subgoal_interval
        return self.state, l_reward, h_reward, terminated, truncated, info
