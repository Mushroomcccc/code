from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils import RunningMeanStd


class HA2CPolicy(BasePolicy):
    # 分层ac、optim
    def __init__(
            self,
            h_actor: torch.nn.Module,
            l_actor: torch.nn.Module,
            h_critic: torch.nn.Module,
            l_critic: torch.nn.Module,
            h_optim: torch.optim.Optimizer,
            l_optim: torch.optim.Optimizer,
            h_dist_fn: Type[torch.distributions.Distribution],
            l_dist_fn: Type[torch.distributions.Distribution],
            h_state_tracker=None,
            l_state_tracker=None,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            max_grad_norm: Optional[float] = None,
            discount_factor: float = 0.99,
            gae_lambda: float = 0.95,
            max_batchsize: int = 256,
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            reward_normalization: bool = False,
            deterministic_eval: bool = False,
            subgoal_interval: int = 4,
            device=None,
            lambda_guide: float=0.1,
            k_near: int =1000,
            **kwargs: Any
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.h_actor = h_actor
        self.l_actor = l_actor
        self.h_critic = h_critic
        self.l_critic = l_critic
        self.h_optim = h_optim
        self.l_optim = l_optim
        self.h_dist_fn = h_dist_fn
        self.l_dist_fn = l_dist_fn
        self.h_state_tracker = h_state_tracker
        self.l_state_tracker = l_state_tracker
        self._h_actor_critic = ActorCritic(self.h_actor, self.h_critic)
        self._l_actor_critic = ActorCritic(self.l_actor, self.l_critic)

        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._gamma = discount_factor
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._batch_size = max_batchsize
        self.subgoal_interval = subgoal_interval
        self._deterministic_eval = deterministic_eval
        self.device = device
        self.lambda_guide = lambda_guide
        self.k_near = k_near

        self.n_items = h_state_tracker.num_item
        self.emb_dim = h_state_tracker.emb_dim
        item_index = np.expand_dims(np.arange(self.n_items), -1)  # [n_items, 1]
        item_embs = self.h_state_tracker.get_embedding(item_index, "action")  # [n_items, emb_dim]
        self.item_embs = item_embs


    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    # 分层探索
    def forward(
            self,
            batch: Batch,
            buffer: Optional[ReplayBuffer],
            indices: np.ndarray = None,
            is_obs=None,
            is_train=True,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            use_batch_in_statetracker=False,
            **kwargs: Any,
    ) -> Batch:
        # 上层智能体动作
        h_obs_emb = self.h_state_tracker(buffer=buffer, indices=indices, is_obs=is_obs, batch=batch, is_train=is_train,
                                         use_batch_in_statetracker=use_batch_in_statetracker)
        h_logits, _ = self.h_actor(h_obs_emb)

        if isinstance(h_logits, tuple):
            h_dist = self.h_dist_fn(*h_logits)
        else:
            h_dist = self.h_dist_fn(h_logits)
        h_act = h_dist.sample() # [B, emb_dim]

        # 下层智能体采取动作
        is_new_subgoal = (torch.tensor(batch.env_step) % self.subgoal_interval) == 0
        is_new_subgoal = is_new_subgoal.reshape(-1, 1).repeat(1, h_act.shape[1]).to(self.device)
        if isinstance(batch.h_act, np.ndarray) and batch.h_act.shape[0] > 0:
            h_act = torch.tensor(batch.h_act).to(self.device) * (~is_new_subgoal) + h_act * is_new_subgoal
        l_obs_emb = self.l_state_tracker(buffer=buffer, indices=indices, is_obs=is_obs, batch=batch, is_train=is_train,
                                         use_batch_in_statetracker=use_batch_in_statetracker)
        l_logits, _ = self.l_actor(torch.cat([l_obs_emb, h_act], dim=-1))
        

        # -------趋向性约束-------
        l_obs_emb = l_obs_emb[:, :-1]
        b, emb_dim = l_obs_emb.shape
        item_num = self.item_embs.shape[0]
        # 计算h_act与item_embs之间的欧氏距离
        l_obs_emb = l_obs_emb.unsqueeze(1).expand(b, item_num, emb_dim)  # 扩展l_obs_emb为[b, item_num, emb_dim]
        distances = torch.norm(l_obs_emb - self.item_embs, p=2, dim=2)  # 计算欧氏距离，结果形状为[b, item_num]
        # 获取每个样本中最小的k_near个距离对应的索引
        _, topk_indices = torch.topk(distances, k=self.k_near, dim=1, largest=False, sorted=False)  # 形状为[b, k_near]
        # 生成一个掩码矩阵，初始为全零
        mask = torch.zeros_like(distances)
        #print(mask.shape)
        # 将最小的k_near个距离的位置标记为1
        for i in range(b):
            mask[i, topk_indices[i]] = 1  # 对于第i个样本，将前k_near个最近的item的位置标记为1

        if self.action_type == "discrete":
            if is_obs:
                l_logits = l_logits * batch.mask #* mask  # logits is in [0, 1]  the final layer of actor is softmax_layer
            else:
                l_logits = l_logits * batch.next_mask #* mask
        
        if isinstance(l_logits, tuple):
            l_dist = self.l_dist_fn(*l_logits)
        else:
            l_dist = self.l_dist_fn(l_logits)
        l_act = None
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                l_act = l_logits.argmax(-1)
            elif self.action_type == "continuous":
                l_act = l_logits[0]
        else:
            l_act = l_dist.sample()


        return Batch(h_act=h_act, l_act=l_act, h_dist=h_dist, l_dist=l_dist)

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indices)
        batch.h_act = to_torch_as(batch.h_act, batch.h_v_s)
        batch.l_act = to_torch_as(batch.act, batch.l_v_s)
        return batch

    def _compute_returns(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:

        # 计算上层智能体信息
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch_size, shuffle=False, merge_last=True):
                obs_emb = self.h_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=True)
                v_s.append(self.h_critic(obs_emb))
                next_indices = minibatch.indices
                for _ in range(self.subgoal_interval):
                    next_indices = self._buffer.next(next_indices)
                next_indices_r = minibatch.indices
                for _ in range(self.subgoal_interval - 1):
                    next_indices_r = self._buffer.next(next_indices_r)
                done = torch.tensor(buffer.terminated[next_indices]).to(self.device).squeeze()
                h_obs_next_emb = self.h_state_tracker(buffer=buffer, indices=next_indices, is_obs=True)
                # print(self.h_critic(h_obs_next_emb).shape, torch.tensor(buffer[next_indices_r].h_rew).to(self.device).squeeze().shape, done.shape)
                target_v = self.h_critic(h_obs_next_emb).squeeze() * (~done) * self._gamma + torch.tensor(
                    buffer[next_indices_r].h_rew).to(self.device).squeeze()
                v_s_.append(target_v)

        next_indices = batch.indices
        for _ in range(self.subgoal_interval):
            next_indices = self._buffer.next(next_indices)
        batch.h_v_s = torch.cat(v_s, dim=0).flatten()  # old value
        batch.h_v_s_ = torch.cat(v_s_, dim=0).flatten()  # target value
        batch.h_adv = batch.h_v_s_ - batch.h_v_s.detach()
        is_update1 = torch.tensor((batch.env_step % self.subgoal_interval) == 0).to(self.device)
        is_update2 = torch.tensor(buffer[next_indices].env_step % self.subgoal_interval == 0).to(self.device)
        is_update3 = torch.tensor(buffer.terminated[next_indices]).to(self.device).squeeze()
        is_update4 = torch.tensor(indices != next_indices).to(self.device).squeeze()
        is_update = is_update1 & (is_update2 | is_update3) & is_update4
        batch.is_h_update = is_update
        next_indices_r = batch.indices
        for _ in range(self.subgoal_interval - 1):
            next_indices_r = self._buffer.next(next_indices_r)

        # 计算下层智能体信息
        # 计算相似度,添加内部奖励
        if self.action_type == 'continuous':
            h_act = torch.tensor(batch.h_act).to(self.device)
            l_act = torch.tensor(batch.act).to(self.device)
        else:
            # 上层traget指引
            indices = batch.indices
            n_indices = buffer.next(indices)
            obs_next_emb = self.l_state_tracker(buffer=buffer, indices=indices, is_obs=False)
            obs_emb = self.l_state_tracker(buffer=buffer, indices=indices, is_obs=True)
            h_act = torch.tensor(batch.h_act).to(self.device)
            sim = -(torch.norm(h_act - obs_next_emb, p=2, dim=1) - torch.norm(h_act - obs_emb, p=2, dim=1)).squeeze().detach().cpu().numpy()
            sim = sim * (n_indices != indices)
        batch.rew = (batch.rew + sim * self.lambda_guide) 
        # 计算target_v,adv
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch_size, shuffle=False, merge_last=True):
                obs_emb = self.l_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=True)
                h_act = torch.tensor(minibatch.h_act).to(self.device)
                v_s.append(self.l_critic(torch.cat([obs_emb, h_act], dim=-1)))
                next_indices = self._buffer.next(minibatch.indices)
                next_h_act = torch.tensor(buffer[next_indices].h_act).to(self.device)
                obs_next_emb = self.l_state_tracker(buffer=buffer, indices=minibatch.indices, is_obs=False)
                v_s_.append(self.l_critic(torch.cat([obs_next_emb, next_h_act], dim=-1)))
        batch.l_v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.l_v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                            np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.l_v_s)
        batch.l_adv = to_torch_as(advantages, batch.l_v_s)
        return batch

    # 分层学习
    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        h_losses, losses, actor_losses, vf_losses, ent_losses = [], [], [], [], []
        h_optim_RL, h_optim_state = self.h_optim
        l_optim_RL, l_optim_state = self.l_optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # 更新上层智能体
                # calculate loss for actor
                h_dist = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True).h_dist
                h_log_prob = h_dist.log_prob(minibatch.h_act)
                h_log_prob = h_log_prob.reshape(len(minibatch.h_adv), -1).transpose(0, 1)
                idx = ~torch.isinf(h_log_prob)
                h_actor_loss = -((h_log_prob * minibatch.h_adv) * minibatch.is_h_update)[idx].mean()
                # calculate loss for critic
                h_obs_emb = self.h_state_tracker(self._buffer, minibatch.indices, is_obs=True)
                h_value = self.h_critic(h_obs_emb).flatten()
                h_vf_loss = ((minibatch.h_v_s_.flatten() - h_value).pow(2) * minibatch.is_h_update).mean()

                # calculate regularization and overall loss
                h_ent_loss = (h_dist.entropy() * minibatch.is_h_update).mean()
                h_loss = h_actor_loss + self._weight_vf * h_vf_loss - self._weight_ent * h_ent_loss
                h_optim_RL.zero_grad()
                h_optim_state.zero_grad()
                h_loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._h_actor_critic.parameters(), max_norm=self._grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.h_state_tracker.parameters(), max_norm=self._grad_norm
                    )
                h_optim_RL.step()
                h_optim_state.step()

                # 更新下层智能体
                l_dist = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True).l_dist
                l_log_prob = l_dist.log_prob(minibatch.l_act)
                l_log_prob = l_log_prob.reshape(len(minibatch.l_adv), -1).transpose(0, 1)
                idx = ~torch.isinf(l_log_prob)

                l_actor_loss = -(l_log_prob * minibatch.l_adv)[idx].mean()
                # calculate loss for critic
                h_act = torch.tensor(minibatch.h_act.clone().detach()).to(self.device)
                l_obs_emb = self.l_state_tracker(self._buffer, minibatch.indices, is_obs=True)
                l_value = self.l_critic(torch.cat([l_obs_emb, h_act], dim=-1)).flatten()
                l_vf_loss = F.mse_loss(minibatch.returns.flatten(), l_value)
                # calculate regularization and overall loss
                l_ent_loss = l_dist.entropy().mean()
                l_loss = l_actor_loss + self._weight_vf * l_vf_loss - self._weight_ent * l_ent_loss
                
                l_optim_RL.zero_grad()
                l_optim_state.zero_grad()
                l_loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._l_actor_critic.parameters(), max_norm=self._grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.l_state_tracker.parameters(), max_norm=self._grad_norm
                    )
                l_optim_RL.step()
                l_optim_state.step()

                actor_losses.append(l_actor_loss.item())
                vf_losses.append(l_vf_loss.item())
                ent_losses.append(l_ent_loss.item())
                losses.append(l_loss.item())
                h_losses.append(h_loss.item())

        return {
            "h_loss": h_losses,
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
