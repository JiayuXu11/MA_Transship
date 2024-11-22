"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = [None] * self.num_agents
        critic_train_infos = [None] * len(self.critic_buffer) 
        # critic_train_before_tf = False        # 标识critic被train过一次没有，避免重复train

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        advantages_collector = []
        for critic_buffer_id in range(len(self.critic_buffer)):
            if self.value_normalizer is not None:
                # shape: (episode_length, n_rollout_threads, dim)
                advantage = self.critic_buffer[critic_buffer_id].returns[
                    :-1
                ] - self.value_normalizer[critic_buffer_id].denormalize(
                    self.critic_buffer[critic_buffer_id].value_preds[:-1]
                )
            else:
                # shape: (episode_length, n_rollout_threads, dim)
                advantage = (
                    self.critic_buffer[0].returns[:-1] - self.critic_buffer[0].value_preds[:-1]
                )
            advantages_collector.append(advantage)
        # shape: (num_critic_buffer, episode_length, n_rollout_threads, dim)
        advantages = np.array(advantages_collector)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        # 如果只有一个actor，那就共用所有actor_buffer的数据，混一起训练
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, "EP"
            )
            actor_train_infos = []
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        # 如果有多个actor，那就分别训练，各个actor只用自己的actor_buffer数据
        else:
            for agent_id in agent_order:
                self.actor_buffer[agent_id].update_factor(
                    factor
                )  # current actor save factor

                # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                # compute action log probs for the actor before update.
                if self.use_factor:
                    old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                        self.actor_buffer[agent_id]
                        .obs[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                        self.actor_buffer[agent_id]
                        .rnn_states[0:1]
                        .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                        self.actor_buffer[agent_id].actions.reshape(
                            -1, *self.actor_buffer[agent_id].actions.shape[2:]
                        ),
                        self.actor_buffer[agent_id]
                        .masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                        available_actions,
                        self.actor_buffer[agent_id]
                        .active_masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                    )

                # update actor
                actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], 
                        advantages[min(agent_id, len(self.critic_buffer) - 1)].copy(), 
                        "EP"
                    )
                # compute action log probs for updated agent\
                if self.use_factor:
                    new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                        self.actor_buffer[agent_id]
                        .obs[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                        self.actor_buffer[agent_id]
                        .rnn_states[0:1]
                        .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                        self.actor_buffer[agent_id].actions.reshape(
                            -1, *self.actor_buffer[agent_id].actions.shape[2:]
                        ),
                        self.actor_buffer[agent_id]
                        .masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                        available_actions,
                        self.actor_buffer[agent_id]
                        .active_masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                    )

                    # update factor for next agent
                    factor = factor * _t2n(
                        getattr(torch, self.action_aggregation)(
                            torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                        ).reshape(
                            self.algo_args["train"]["episode_length"],
                            self.algo_args["train"]["n_rollout_threads"],
                            1,
                        )
                    )
                actor_train_infos[agent_id] = actor_train_info

        # update critic
        # 单个critic_buffer，单个critic
        if (not self.multi_critic_buffer_tf) and (not self.multi_critic_tf):
            critic_train_info = self.critic[0].train(self.critic_buffer[0], self.value_normalizer[0])
            critic_train_infos[0] = critic_train_info
        # 多个critic_buffer，多个critic
        elif self.multi_critic_buffer_tf and self.multi_critic_tf:
            for agent_id in agent_order:
                critic_train_info = self.critic[agent_id].train(self.critic_buffer[agent_id], self.value_normalizer[agent_id])
                critic_train_infos[agent_id] = critic_train_info
        # 多个critic_buffer，单个critic
        elif self.multi_critic_buffer_tf and (not self.multi_critic_tf):
            for agent_id in agent_order:
                critic_train_info = self.critic[0].share_param_train(self.critic_buffer, self.value_normalizer[0])
                for _ in torch.randperm(self.num_agents):
                    critic_train_infos[agent_id] = critic_train_info
        else:
            raise NotImplementedError

        return actor_train_infos, critic_train_infos
