"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner_mechanism import MixPolicyBaseRunner


class OnMixPolicyHaRunner(MixPolicyBaseRunner):
    """Runner for on-policy HA algorithms with hierarchical agents"""

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

    def off_train(self):
        """Train the model"""
        self.total_it += 1
        data = self.off_buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.off_critic.turn_on_grad()
        if self.args["algo_mechanism"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.off_num_agents):
                next_action, next_logp_action = self.off_actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            self.off_critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.off_value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.off_num_agents):
                next_actions.append(
                    self.off_actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.off_critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.off_critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo_mechanism"] == "hasac":
                actions = []
                logp_actions = []
                with torch.no_grad():
                    for agent_id in range(self.off_num_agents):
                        action, logp_action = self.off_actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.off_num_agents))
                else:
                    agent_order = list(np.random.permutation(self.off_num_agents))
                for agent_id in agent_order:
                    self.off_actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id], logp_actions[agent_id] = self.off_actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )

                    logp_action = logp_actions[agent_id]
                    actions_t = torch.cat(actions, dim=-1)

                    value_pred = self.off_critic.get_values(sp_share_obs, actions_t)
                    if self.algo_mechanism_args["algo"]["use_policy_active_masks"]:
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha[agent_id] * logp_action)
                                * sp_valid_transition[agent_id]
                            )
                            / sp_valid_transition[agent_id].sum()
                        )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    self.off_actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.off_actor[agent_id].actor_optimizer.step()
                    self.off_actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_mechanism_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.off_actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # train critic's alpha
                if self.algo_mechanism_args["algo"]["auto_alpha"]:
                    self.off_critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo_mechanism"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.off_num_agents):
                            actions.append(
                                self.off_actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.off_critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.off_num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.off_num_agents))
                    for agent_id in agent_order:
                        self.off_actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.off_actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.off_actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.off_actor[agent_id].actor_optimizer.step()
                        self.off_actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.off_num_agents):
                            actions.append(
                                self.off_actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.off_num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.off_num_agents))
                    for agent_id in agent_order:
                        self.off_actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.off_actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.off_critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.off_actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.off_actor[agent_id].actor_optimizer.step()
                        self.off_actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.off_actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.off_num_agents):
                    self.off_actor[agent_id].soft_update()
            self.off_critic.soft_update()
