"""Base runner for mix-policy algorithms.
Each agent has its own on-policy agent, while they share one off-policy agent.
"""
import os
import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
# from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    get_num_agents,
)
from harl.utils.basic_tools import set_seed
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY

# off 专用
from torch.distributions import Categorical
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
class MixPolicyBaseRunner:
    """Base runner for mix-policy algorithms."""

    def __init__(self, args, algo_args, env_args, algo_mechanism_args):
        """Initialize the MixPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.multi_critic_tf = algo_args["algo"]["multi_critic_tf"]
        self.multi_critic_buffer_tf = algo_args["algo"]["multi_critic_buffer_tf"]
        self.use_factor = algo_args["algo"]["use_factor"]

        assert (
            self.multi_critic_tf and self.multi_critic_buffer_tf
        ) or not self.multi_critic_tf, "multi_critic_tf为True时，multi_critic_buffer_tf必须为True"
        self.sample_mean_advantage_tf = algo_args["train"]["sample_mean_advantage_tf"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir, algo_mechanism_args)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
                async_tf=algo_args["train"]["async_tf"],
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                    async_tf=algo_args["train"]["async_tf"],
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        # actor
        # share_param为True时，所有agent共享一个actor，但每个agent有自己的actor_buffer
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            # multi_critic_tf为True时，每个agent有自己的critic，否则所有agent共享一个critic.当为False时，self.critic = [critic0,] * 3
            # multi_critic_buffer_tf为True时，每个agent有自己的critic buffer，否则所有agent共享一个critic buffer（即所有agent的share_obs是一样的）。
            # 当为False时，self.critic_buffer = [critic_buffer0] 而非[critic_buffer0, critic_buffer1, critic_buffer2]
            # one_critic, multi_critic_buffer => 所有agent share一个critic，但critic是根据各自agent的share_obs计算各自的value
            # multi_critic, one_critic_buffer 不存在，在初始化时就报错
            self.critic = []
            if not self.multi_critic_tf:
                # critic定义
                share_observation_space = self.envs.share_observation_space[0]
                critic_agent = VCritic(
                    {**algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    device=self.device,
                )
                self.critic.append(critic_agent)
                for agent_id in range(1, self.num_agents):
                    assert (
                        self.envs.share_observation_space[agent_id]
                        == self.envs.share_observation_space[0]
                    ), "Agents have heterogeneous share observation spaces, parameter sharing for critic is not valid."
                    self.critic.append(self.critic[0])
            else:
                for agent_id in range(self.num_agents):
                    critic_agent = VCritic(
                        {**algo_args["model"], **algo_args["algo"]},
                        self.envs.share_observation_space[agent_id],
                        device=self.device,
                    )
                    self.critic.append(critic_agent)
            # critic buffer定义
            self.critic_buffer = []
            if not self.multi_critic_buffer_tf:
                share_observation_space = self.envs.share_observation_space[0]
                cbu = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                )
                for agent_id in range(1, self.num_agents):
                    assert (
                        self.envs.share_observation_space[agent_id]
                        == self.envs.share_observation_space[0]
                    ), "Agents have heterogeneous share observation spaces, sharing critic buffer is not valid."
                self.critic_buffer.append(cbu)
            else:
                for agent_id in range(self.num_agents):
                    cbu = OnPolicyCriticBufferEP(
                        {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                        self.envs.share_observation_space[agent_id],
                    )
                    self.critic_buffer.append(cbu)

            if self.algo_args["train"]["use_valuenorm"] is True:
                if not self.multi_critic_buffer_tf:
                    self.value_normalizer = [ValueNorm(1, device=self.device)]
                else:
                    self.value_normalizer = [
                        ValueNorm(1, device=self.device) for _ in range(self.num_agents)
                    ]
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

        # For Off-policy, 同名variable加个off前缀以作区分
        self.off_num_agents = 1
        self.algo_mechanism_args = algo_mechanism_args

        if "policy_freq" in self.algo_mechanism_args["algo"]:
            self.policy_freq = self.algo_mechanism_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.off_share_param = algo_mechanism_args["algo"]["share_param"]
        self.off_fixed_order = algo_mechanism_args["algo"]["fixed_order"]

        self.agent_deaths = np.zeros(
            (self.algo_mechanism_args["train"]["n_rollout_threads"], self.off_num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.off_num_agents):
            self.action_spaces[agent_id].seed(algo_mechanism_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.mechanism_share_observation_space)
        print("observation_space: ", self.envs.mechanism_observation_space)
        print("action_space: ", self.envs.mechanism_action_space)

        self.off_actor = []
        agent = ALGO_REGISTRY[args["algo_mechanism"]](
            {**algo_mechanism_args["model"], **algo_mechanism_args["algo"]},
            self.envs.mechanism_observation_space[0],
            self.envs.mechanism_action_space[0],
            device=self.device,
        )
        self.off_actor.append(agent)
            
        if not self.algo_mechanism_args["render"]["use_render"]:
            self.off_critic = CRITIC_REGISTRY[args["algo_mechanism"]](
                {**algo_mechanism_args["train"], **algo_mechanism_args["model"], **algo_mechanism_args["algo"]},
                self.envs.mechanism_share_observation_space[0],
                self.envs.mechanism_action_space,
                self.off_num_agents,
                self.state_type,
                device=self.device,
            )

            self.off_buffer = OffPolicyBufferEP(
                    {**algo_mechanism_args["train"], **algo_mechanism_args["model"], **algo_mechanism_args["algo"]},
                    self.envs.mechanism_share_observation_space[0],
                    self.off_num_agents,
                    self.envs.mechanism_observation_space,
                    self.envs.mechanism_action_space,
                )

        if (
            "use_valuenorm" in self.algo_mechanism_args["train"].keys()
            and self.algo_mechanism_args["train"]["use_valuenorm"]
        ):
            self.off_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.off_value_normalizer = None

        if self.algo_mechanism_args["train"]["model_dir"] is not None:
            self.mechanism_restore()

        self.total_it = 0  # total iteration

        if (
            "auto_alpha" in self.algo_mechanism_args["algo"].keys()
            and self.algo_mechanism_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.off_num_agents):
                if (
                    self.envs.mechanism_action_space[agent_id].__class__.__name__ == "Box"
                ):  # Differential entropy can be negative
                    self.target_entropy.append(
                        -np.prod(self.envs.mechanism_action_space[agent_id].shape)
                    )
                else:  # Discrete entropy is always positive. Thus we set the max possible entropy as the target entropy
                    self.target_entropy.append(
                        -0.98
                        * np.log(1.0 / np.prod(self.envs.mechanism_action_space[agent_id].shape))
                    )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.off_num_agents):
                _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha], lr=self.algo_mechanism_args["algo"]["alpha_lr"]
                    )
                )
                self.alpha.append(torch.exp(_log_alpha.detach()))
        elif "alpha" in self.algo_mechanism_args["algo"].keys():
            self.alpha = [self.algo_mechanism_args["algo"]["alpha"]] * self.off_num_agents

    def run(self):
        """Run the training (or rendering) pipeline."""
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        self.train_episode_rewards = np.zeros(
            self.algo_mechanism_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        # train and eval
        already_steps = 0
        steps = (
            self.algo_mechanism_args["train"]["num_env_steps"]
            // self.algo_mechanism_args["train"]["n_rollout_threads"]
        )
        update_num = int(  # update number per train
            self.algo_mechanism_args["train"]["update_per_train"]
            * self.algo_mechanism_args["train"]["train_interval"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training
        eval_avg_reward_best = -1e9
        eval_times_since_last_best = 0
        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                if not self.multi_critic_tf:
                    self.critic[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.critic[agent_id].lr_decay(episode, episodes)
            elif self.algo_args["train"][
                "use_fragment_lr_decay"
            ]:  # fragment decay of learning rate
                if self.share_param:
                    self.actor[0].fragment_lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].fragment_lr_decay(episode, episodes)
                if not self.multi_critic_tf:    
                    self.critic[0].fragment_lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.critic[agent_id].fragment_lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                already_steps += 1
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                # 执行each agent的action，得到mechanism agent的obs, share_obs
                off_new_obs, off_new_share_obs = self.envs.step_prepare(actions)
                off_next_obs = off_new_obs.copy()
                off_next_share_obs = off_new_share_obs.copy()
                off_next_available_actions = None
                # 获取mechanism agent的action
                off_actions = self.get_off_actions(off_new_obs, add_random=False)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(off_actions)
                # 处理并存储on-policy数据
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(data)  # logger callback at each step
                self.insert(data)  # insert data into buffer

                # 把数据转换为off-policy数据，shape从(n_threads, n_agents, dim) -> (n_threads, 1, dim)
                # 为防止没有off的前置数据
                if step > 0:
                    # 生成off_rewards, 为infos中的mechanism_reward
                    off_rewards = np.array([[[info[0]['mechanism_reward']]] for info in infos])
                    off_dones = dones[:, 0:1]
                    off_infos = [[info[0]] for info in infos]
                    off_data = (
                        off_share_obs,
                        off_obs.transpose(1, 0, 2),
                        off_actions.transpose(1, 0, 2),
                        off_available_actions.transpose(1, 0, 2)
                        if len(np.array(off_available_actions).shape) == 3
                        else None,
                        off_rewards,
                        off_dones,
                        off_infos,
                        off_next_share_obs,
                        off_next_obs,
                        off_next_available_actions.transpose(1, 0, 2)
                        if len(np.array(off_next_available_actions).shape) == 3
                        else None,
                    )
                    self.off_insert(off_data)
                # 存储previous obs, share_obs, available_actions
                off_obs = off_new_obs
                off_share_obs = off_new_share_obs
                off_available_actions = off_next_available_actions
                if (already_steps % self.algo_mechanism_args["train"]["train_interval"] == 0 and 
                    already_steps > self.algo_mechanism_args["train"]["warmup_steps"]):
                    if self.algo_mechanism_args["train"]["use_linear_lr_decay"]:
                        self.off_actor[0].lr_decay(already_steps * self.algo_args["train"]["n_rollout_threads"], steps)
                        self.off_critic.lr_decay(already_steps * self.algo_args["train"]["n_rollout_threads"], steps)
                    elif self.algo_mechanism_args["train"]["use_fragment_lr_decay"]:
                        self.off_actor[0].fragment_lr_decay(already_steps * self.algo_args["train"]["n_rollout_threads"], steps)
                        self.off_critic.fragment_lr_decay(already_steps * self.algo_args["train"]["n_rollout_threads"], steps)
                    for _ in range(update_num):
                        self.off_train()

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_infos = self.train()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_infos,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    eval_avg_reward = self.eval('eval')
                    if (eval_avg_reward > eval_avg_reward_best):
                        print(f"the new best eval_avg_reward is: {eval_avg_reward}, history best is: {eval_avg_reward_best}") 
                        eval_avg_reward_best = eval_avg_reward
                        self.save()
                        print("model saved")
                        eval_times_since_last_best = 0
                    else:
                        eval_times_since_last_best += 1
                        if eval_times_since_last_best >= self.algo_args["train"]["early_stop"]:
                            print(f"early stop at episode {episode}")
                            self.restore(last_best_tf = True)
                            break
                else:
                    self.save()

            self.after_update()
        # 训练时间结束，在测试集上看下表现
        self.eval('test')

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        if self.multi_critic_buffer_tf:
            for agent_id in range(self.num_agents):  
                self.critic_buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
        else:
            self.critic_buffer[0].share_obs[0] = share_obs[:, 0].copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if not self.multi_critic_buffer_tf:
            value, rnn_state_critic = self.critic[0].get_values(
                self.critic_buffer[0].share_obs[step],
                self.critic_buffer[0].rnn_states_critic[step],
                self.critic_buffer[0].masks[step],
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        # collect values, rnn_states_critic from n critics
        else:
            value_collector = []
            rnn_state_critic_collector = []
            for agent_id in range(self.num_agents):
                value, rnn_state_critic = self.critic[agent_id].get_values(
                    self.critic_buffer[agent_id].share_obs[step],
                    self.critic_buffer[agent_id].rnn_states_critic[step],
                    self.critic_buffer[agent_id].masks[step],
                )
                value_collector.append(_t2n(value))
                rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
            values = np.array(value_collector).transpose(1, 0, 2)
            rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # single_critic_buffer: (n_threads, dim), others: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # single_critic_buffer: (n_threads, dim), others: (n_threads, n_agents, dim)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        # TODO: 有没有可能这里用个初始化会更好？感觉每次第一天的action和critic预测都怪怪的
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if not self.multi_critic_buffer_tf:
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        else:
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        # 如果都结束了，意味着就要重开了，所以active_masks都重置为1
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        # TODO: truncation和termination的处理方式分别是？truncation后面会加个预测的value，termination后面加0
        if not self.multi_critic_buffer_tf:
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        else:
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if not self.multi_critic_buffer_tf:
            self.critic_buffer[0].insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        else:
            for agent_id in range(self.num_agents):
                self.critic_buffer[agent_id].insert(
                    share_obs[:, agent_id],
                    rnn_states_critic[:, agent_id],
                    values[:, agent_id],
                    rewards[:, agent_id],
                    masks[:, agent_id],
                    bad_masks[:, agent_id],
                )

    def off_insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        terms = np.full((self.algo_mechanism_args["train"]["n_rollout_threads"], 1), False)
        for i in range(self.algo_mechanism_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                if not (
                    "bad_transition" in infos[i][0].keys()
                    and infos[i][0]["bad_transition"] == True
                ):
                    terms[i][0] = True

        train_available = np.full((self.algo_mechanism_args["train"]["n_rollout_threads"], 1), True)
        for i in range(self.algo_mechanism_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_mechanism_args["train"]["n_rollout_threads"], self.off_num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()
            if 'train_available' in infos[i][0]:
                train_available[i][0] = infos[i][0]['train_available']
        data = (
            share_obs[:, 0],  # (n_threads, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards[:, 0],  # (n_threads, 1)
            np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
            valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
            terms,  # (n_threads, 1)
            next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
            next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            train_available,  # (n_threads, 1)
        )

        self.off_buffer.insert(data)
    
    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if not self.multi_critic_buffer_tf:
            next_value, _ = self.critic[0].get_values(
                self.critic_buffer[0].share_obs[-1],
                self.critic_buffer[0].rnn_states_critic[-1],
                self.critic_buffer[0].masks[-1],
            )
            next_value = _t2n(next_value)
            self.critic_buffer[0].compute_returns(next_value, self.value_normalizer[0])
        else:
            for agent_id in range(self.num_agents):
                next_value, _ = self.critic[agent_id].get_values(
                    self.critic_buffer[agent_id].share_obs[-1],
                    self.critic_buffer[agent_id].rnn_states_critic[-1],
                    self.critic_buffer[agent_id].masks[-1],
                )
                next_value = _t2n(next_value)
                self.critic_buffer[agent_id].compute_returns(next_value, self.value_normalizer[agent_id])

    def train(self):
        """Train the model."""
        raise NotImplementedError
    
    def off_train(self):
        """Train the mechanism model."""
        raise NotImplementedError
    
    @torch.no_grad()
    def get_off_actions(self, obs, available_actions=None, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        if self.args["algo_mechanism"] == "hasac":
            actions = []
            for agent_id in range(self.off_num_agents):
                if (
                    len(np.array(available_actions).shape) == 3
                ):  # (n_threads, n_agents, action_number)
                    actions.append(
                        _t2n(
                            self.off_actor[agent_id].get_actions(
                                obs[:, agent_id],
                                available_actions[:, agent_id],
                                add_random,
                            )
                        )
                    )
                else:  # (n_threads, ) of None
                    actions.append(
                        _t2n(
                            self.off_actor[agent_id].get_actions(
                                obs[:, agent_id], stochastic=add_random
                            )
                        )
                    )
        else:
            actions = []
            for agent_id in range(self.off_num_agents):
                actions.append(
                    _t2n(self.off_actor[agent_id].get_actions(obs[:, agent_id], add_random))
                )
        return np.array(actions).transpose(1, 0, 2)
    
    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        for critic_buffer_id in range(len(self.critic_buffer)):
            self.critic_buffer[critic_buffer_id].after_update()

    @torch.no_grad()
    def eval(self, usage='eval'):
        """Evaluate the model."""
        self.logger.eval_init(usage)  # logger callback at the beginning of evaluation
        # NOTICE: 对于multi_lt_transship环境，应该eval_episodes = n_eval_rollout_threads，不然可能log时会有bug
        eval_episode = 0
        if self.algo_args["eval"]["dataset_for_eval_test_tf"]:
            eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset_for_test_datatset(usage)
        else:
            eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset(usage)

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            off_new_obs, off_new_share_obs = self.eval_envs.step_prepare(eval_actions)
            off_actions = self.get_off_actions(off_new_obs, add_random=False)
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(off_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                eval_avg_rewards = self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                return np.mean(eval_avg_rewards)

    def prep_rollout(self):
        """Prepare for rollout."""
        if self.share_param:
            self.actor[0].prep_rollout()
        else:
            for agent_id in range(self.num_agents):
                self.actor[agent_id].prep_rollout()
        if not self.multi_critic_buffer_tf:
            self.critic[0].prep_rollout()
        else:
            for agent_id in range(self.num_agents):
                self.critic[agent_id].prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        if not self.multi_critic_buffer_tf:
            self.critic[0].prep_training()
        else:
            for agent_id in range(self.num_agents):
                self.critic[agent_id].prep_training()

    def save(self):
        """Save model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
        if not self.multi_critic_buffer_tf:
            policy_critic = self.critic[0].critic
            torch.save(
                policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
            )
            if self.value_normalizer is not None:
                torch.save(
                    self.value_normalizer[0].state_dict(),
                    str(self.save_dir) + "/value_normalizer" + ".pt",
                )
        else:
            for agent_id in range(self.num_agents):
                policy_critic = self.critic[agent_id].critic
                torch.save(
                    policy_critic.state_dict(),
                    str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
                )
                if self.value_normalizer is not None:
                    torch.save(
                        self.value_normalizer[agent_id].state_dict(),
                        str(self.save_dir) + "/value_normalizer" + str(agent_id) + ".pt",
                    )

    def restore(self, last_best_tf=False):
        """Restore model parameters."""
        path = str(self.save_dir) if last_best_tf else str(self.algo_args["train"]["model_dir"])
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                path
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not self.algo_args["render"]["use_render"]:
            if not self.multi_critic_buffer_tf:
                policy_critic_state_dict = torch.load(
                    path + "/critic_agent" + ".pt"
                )
                self.critic[0].critic.load_state_dict(policy_critic_state_dict)
                if self.value_normalizer is not None:
                    value_normalizer_state_dict = torch.load(
                        path
                        + "/value_normalizer"
                        + ".pt"
                    )
                    self.value_normalizer[0].load_state_dict(value_normalizer_state_dict)
            else:
                for agent_id in range(self.num_agents):
                    policy_critic_state_dict = torch.load(
                        path
                        + "/critic_agent"
                        + str(agent_id)
                        + ".pt"
                    )
                    self.critic[agent_id].critic.load_state_dict(policy_critic_state_dict)
                    if self.value_normalizer is not None:
                        value_normalizer_state_dict = torch.load(
                            path
                            + "/value_normalizer"
                            + str(agent_id)
                            + ".pt"
                        )
                        self.value_normalizer[agent_id].load_state_dict(
                            value_normalizer_state_dict
                        )

    def mechanism_restore(self):
        """Restore mechanism agent's model parameters."""
        for agent_id in range(self.off_num_agents):
            self.off_actor[agent_id].restore(self.algo_mechanism_args["train"]["model_dir"], agent_id)
        if not self.algo_mechanism_args["render"]["use_render"]:
            self.off_critic.restore(self.algo_mechanism_args["train"]["model_dir"])
            if self.off_value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_mechanism_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.off_value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()
