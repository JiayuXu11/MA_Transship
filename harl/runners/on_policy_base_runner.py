"""Base runner for on-policy algorithms."""

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


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
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
        # deprecated
        self.sample_mean_advantage_tf = algo_args["train"].get("sample_mean_advantage_tf", True)
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
            save_config(args, algo_args, env_args, self.run_dir)
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

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
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
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
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
        # TODO: 这个shape还有待考量
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
        # TODO: 对于multi_lt_transship环境，应该eval_episodes = n_eval_rollout_threads，不然可能log时会有bug
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

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
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

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
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
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break

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
