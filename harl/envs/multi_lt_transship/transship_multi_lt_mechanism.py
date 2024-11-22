import numpy as np
from gym import spaces

from harl.envs.multi_lt_transship.transship_multi_lt import MultiLtTransshipEnv

class MultiLtTransshipMechanismEnv(MultiLtTransshipEnv):

    def __init__(self,args):
        super(MultiLtTransshipMechanismEnv, self).__init__(args)

        # only for mechanism agent
        self.actor_obs_type_mechanism = self.args['actor_obs_type_mechanism']  # 'intentions_only', 'same_as_critic'
        self.critic_obs_type_mechanism = self.args['critic_obs_type_mechanism']  # 'intentions_only', 'full'
        self.mechanism_share_observation_space = [self.get_mechanism_obs_space(self.critic_obs_type_mechanism)]
        self.mechanism_observation_space = ([self.get_mechanism_obs_space(self.actor_obs_type_mechanism)]
                                            if self.actor_obs_type_mechanism != 'same_as_critic' 
                                            else self.mechanism_share_observation_space)
        self.mechanism_action_space = self.get_mechanism_action_space()

    def get_mechanism_obs_space(self, mechanism_obs_type):
        '''
        Get the observation space for the mechanism agent
        '''
        basic_dim = self.n_agents * (3 + self.inventory_est_horizon)
        if mechanism_obs_type == 'intentions_only':
            obs_dim = basic_dim
        elif mechanism_obs_type == 'full':
            order_dim = self.n_agents
            other_dim = self.get_actor_obs_space('full', self.intransit_info_for_actor_critic)['agent0'].shape[0]
            obs_dim = order_dim + basic_dim + other_dim
        else:
            raise Exception('wrong mechanism_obs_type')
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def get_mechanism_action_space(self):
        '''
        Get the action space for the mechanism agent
        '''
        # central agent作为mechanism存在
        if self.product_allocation_method == 'mechanism_agent':
            return [spaces.Box(low=np.array([0.] * self.n_agents), high=np.array([1.] * self.n_agents), shape=(self.n_agents,), dtype=np.float32)]
        # central agent作为decision agent存在
        else:
            transship_dim_list = []
            for agent_name in self.agents:
                transship_dim = (self.transship_ub[agent_name] - self.transship_lb[agent_name]) / self.transship_step_size 
                transship_dim_list.append(transship_dim)
            return [spaces.MultiDiscrete([int(transship_dim) for transship_dim in transship_dim_list])]
        
    def step_prepare(self, actions):
        '''
        输入每个agent的actions，来帮助生成并返回mechanism_agent的obs，不做状态更新
        '''
        # 转换格式为订货和意图转运量, 并存储下来
        order_amounts, transship_intentions = self.action_map(actions) 
        self.order = order_amounts
        self.transship_intend = transship_intentions
        critic_obs = self.get_step_mechanism_obs(self.state_normalize, self.critic_obs_type_mechanism)
        actor_obs = self.get_step_mechanism_obs(self.state_normalize, self.actor_obs_type_mechanism) if self.actor_obs_type_mechanism != 'same_as_critic' else critic_obs
        return [[actor_obs], [critic_obs]]

    def step(self, actions):
        '''
        输入只包含mechanism agent的action，完成状态更新，并返回除mechanism_agent外的obs, rewards, dones, infos
        '''
        # 获取transship_ratio
        # 获取意图转运量/转运分配概率
        transship_intentions, transship_fulfill_ratio = self.action_map_mechanism(actions)
        # transship_fulfill_ratio = actions[0].copy()
        order_amounts = self.order
        rewards = self.state_update(order_amounts, transship_intentions, transship_fulfill_ratio)
        # 更新demand_est_dict
        self.demand_est_dict, self.demand_std_est_dict = self.estimate_demand_mean_std()
        # 更新MOQ_set
        self.update_MOQ_set()
        # 获取 obs
        actor_obs = self.get_step_obs(self.state_normalize, self.actor_obs_type, self.intransit_info_for_actor_critic)
        critic_obs = self.get_step_obs_critic(self.state_normalize, self.critic_obs_type, self.intransit_info_for_actor_critic, 
                                              self.demand_info_for_critic, self.inventory_est_for_critic_tf, self.inventory_est_others_for_critic_tf)
        # 考虑自私因素的reward
        rewards_self_interest = self.get_processed_rewards(rewards) 
        # algo的输入格式要求
        rewards_self_interest = [[rewards_self_interest[agent_name]] for agent_name in self.agents]

        # 训练阶段不设置done，因为这个done会让算法以为是自然结束的(termination)，从而认为后续value为0，但实际上是truncation，故全部False
        # 验证/测试阶段，因为不参与训练，所以为了方便就还是使用done来标识结束。TODO: 后续可能要训练/验证的结束都统一标识
        if self.step_num >= self.episode_length and (not self.train_tf):
            agent_done = [True for _ in range(self.n_agents)]
        else:
            agent_done = [False for _ in range(self.n_agents)]
        # if(self.step_num >= self.episode_length):
        #     agent_done = [True for _ in range(self.n_agents)]
        # else:
        #     agent_done = [False for _ in range(self.n_agents)]

        # 只有eval/test的时候才会用到info，方便tensorboard上进行展示
        agent_info = [{} for _ in range(self.n_agents)]
        if not self.train_tf:
            agent_info = self.get_info()

        # 如果是train的话，那么如果step_num >= episode_length，那么就释放“truncation”信号，提前reset
        if self.step_num >= self.episode_length and self.train_tf:
            for agent_info_per in agent_info:
                agent_info_per["bad_transition"] = True

        # Add this before the return statement
        train_available = self.check_train_available(transship_intentions)  
        for agent_info_per in agent_info:
            agent_info_per['train_available'] = train_available
            agent_info_per['mechanism_reward'] = self.mechanism_reward['agent0']

        return [self.unwrap(actor_obs), self.unwrap(critic_obs), rewards_self_interest, agent_done, agent_info, self.get_avail_actions()]
    
    def check_train_available(self, transship_intentions):
        '''
        检查是否可以参与训练
        '''
        if self.episode_length - self.step_num <= 20:
            return False
        if self.product_allocation_method == 'mechanism_agent':
            intentions = self.unwrap(transship_intentions)
            intentions_sum = sum(intentions)
            pos_count = sum(1 for i in intentions if i > 1e-5)
            neg_count = sum(1 for i in intentions if i < -1e-5)

            if abs(intentions_sum) < 1e-5 or pos_count == 0 or neg_count == 0:
                return False
            if (intentions_sum > 1e-5 and pos_count == 1) or (intentions_sum < -1e-5 and neg_count == 1):
                return False

        return True

    def action_map(self, actions):
        '''
        把actions映射到实际的订货和转运量
        '''
        actions = self.wrap(np.squeeze(actions))

        if self.action_type == 'heu_discrete':
            order_amounts = {key:val[0] for (key,val) in actions.items()}
            transship_intentions = {key:val[1] for (key,val) in actions.items()}
            return order_amounts, transship_intentions
        
        # 这两个都是action ratio, 实际量要再乘以对应est_demand（其实也不是那么简单）
        order_action_ratio_dict = {agent_name: 0 for agent_name in self.agents}
        transship_intention_action_ratio_dict = {agent_name: 0 for agent_name in self.agents}
        # 还需要记录下transship_action_index， 因为transship_type == 'half_autonomy'的时候，需要根据这个来决定是否转运
        transship_intention_action_index_dict = {agent_name: 0 for agent_name in self.agents}

        if self.action_type == 'discrete':
            for agent_name in self.agents:
                # 考虑MOQ
                if self.MOQ[agent_name] > 0 or self.average_frequency[agent_name] > 1:
                    order_dim = self.order_ub[agent_name] / self.order_step_size + 2
                    order_index = actions[agent_name] % order_dim 
                    # 计算对应的ratio
                    order_ratio = ((order_index - 1) * self.order_step_size + 1) if order_index > 0 else 0
                # 不考虑MOQ
                else:
                    order_dim = self.order_ub[agent_name] / self.order_step_size + 1
                    order_index = actions[agent_name] % order_dim 
                    # 计算对应的ratio
                    order_ratio = order_index * self.order_step_size

                transship_index = actions[agent_name] // order_dim
                # 计算对应的ratio
                transship_ratio = transship_index * self.transship_step_size + self.transship_lb[agent_name]
                order_action_ratio_dict[agent_name] = order_ratio
                transship_intention_action_ratio_dict[agent_name] = transship_ratio
                transship_intention_action_index_dict[agent_name] = transship_index
        elif self.action_type == 'multi_discrete':
            for agent_name in self.agents:
                order_index=actions[agent_name][0]
                transship_index=actions[agent_name][1]
                # 计算对应的ratio
                if self.MOQ[agent_name] > 0 or self.average_frequency[agent_name] > 1:
                    order_ratio = ((order_index - 1) * self.order_step_size + 1) if order_index > 0 else 0
                else:
                    order_ratio = order_index * self.order_step_size
                transship_ratio = transship_index * self.transship_step_size + self.transship_lb[agent_name]
                order_action_ratio_dict[agent_name] = order_ratio
                transship_intention_action_ratio_dict[agent_name] = transship_ratio
                transship_intention_action_index_dict[agent_name] = transship_index
        elif self.action_type == 'continue':
            for agent_name in self.agents:
                order_ratio, transship_ratio = actions[agent_name]
                order_action_ratio_dict[agent_name] = order_ratio
                transship_intention_action_ratio_dict[agent_name] = transship_ratio
        else:
            raise Exception("wrong action_type")
        
        # 把ratio/index转换为实际的量
        order_amounts = {agent_name: round(max(order_action_ratio_dict[agent_name] - 1, 0) * self.demand_est_dict[agent_name] 
                         + (self.MOQ_set[agent_name] if order_action_ratio_dict[agent_name] > 0 else 0))
                         for agent_name in self.agents}
        if self.transship_type == 'pure_autonomy':
            # 最大转运量限制
            transship_out_ub = {}
            if self.reactive_tf:
                transship_out_ub = {agent_name: max(self.inventory[agent_name], 0) for agent_name in self.agents}
            else:
                cur_demand = {agent_name: self.demand_dict[agent_name][self.step_num]
                            for agent_name in self.agents}
                transship_out_ub = {agent_name: max(self.inventory[agent_name] - cur_demand[agent_name], 0) for agent_name in self.agents}

            transship_intentions = {agent_name: round(transship_intention_action_ratio_dict[agent_name] * self.demand_est_dict[agent_name])
                                    if transship_intention_action_ratio_dict[agent_name] > 0
                                    else round(transship_intention_action_ratio_dict[agent_name] * transship_out_ub[agent_name])
                                    for agent_name in self.agents}
        elif self.transship_type == 'half_autonomy':
            # 1 表示转运，0表示不转运
            transship_intentions = {agent_name: transship_intention_action_index_dict[agent_name] 
                                    for agent_name in self.agents}
        elif self.transship_type == 'no':
            # 每个人都不转运
            transship_intentions = {agent_name: 0 for agent_name in self.agents}
        elif self.transship_type == 'force_to_transship':
            # 每个人都转运
            transship_intentions = {agent_name: 1 for agent_name in self.agents}
        else:
            raise Exception('wrong transship_type')


        return order_amounts, transship_intentions
                
    def action_map_mechanism(self, actions):
        '''
        把actions映射到转运量/转运分配概率
        '''
        if self.product_allocation_method == 'mechanism_agent':
            transship_intentions = self.transship_intend
            cur_demand = {agent_name: self.demand_dict[agent_name][self.step_num]
                            for agent_name in self.agents}
            # 把transship_intentions转换为transship_actual
            ## 防止transship的量多于库存，如果不是pure_autonomy的话，transship_intentions就是是否愿意让central controller支配转运，故不需要这个限制
            if self.reactive_tf and self.transship_type == 'pure_autonomy':
                transship_intentions = {agent_name: max(- max(self.inventory[agent_name], 0), transship_intentions[agent_name])
                                        for agent_name in self.agents}
            else:
                transship_intentions = {agent_name: max(- max(self.inventory[agent_name] - cur_demand[agent_name], 0), transship_intentions[agent_name])
                                        for agent_name in self.agents}
            transship_fulfill_ratio = actions[0].copy()
        else:
            actions = self.wrap(np.squeeze(actions))
            # 若是central agent作为decision agent存在，那么transship_intend表示的是否参与转运调配
            transship_intend_tf = self.transship_intend.copy()
            # 读取action，并转换为transship_intentions
            transship_intention_action_ratio_dict = {agent_name: 0 for agent_name in self.agents}
            for agent_name in self.agents:
                transship_index=actions[agent_name]
                transship_ratio = transship_index * self.transship_step_size + self.transship_lb[agent_name]
                transship_intention_action_ratio_dict[agent_name] = transship_ratio
            transship_intentions = {agent_name: round(transship_intention_action_ratio_dict[agent_name] * self.demand_est_dict[agent_name])
                                    if transship_intention_action_ratio_dict[agent_name] > 0
                                    else round(transship_intention_action_ratio_dict[agent_name] * max(self.inventory[agent_name], 0))
                        for agent_name in self.agents}
            # 受制于库存限制，对transship_intentions进行修正
            cur_demand = {agent_name: self.demand_dict[agent_name][self.step_num]
                for agent_name in self.agents}
            if self.reactive_tf:
                transship_intentions = {agent_name: max(- max(self.inventory[agent_name], 0), transship_intentions[agent_name])
                                        for agent_name in self.agents}
            else:
                transship_intentions = {agent_name: max(- max(self.inventory[agent_name] - cur_demand[agent_name], 0), transship_intentions[agent_name])
                                        for agent_name in self.agents}
            # 受制于各agent主观意愿，对transship_intentions进行修正
            transship_intentions = {agent_name: transship_intentions[agent_name] * transship_intend_tf[agent_name]
                                    for agent_name in self.agents}
            transship_fulfill_ratio = {agent_name: 1 for agent_name in self.agents}

        return transship_intentions, transship_fulfill_ratio
    
    def get_est_inv_with_est_demand(self):
        '''
        使用预估需求估计未来库存水平
        假设期间不transship + order, self.inventory_est_horizon时间内, 每天还剩多少库存(+)/缺货多少(-)。
        为避免信息冗余, 不包含当天的库存
        '''
        # 根据self.intransit, self.inventory, self.demand_dict update self.est_inventory
        est_inventory = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        order = self.order
        # est_arrival_qty_list: {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        est_arrival_qty_list = self.get_est_arrival_qty_list(self.intransit, order)
        for t_shift in range(self.inventory_est_horizon):
            # intransit, arrival_items = self.update_intransit(order, intransit, self.step_num + t_shift)
            order = {agent_name: 0 for agent_name in self.agents}
            for agent_name in self.agents:
                arrival_num = est_arrival_qty_list[agent_name][t_shift-1] if t_shift > 0 else 0
                inv = self.inventory[agent_name] if t_shift == 0 else est_inventory[agent_name][t_shift - 1]
                inv = inv if self.backlog_tf else max(0, inv)
                inv += arrival_num
                demand = self.demand_est_dict[agent_name]
                est_inventory[agent_name][t_shift] = inv - demand
        return est_inventory
    
    def get_est_arrival_qty_list(self, intransit, order):
        '''
        获取未来每天到达的量
        '''
        est_arrival_qty_list = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        # Iterate over each agent to estimate arrival quantities
        for agent_name in self.agents:
            # Access the in-transit items for each agent
            agent_intransit = intransit[agent_name]
            # Check each fragment/stage for items
            for frag_idx, stage_items in enumerate(agent_intransit):
                # For each item in this fragment/stage
                for item in stage_items:
                    lt_gen = self.LT_generators[agent_name][frag_idx]
                    LT_left = round(self.get_dist_mean_std(lt_gen)[0] - item['passed_days'])
                    for frag_idx_left in range(frag_idx+1, len(agent_intransit)):
                        LT_left += round(self.get_dist_mean_std(self.LT_generators[agent_name][frag_idx_left])[0])
                    qty_final = round(item['qty'] * (1 - self.get_dist_mean_std(self.shipping_loss_generators[agent_name])[0]))
                    arrival_day = max(LT_left -1, 0)
                    if arrival_day < self.inventory_est_horizon:
                        est_arrival_qty_list[agent_name][arrival_day] += qty_final

        return est_arrival_qty_list
    
    def get_step_mechanism_obs(self, state_normalize, mechanism_obs_type):
        '''
        获取mechanism_agent的obs
        '''
        demand_est_mean_vec = np.array([self.demand_mean_sta_dict[agent_name] for agent_name in self.agents])
        demand_est_vec = np.array([self.demand_est_dict[agent_name] for agent_name in self.agents])
        norm_helper = np.maximum(demand_est_vec, 1)
        intentions_scale_helper = np.average(norm_helper)
        intentions_obs = self.unwrap(self.transship_intend)
        intentions_obs_rel_inv = np.array([self.transship_intend[agent_name] / max(self.inventory[agent_name], 1) 
                                            for agent_name in self.agents])
        est_inventory = self.get_est_inv_with_est_demand()
        order_obs = np.array([])
        other_obs = np.array([])
        if mechanism_obs_type == 'full':
            order_obs = np.array([self.order[agent_name] for agent_name in self.agents])
            other_obs = self.get_step_obs(state_normalize, 'full', self.intransit_info_for_actor_critic)['agent0']
        
        if state_normalize:
            order_obs = (order_obs / norm_helper- 1) if len(order_obs) > 0 else order_obs
            order_obs = self.clip(order_obs, self.state_clip_list[0], self.state_clip_list[1])
            intentions_obs = (intentions_obs / max(intentions_scale_helper, 1e-6)) if len(intentions_obs) > 0 else intentions_obs
            demand_est_vec = demand_est_vec / np.maximum(demand_est_mean_vec, 1e-6) - 1
            intentions_obs_rel_inv = self.clip(intentions_obs_rel_inv, self.state_clip_list[0], self.state_clip_list[1])
            est_inventory_vec = np.array([est_inventory_i / np.maximum(1/2 * norm_helper[agent_idx], 1e-6) - 1 
                                          for agent_idx, agent_name in enumerate(self.agents)
                                          for est_inventory_i in est_inventory[agent_name]])
            est_inventory_vec = self.clip(est_inventory_vec, self.state_clip_list[0], self.state_clip_list[1])
        
        return np.concatenate([intentions_obs, intentions_obs_rel_inv, demand_est_vec, order_obs, other_obs, est_inventory_vec])
    
    def state_update(self, order_amounts, transship_intentions, transship_fulfill_ratio=None):
        '''
        更新状态, 计算reward
        
        Inputs:
            - order_amounts: dict, 订货量
            - transship_intentions: dict, 转运意图
            - transship_fulfill_ratio: dict, each agent被sample到fulfill transshipment intention的概率
        '''
        cur_demand = {agent_name: self.demand_dict[agent_name][self.step_num]
                      for agent_name in self.agents}
        rewards = {agent_name: 0 for agent_name in self.agents}
        # 计算transship matrix, payment_arr
        self.transship_matrix, payment_arr = self.dist_mechanism.get_mechanism_result(self.unwrap(transship_intentions), 
                                                                                      transship_fulfill_ratio = transship_fulfill_ratio)
        # 表示从transship中获得了多少货物
        self.transship_actual = self.wrap(self.transship_matrix.sum(axis=1))
        # 更新后的在途情况， 以及当天到货的货物
        self.intransit, arrival_items = self.update_intransit(self.order, self.intransit, self.step_num)

        # 更新状态 + 计算reward 
        for agent_idx, agent_name in enumerate(self.agents):
            self.ordering_times[agent_name] += (1 if self.order[agent_name] > 0 else 0)
            self.transship_intend[agent_name] = transship_intentions[agent_name]
            inv_cost = 0  # 所消耗的库存的订货/transship成本
            # reactive: 先transship, 再fulfill demand, 再到货
            if self.reactive_tf:
                # 记录transship后的库存成本信息
                if self.transship_actual[agent_name] > 0:
                    self.inventory_batch[agent_name].append({'qty': self.transship_actual[agent_name], 
                                                            'avg_cost': payment_arr[agent_idx] / self.transship_actual[agent_name] 
                                                            if self.transship_actual[agent_name] > 0 else 0})
                transship_out_qty = max(- self.transship_actual[agent_name], 0)
                transship_in_qty = max(self.transship_actual[agent_name], 0)

                # 用transship的货交付延期需求
                demand_fulfilled_late_by_transship = max(min(-self.inventory[agent_name], self.transship_actual[agent_name]), 0)
                inventory_after_transship = self.inventory[agent_name] + self.transship_actual[agent_name]

                # 用库存交付当天需求
                self.demand_fulfilled_intime[agent_name] = max(min(inventory_after_transship, cur_demand[agent_name]), 0)
                self.shortage[agent_name] = max(cur_demand[agent_name] - inventory_after_transship, 0)
                inventory_after_fulfill = inventory_after_transship - (cur_demand[agent_name] if self.backlog_tf else self.demand_fulfilled_intime[agent_name])
                self.inventory_before_arrive[agent_name] = inventory_after_fulfill
                self.transship_instantly_used[agent_name] = max(transship_in_qty - max(inventory_after_fulfill, 0), transship_in_qty)
                # 到货
                arrival_num = sum([item['qty'] for item in arrival_items[agent_name]])
                # 记录最新到货这批库存的成本信息
                for arrival_item in arrival_items[agent_name]:
                    self.inventory_batch[agent_name].append(arrival_item.copy())
                demand_fulfilled_late_by_order = max(min(-inventory_after_fulfill, arrival_num), 0)
                self.inventory[agent_name] = inventory_after_fulfill + arrival_num

            # proactive: 先fulfill demand, 再transship
            else:
                # 交付当天的需求
                inv_start = self.inventory[agent_name]
                self.demand_fulfilled_intime[agent_name] = max(min(inv_start, cur_demand[agent_name]), 0)
                self.shortage[agent_name] = max(cur_demand[agent_name] - inv_start, 0)
                inventory_after_fulfill = inv_start - (cur_demand[agent_name] if self.backlog_tf else self.demand_fulfilled_intime[agent_name])

                # 记录transship后的库存成本信息
                if self.transship_actual[agent_name] > 0:
                    self.inventory_batch[agent_name].append({'qty': self.transship_actual[agent_name], 
                                                            'avg_cost': payment_arr[agent_idx] / self.transship_actual[agent_name] 
                                                            if self.transship_actual[agent_name] > 0 else 0})
                demand_fulfilled_late_by_transship = max(min(-inventory_after_fulfill, self.transship_actual[agent_name]), 0)
                transship_out_qty = max(- self.transship_actual[agent_name], 0)
                inventory_after_transship = inventory_after_fulfill + self.transship_actual[agent_name]
                self.inventory_before_arrive[agent_name] = inventory_after_transship
                self.transship_instantly_used[agent_name] = 0

                # 到货
                arrival_num = sum([item['qty'] for item in arrival_items[agent_name]])
                # 记录最新到货这批库存的成本信息
                for arrival_item in arrival_items[agent_name]:
                    self.inventory_batch[agent_name].append(arrival_item.copy())
                demand_fulfilled_late_by_order = max(min(-inventory_after_transship, arrival_num), 0)
                self.inventory[agent_name] = inventory_after_transship + arrival_num
                
            # 需求实现的明细记录
            self.demand_fulfilled_late[agent_name] = demand_fulfilled_late_by_transship + demand_fulfilled_late_by_order
            self.demand_fulfilled_all[agent_name] = self.demand_fulfilled_intime[agent_name] + self.demand_fulfilled_late[agent_name]
            # 更新库存明细，返回消耗库存的订货成本
            inv_cost = self.consume_inventory(agent_name, self.demand_fulfilled_all[agent_name] + transship_out_qty)

            # inventory_batch里的库存应该和inventory相符
            assert max(self.inventory[agent_name], 0) == sum([item['qty'] for item in self.inventory_batch[agent_name]]), 'inventory_batch和inventory不符'
                
            # 成本计算
            ## 计算运费（只做展示，并不参与计算）
            self.shipping_cost_pure[agent_name] = sum([self.shipping_cost_matrix[agent_idx][j] * self.transship_matrix[agent_idx][j] 
                                                       if self.transship_matrix[agent_idx][j] > 0 else 0 for j in range(self.n_agents)])
            ## 计算运费 + 买卖货的费用。（如果是selling_first，那么对于买货的一方，他的cost会在卖掉时才计算，且为了方便是放在ordering_cost进行计算的）
            self.shipping_cost_all[agent_name] = payment_arr[agent_idx] if (
                self.payment_type == 'pay_first' or self.payment_type == 'product_first' or self.transship_actual[agent_name] < 0) else 0
            ## 持货成本
            self.holding_cost[agent_name] = self.H[agent_name] * max(inventory_after_fulfill, 0)
            ## backlog成本
            self.penalty_cost[agent_name] = self.B[agent_name] * self.shortage[agent_name]
            ## selling revenue
            self.selling_revenue[agent_name] = self.P[agent_name] * self.demand_fulfilled_all[agent_name]
            ## 订货成本
            if self.payment_type == 'pay_first':
                self.ordering_cost[agent_name] = self.C[agent_name] * self.order[agent_name] + self.K[agent_name] * (1 if self.order[agent_name] > 0 else 0)
            elif self.payment_type == 'product_first':
                self.ordering_cost[agent_name] = sum(arr_item['qty'] * arr_item['avg_cost'] for arr_item in arrival_items[agent_name])
            elif self.payment_type == 'selling_first':
                self.ordering_cost[agent_name] = inv_cost
            else:
                raise Exception('wrong payment_type')

            # 最后一天将仓库内剩余货品按成本价折算
            residual_val = 0
            if self.step_num >= self.episode_length - 1 and (not self.train_tf):
                if self.payment_type == 'pay_first':
                    intransit_value = sum([item['qty'] * item['avg_cost'] 
                                           for frag in range(len(self.intransit[agent_name]))
                                           for item in self.intransit[agent_name][frag] ])
                    residual_val = self.inventory[agent_name] * self.C[agent_name] + intransit_value
                elif self.payment_type == 'product_first':
                    residual_val = max(self.inventory[agent_name], 0) * self.C[agent_name]
                elif self.payment_type == 'selling_first':
                    residual_val = 0
                else:
                    raise Exception('wrong payment_type')
                
            # 是否通过偏移把reward变成均值为0的数
            if 'norm' in self.reward_type and self.train_tf:
                # TODO: 感觉还有待考虑，需要scale到(-1, 1)吗？以及用当期需求/预测需求/平均需求做scale的区别是什么？
                fill_rate_assumed = 0.8
                norm_drift = cur_demand[agent_name] * (- fill_rate_assumed * self.H[agent_name] 
                                                       + self.P[agent_name] 
                                                       - self.C[agent_name] 
                                                       - (1 - fill_rate_assumed) * self.B[agent_name])
            else:
                norm_drift = 0

            # 是否对reward通过demand_est进行scale
            norm_scale = 1
            if self.reward_scale_tf and self.train_tf: 
                norm_scale = max(self.demand_est_dict[agent_name], 1)

            reward = (-self.ordering_cost[agent_name] - self.shipping_cost_all[agent_name] 
                      - self.holding_cost[agent_name] - self.penalty_cost[agent_name] 
                      + self.selling_revenue[agent_name] + residual_val - norm_drift) / norm_scale


            rewards[agent_name] = reward
            self.reward_selfish[agent_name] = reward
        self.step_num += 1
        return rewards
    

if __name__ == '__main__':
    args = {'action_type': "heu_discrete", 
            'stationary_tf': False,
            'k1': 0.5, 'k2': 0.5,
            'payment_type': 'product_first',
            'transship_type': 'force_to_transship',
            'product_allocation_method': {'allocate_func':'anup', 'allocate_args':{'ratio_tf': True}}}
    # {'allocate_func':'anup', 'allocate_args':{'ratio_tf': True}}
    # {'allocate_func':'homo_distance', 'allocate_args':{'ratio_tf': True}}
    env =MultiLtTransshipEnv(args)
    from harl.envs.multi_lt_transship.baseline import Baseline
    baseline = Baseline(args)

    obs = env.reset_for_dataset('eval', 0)[0]

    while(True):  
        actions = baseline.get_action(env, obs) 
        # actions = [(2, 0), (2, 0), (2, 0)]
        print(actions)

        obs = env.step(actions)[0]
        # if env.step_num >= 20:
        #     break

