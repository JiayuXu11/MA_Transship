import os 
import random
import copy

import numpy as np
from gym import spaces

from harl.envs.multi_lt_transship import generator
from harl.envs.multi_lt_transship import ENV_ARGS_DEFAULT, DISTANCE_LIST, TEST_DIR, EVAL_DIR, TRAIN_DIR
from harl.utils.configs_tools import deep_update

class MultiLtTransshipEnv(object):

    def __init__(self,args):
        # basic
        # self.args = copy.deepcopy(args)
        self.args = copy.deepcopy(ENV_ARGS_DEFAULT)
        self.args.update(args)
        ## TODO: 这个scenario或许可以根据参数选择建一个
        self.scenario = self.args["scenario"]
        self.cur_step = 0
        self.episode_length = self.args['episode_length']
        self.n_agents = self.args['num_agents']
        self.agents = ['agent{}'.format(i) for i in range(self.n_agents)]
        self.backlog_tf = self.args['backlog_tf']  # backlog 还是 lost sales
        self.gamma = self.args['gamma']  # discount factor

        # generators
        self.LT_generators = self.scalar_to_dict(self.args['LT_generators'])
        self.demand_generators = self.scalar_to_dict(self.args['demand_generators'])
        self.shipping_loss_generators = self.scalar_to_dict(self.args['shipping_loss_generators'])

        # cost parameter
        self.H = self.scalar_to_dict(self.args['H'])  # holding cost
        self.B = self.scalar_to_dict(self.args['B'])  # backlog/penalty cost
        self.P = self.scalar_to_dict(self.args['P'])  # selling price
        self.C = self.scalar_to_dict(self.args['C'])  # ordering cost
        self.shipping_cost_per_distance = self.args['shipping_cost_per_distance']

        ## ordering frequency
        self.K = self.scalar_to_dict(self.args['K'])  # fixed ordering cost
        self.MOQ = self.scalar_to_dict(self.args['MOQ'])
        self.average_frequency = self.scalar_to_dict(self.args['average_frequency']) # 会根据这个算出来一个MOQ，这个和given的MOQ取max
        
        ## 这个是真实使用的MOQ
        self.MOQ_set = {agent_name: 0 for agent_name in self.agents}

        # transshipment
        self.distance_matrix = DISTANCE_LIST[self.args['distance_matrix_index']][:self.n_agents, :self.n_agents]
        self.shipping_cost_matrix = self.shipping_cost_per_distance * self.distance_matrix
        
        ## transshipment相关参数读取
        self.reactive_tf = self.args['reactive_tf']  # 是否reactive
        self.transship_type = self.args['transship_type']  # 转运类型 'pure_autonomy': 自己决策transship的量, 
                                                           # 'half_autonomy': 只决策是否参与transship, 'no': 不参与transship, 'force_to_transship': 强制转运
        self.product_allocation_method = self.args['product_allocation_method']['allocate_func']  # 'ratio', 'least_distance', 'hier', 'no', 'anup', 'mechanism_agent'
        self.allocate_args = self.args['product_allocation_method'].get('allocate_args', {})
        self.revenue_allocation_method = self.args['revenue_allocation_method']  # 'constant' or 'ratio'
        self.constant_transshipment_price = self.args['constant_transshipment_price']  # only used when revenue_allocation_method is 'constant'
        self.distance_matrix_index = self.args['distance_matrix_index']  
        
        ## transship机制选择
        if self.product_allocation_method == 'homo_distance':
            from harl.envs.multi_lt_transship.dist_mechanism.homo_dist import HomoDistribution as mechanism
        elif self.product_allocation_method == 'least_distance':
            from harl.envs.multi_lt_transship.dist_mechanism.least_dist import LeastDistanceDistribution as mechanism
        elif self.product_allocation_method == 'hier':
            from harl.envs.multi_lt_transship.dist_mechanism.hier_dist import HierDistribution as mechanism
        elif self.product_allocation_method == 'no':
            from harl.envs.multi_lt_transship.dist_mechanism.no_mechanism import NoMechanism as mechanism
        elif self.product_allocation_method == 'anup':
            from harl.envs.multi_lt_transship.dist_mechanism.anupindi_mechanism import AnupindiDistribution as mechanism
        elif self.product_allocation_method == 'baseline':
            from harl.envs.multi_lt_transship.dist_mechanism.baseline_mechanism import BaselineDistribution as mechanism
        # 只在MultiLTTransshipMechanismEnv中才被使用
        elif self.product_allocation_method == 'mechanism_agent':
            from harl.envs.multi_lt_transship.dist_mechanism.agent_mechanism import AgentMechanism as mechanism
        else:
            raise Exception('wrong product_allocation_method')
        self.dist_mechanism = mechanism(self.distance_matrix, self.allocate_args, self)
        self.mechanism_reward_weight_others = self.args.get('mechanism_reward_weight_others', 1)

        # reward calculation
        self.self_interest = self.args['self_interest']  # 自私程度
        self.payment_type = self.args['payment_type']  # 先款后货:'pay_first'，先货后款:'product_first'，先销后款:'selling_first'
        self.reward_type = self.args['reward_type']  # 'profit', 'cost', 'norm_profit', 'norm_cost'
        self.reward_scale_tf = self.args['reward_scale_tf']  # 是否对reward进行scale

        # observation
        self.state_normalize = self.args['state_normalize']  # 是否对state进行normalize
        self.state_clip_list = self.args['state_clip_list']  # 强制控制norm后的state在指定范围，防止出现过于破坏分布的数据点，从而阻碍训练
        self.actor_obs_type = self.args['actor_obs_type']  # 'full', 'partial', 'self', 
        self.critic_obs_type = self.args['critic_obs_type']  # 'full', 'partial', 'self'
        self.demand_info_for_critic = self.args['demand_info_for_critic']  # ['long_quantile', 'long_mean', 'short_all', 'short_mean']
        self.inventory_est_for_critic_tf = self.args['inventory_est_for_critic_tf']  # True or False
        self.inventory_est_others_for_critic_tf = self.args['inventory_est_others_for_critic_tf']  # True or False
        self.intransit_info_for_actor_critic = self.args['intransit_info_for_actor_critic']  # ['tomorrow', 'passed_days']
        self.critic_looking_len = round(1./(1.-self.gamma +1e-10))
        self.inventory_est_horizon = self.args['inventory_est_horizon']  # inventory估计值的时间跨度, 同时也是区分短期和长期的时间跨度
        
        ## observation space
        self.observation_space = self.unwrap(self.get_actor_obs_space(self.actor_obs_type, self.intransit_info_for_actor_critic))
        self.share_observation_space = self.unwrap(self.get_critic_obs_space(self.critic_obs_type, self.intransit_info_for_actor_critic,
                                                                             self.demand_info_for_critic, self.inventory_est_for_critic_tf, 
                                                                             self.inventory_est_others_for_critic_tf))
        
        # action
        self.est_demand_method = self.args['estimate_demand_method']  # 'mean', 'rolling_mean', 'ema'
        self.rolling_window = self.args['rolling_window']  # rolling mean的窗口大小, 只有在'rolling_mean'的时候才有用
        self.alpha = self.args.get('alpha', 0.2)  # 添加 alpha 参数，默认值为 0.2
        self.pre_ema = {agent_name: 0 for agent_name in self.agents}
        self.pre_ema_d_sqr = {agent_name: 0 for agent_name in self.agents}
        self.action_type = self.args['action_type']  # 'discrete', 'continuous' or 'multi_discrete'
        self.order_ub = self.scalar_to_dict(self.args['order_ub'])  # 最高几倍的MOQ(或0.8*(average_frequency-1)*demand_mean)
        self.transship_lb = self.scalar_to_dict(self.args['transship_lb'])  # 最高转运出去是几倍的demand_mean
        self.transship_ub = self.scalar_to_dict(self.args['transship_ub'])  # 最高转运进来是几倍的demand_mean
        self.order_step_size = self.args['order_step_size']  # 订货的步长
        self.transship_step_size = self.args['transship_step_size']  # 转运的步长
        ## action space
        self.action_space = self.get_action_space()

        # initialization
        self.initial_inventory = self.scalar_to_dict(self.args['initial_inventory'])
        self.initial_in_transit = self.scalar_to_dict(self.args['initial_in_transit'])
        # 声明环境中的变量
        self.train_tf = True
        self.step_num = 0
        self.inventory = {agent_name: 0 for agent_name in self.agents}  # 当日库存
        self.order = {agent_name: 0 for agent_name in self.agents}  # 当日订货
        self.transship_intend = {agent_name: 0 for agent_name in self.agents}  # 当日转运意向
        self.transship_actual = {agent_name: 0 for agent_name in self.agents}  # 当日实际转运
        self.transship_matrix = np.zeros((self.n_agents, self.n_agents))  # 当日转运矩阵
        self.demand_dict = {agent_name: [0] * self.episode_length for agent_name in self.agents}  # 全时段demand
        self.LT_dict = {agent_name: [[0] * len(self.LT_generators[agent_name])] * self.episode_length for agent_name in self.agents}  # 全时段LT
        self.shipping_loss_dict = {agent_name: [0] * self.episode_length for agent_name in self.agents}  # 全时段shipping loss
        self.demand_mean_sta_dict = {agent_name: 0 for agent_name in self.agents} # 静态的demand均值
        self.demand_std_sta_dict = {agent_name: 0 for agent_name in self.agents} # 静态的demand标准差   
        self.demand_est_dict = {agent_name: 0 for agent_name in self.agents}  # 次日demand估计
        self.demand_std_est_dict = {agent_name: 0 for agent_name in self.agents}  # 次日demand标准差估计
        self.LT_est_dict = {agent_name: [0] * len(self.LT_generators[agent_name])  for agent_name in self.agents}  # 不同阶段下LT估计
        ## 其中arrival_days是环境中的隐藏变量，不会披露给agent
        self.intransit = {agent_name: [[{'qty': 0, 'passed_days': 0, 'arrival_days': 0, 'avg_cost': 0}]] * len(self.LT_generators[agent_name]) 
                          for agent_name in self.agents}  # 在途具体情况 {agent_name:[[第一段所有在途], [第二段所有在途], ...]}
        
        ## 按顺序记录每批库存的成本信息,帮助实现不同计费模式
        ## NOTICE: 从state角度，这个东西actor应该不需要，但critic或许需要（尤其当有fixed_cost时，或者成本参数会随时间变化时）
        self.inventory_batch = {agent_name: [{'qty': 0, 'avg_cost': 0}, ] for agent_name in self.agents}  # 按顺序记录每批库存的成本信息
        # 记录一些info, 方便tensorboard上展示，目前是只在eval/test的时候才用到 
        self.demand_fulfilled_all = {agent_name: 0 for agent_name in self.agents}
        self.demand_fulfilled_intime = {agent_name: 0 for agent_name in self.agents}
        self.demand_fulfilled_late = {agent_name: 0 for agent_name in self.agents}
        self.shortage = {agent_name: 0 for agent_name in self.agents}
        self.reward_selfish = {agent_name: 0 for agent_name in self.agents}
        self.reward_selfish_cum = {agent_name: 0 for agent_name in self.agents}
        self.reward = {agent_name: 0 for agent_name in self.agents}
        self.reward_cum = {agent_name: 0 for agent_name in self.agents}
        self.shipping_cost_pure = {agent_name: 0 for agent_name in self.agents} # 纯运输费用
        self.shipping_cost_all = {agent_name: 0 for agent_name in self.agents}  # 包含货物费用 + 运输费用
        self.ordering_cost = {agent_name: 0 for agent_name in self.agents}
        self.penalty_cost = {agent_name: 0 for agent_name in self.agents}
        self.holding_cost = {agent_name: 0 for agent_name in self.agents}
        self.selling_revenue = {agent_name: 0 for agent_name in self.agents}
        self.ordering_times = {agent_name: 0 for agent_name in self.agents}
        self.inventory_before_arrive = {agent_name: 0 for agent_name in self.agents}
        self.transship_instantly_used = {agent_name: 0 for agent_name in self.agents}

        # critic obs增强可能需要用到的一些变量
        self.demand_dy_sorted_long = {agent_name: [] for agent_name in self.agents}
        self.demand_long_mean = {agent_name: 0 for agent_name in self.agents}
        self.demand_short = {agent_name: [] for agent_name in self.agents}
        self.demand_short_mean = {agent_name: 0 for agent_name in self.agents}
        self.demand_q5 = {agent_name: 0 for agent_name in self.agents}
        self.demand_q25 = {agent_name: 0 for agent_name in self.agents}
        self.demand_q50 = {agent_name: 0 for agent_name in self.agents}
        self.demand_q75 = {agent_name: 0 for agent_name in self.agents}
        self.demand_q95 = {agent_name: 0 for agent_name in self.agents}
        self.mechanism_reward = {agent_name: 0 for agent_name in self.agents}
        # self.est_inventory = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}

        # test/eval dataset name
        self.dataset_name = self.args['dataset_name']
    
    def get_data_from_generators_dict(self, generators_dict, round_int_tf=True):
        """
        Generate data from the generators_dict
        Outputs:
            - generated_data_dict: dict, one-episode data generated by the generators
        """
        generated_data_dict = {agent_name:[] for agent_name in self.agents}
        for agent_name in self.agents:
            generators_selected = generators_dict[agent_name]
            if not isinstance(generators_selected, list):
                generators_selected = [generators_selected]
            num_gen_list = []
            for generator_selected in generators_selected:
                if generator_selected['gen_func'] == 'uniform':
                    num_gen = generator.uniform(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'normal':
                    num_gen = generator.normal(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'poisson':
                    num_gen = generator.poisson(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'merton':
                    num_gen = generator.merton(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'kim':
                    num_gen = generator.kim_dist(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'constant':
                    num_gen = generator.constant_dist(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'kim_merton':
                    num_gen = generator.kim_merton(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                elif generator_selected['gen_func'] == 'shanshu':
                    agent_idx = int(''.join(filter(str.isdigit, agent_name)))
                    generator_selected['gen_args']['agent_idx'] = agent_idx
                    num_gen = generator.shanshu(length=self.episode_length * 2, **generator_selected['gen_args']).get_data(round_int_tf)
                else:
                    raise Exception('wrong gen_func')
                num_gen_list.append(num_gen)
            generated_data_dict[agent_name] = np.squeeze(np.array(num_gen_list).transpose(1, 0))
            
        return generated_data_dict
    
    def get_dataset_demand_LT_sl(self, usage, index):
        '''
        Get the demand and LT data from the dataset
        Outputs:
            - demand_dict: dict, one-episode demand data for training
            - LT_dict: dict, one-episode lead time data for training
            - shipping_loss_dict: dict, one-episode shipping loss data for training
        '''
        demand_dict = {agent_name:[] for agent_name in self.agents}
        LT_dict = {agent_name:[] for agent_name in self.agents}
        shipping_loss_dict = {agent_name:[] for agent_name in self.agents}
        if usage == 'test':
            data_dir = TEST_DIR.format(self.dataset_name)
        elif usage == 'eval':
            data_dir = EVAL_DIR.format(self.dataset_name)
        else:
            raise Exception('no such usage except test and eval')
        # 如果不存在数据集，就生成一个
        if not os.path.exists(data_dir):
            self.generate_eval_test_wrapper()
        for agent_name in self.agents:
            # Define file paths for demand and lead time (LT) data
            demand_file_path = os.path.join(data_dir, f"{agent_name}/demand_{index}.txt")
            LT_file_path = os.path.join(data_dir, f"{agent_name}/LT_{index}.txt")
            shipping_loss_file_path = os.path.join(data_dir, f"{agent_name}/shippingloss_{index}.txt")
            
            # Read demand data
            with open(demand_file_path, 'r') as demand_file:
                demand_data = demand_file.readlines()
                demand_dict[agent_name] = np.array([int(float(line.strip())) for line in demand_data])
            
            # Read LT data
            with open(LT_file_path, 'r') as LT_file:
                LT_data = LT_file.readlines()
                LT_dict[agent_name] = np.array([list(map(int, line.strip().split())) for line in LT_data])

            # Read shipping loss data
            with open(shipping_loss_file_path, 'r') as shipping_loss_file:
                shipping_loss_data = shipping_loss_file.readlines()
                shipping_loss_dict[agent_name] = np.array([float(line.strip()) for line in shipping_loss_data])
        
        return demand_dict, LT_dict, shipping_loss_dict

    def get_actor_obs_space(self, actor_obs_type, intransit_info_for_actor_critic):
        '''
        Get the observation space for the actor network
        '''
        obs_space = {}
        full_obs_dim_dict = {agent_name: 5 
                             + (len(self.LT_generators[agent_name]) if 'qty_each' in intransit_info_for_actor_critic else 0)
                             + (len(self.LT_generators[agent_name]) if 'passed_days' in intransit_info_for_actor_critic else 0) 
                             + (2 if 'tomorrow' in intransit_info_for_actor_critic else 0) 
                             for agent_name in self.agents}
        refined_obs_dim_dict = {agent_name: 5 
                                + (2 if 'tomorrow' in self.intransit_info_for_actor_critic else 0)
                                for agent_name in self.agents}
        for agent_name in self.agents:
            if actor_obs_type == 'full':
                obs_dim = sum([full_obs_dim_dict[agent_name_i] for agent_name_i in self.agents])
            elif actor_obs_type == 'partial':
                obs_dim = sum([refined_obs_dim_dict[agent_name_i] for agent_name_i in self.agents]) + (
                    full_obs_dim_dict[agent_name] - refined_obs_dim_dict[agent_name])
            elif actor_obs_type == 'self':
                obs_dim = full_obs_dim_dict[agent_name]
            else:
                raise Exception("wrong actor_obs_type")
            obs_space[agent_name] = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        return obs_space
    
    def get_critic_obs_space(self, critic_obs_type, intransit_info_for_actor_critic, 
                             demand_info_for_critic, inventory_est_for_critic_tf, inventory_est_others_for_critic_tf):
        '''
        Get the observation space for the critic network
        '''
        obs_space = {}
        actor_obs_space = self.get_actor_obs_space(critic_obs_type, intransit_info_for_actor_critic)
        full_obs_dim_dict = {agent_name: (5 if 'long_quantile' in demand_info_for_critic else 0) + 
                             (1 if 'long_mean' in demand_info_for_critic else 0) + 
                             (self.inventory_est_horizon if 'short_all' in demand_info_for_critic else 0) + 
                             (1 if 'short_mean' in demand_info_for_critic else 0) + 
                             (1 * self.inventory_est_horizon if inventory_est_for_critic_tf else 0) + 
                             (2 * self.inventory_est_horizon if inventory_est_others_for_critic_tf and inventory_est_for_critic_tf else 0)
                             for agent_name in self.agents}
        refined_obs_dim_dict = {agent_name: (self.inventory_est_horizon if inventory_est_for_critic_tf else 0)
                                for agent_name in self.agents}
        
        for agent_name in self.agents:
            if critic_obs_type == 'full':
                obs_dim = sum([full_obs_dim_dict[agent_name_i] for agent_name_i in self.agents]) + actor_obs_space[agent_name].shape[0]
            elif critic_obs_type == 'partial':
                obs_dim = sum([refined_obs_dim_dict[agent_name_i] for agent_name_i in self.agents]) + (
                    full_obs_dim_dict[agent_name] - refined_obs_dim_dict[agent_name] + actor_obs_space[agent_name].shape[0])
            elif critic_obs_type == 'self':
                obs_dim = full_obs_dim_dict[agent_name] + actor_obs_space[agent_name].shape[0]
            obs_space[agent_name] = (spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32))
        return obs_space

    def get_action_space(self):
        '''
        Get the action space for the actor network
        dependes on the (self.action_type, self.transship_type)
        '''
        action_space = []
        for agent_name in self.agents:
            # 是否考虑MOQ
            if self.MOQ[agent_name] > 0 or self.average_frequency[agent_name] > 1:
                order_dim = self.order_ub[agent_name] / self.order_step_size + 2
            else:
                order_dim = self.order_ub[agent_name] / self.order_step_size + 1
            # 根据transship_type来决定transship_dim
            transship_dim = 0
            if self.transship_type == 'pure_autonomy':
                transship_dim = (self.transship_ub[agent_name] - self.transship_lb[agent_name]) / self.transship_step_size 
            elif self.transship_type == 'half_autonomy':
                transship_dim = 2
            elif self.transship_type == 'no' or self.transship_type == 'force_to_transship':
                transship_dim = 1
            else:
                raise Exception('wrong transship_type')
            # 根据action_type，以及dim来决定action_space
            if self.action_type == 'multi_discrete':
                action_space.append(spaces.MultiDiscrete([int(order_dim), int(transship_dim)]))
            # physical action space
            elif self.action_type == 'discrete':
                action_space.append(spaces.Discrete(int(order_dim * transship_dim)))
            elif self.action_type == 'continuous' or self.action_type == 'heu_discrete':
                action_space.append(spaces.Box(low=np.array([0., self.transship_lb[agent_name]]), 
                                               high=np.array([self.order_ub[agent_name], self.transship_ub[agent_name]]), 
                                               shape=(2,), dtype=np.float32))
            else:
                raise Exception("wrong action_type")
        return action_space
    
    def estimate_demand_mean_std(self):
        '''
        Estimate the mean and std of future demand
        dependes on the (self.demand_generators, self.estimate_demand_method)
        '''
        # 如果是rolling_mean，那么就是用前面的数据来估计；如果是mean，那么就是直接根据分布来估计
        if self.est_demand_method == 'mean':
            demand_est_dict = self.demand_mean_sta_dict
            demand_std_est_dict = self.demand_std_sta_dict
        elif self.est_demand_method == 'rolling_mean':
            rolling_start_idx = max(0, self.step_num + 1 - self.rolling_window)
            demand_est_dict = {agent_name: np.mean(self.demand_dict[agent_name][rolling_start_idx : self.step_num + 1]) 
                               for agent_name in self.agents}
            demand_std_est_dict = {agent_name: np.std(self.demand_dict[agent_name][rolling_start_idx : self.step_num + 1]) 
                               for agent_name in self.agents}
        elif self.est_demand_method == 'ema':
            demand_est_dict = {}
            demand_std_est_dict = {}
            for agent_name in self.agents:
                demand = self.demand_dict[agent_name][self.step_num]
                if self.step_num == 0:
                    demand_est_dict[agent_name] = demand
                    demand_std_est_dict[agent_name] = 3
                    self.pre_ema[agent_name] = demand
                    self.pre_ema_d_sqr[agent_name] = 9
                else:
                    demand_est_dict[agent_name] = self.alpha * demand + (1 - self.alpha) * self.pre_ema[agent_name]
                    demand_std_est_dict[agent_name] = (self.alpha * (demand - demand_est_dict[agent_name])**2 + \
                        (1 - self.alpha) * self.pre_ema_d_sqr[agent_name])**0.5
                    self.pre_ema[agent_name] = demand_est_dict[agent_name]
                    self.pre_ema_d_sqr[agent_name] = demand_std_est_dict[agent_name]**2
        else:
            raise Exception('wrong estimate_demand_method')
        return demand_est_dict, demand_std_est_dict
    
    def estimate_LT_mean(self):
        '''
        Estimate the mean of future LT
        dependes on the (self.LT_generators)
        '''
        LT_est_dict = {agent_name: [self.get_dist_mean_std(LT_gen)[0] for LT_gen in self.LT_generators[agent_name]] 
                       for agent_name in self.agents}
        return LT_est_dict
    
    def update_MOQ_set(self):
        '''
        根据self.MOQ 和 self.avearge_frequency更新self.MOQ_set
        若MOQ_set 为 0, 则不从action维度限制补货
        '''
        self.MOQ_set = {agent_name: max(self.MOQ[agent_name], 
                                        0.8 * (self.average_frequency[agent_name] - 1) * self.demand_est_dict[agent_name])
                        for agent_name in self.agents}
        
    def reset_for_dataset(self, usage, index):
        '''
        Reset the environment according to the dataset
        Args:
            usage: str, 'train' or 'test'
            index: int, the index of the dataset
        '''
        self.train_tf = False
        self.reset_variables(self.train_tf)
        # 从文件里读取数据
        self.demand_dict, self.LT_dict, self.shipping_loss_dict = self.get_dataset_demand_LT_sl(usage, index)

        
        for agent_name in self.agents:
            if self.demand_generators[agent_name]['gen_func'] == 'shanshu':
                agent_idx = int(''.join(filter(str.isdigit, agent_name)))
                self.demand_generators[agent_name]['gen_args']['agent_idx'] = agent_idx
            d_mean, d_std = self.demand_mean_sta_dict[agent_name], self.demand_std_sta_dict[agent_name]
            self.demand_mean_sta_dict[agent_name] = d_mean
            self.demand_std_sta_dict[agent_name] = d_std

        self.demand_est_dict, self.demand_std_est_dict = self.estimate_demand_mean_std()
        self.LT_est_dict = self.estimate_LT_mean()
        # update MOQ_set
        self.update_MOQ_set()
        # 获取 obs
        actor_obs = self.get_step_obs(self.state_normalize, self.actor_obs_type, self.intransit_info_for_actor_critic)
        critic_obs = self.get_step_obs_critic(self.state_normalize, self.critic_obs_type, self.intransit_info_for_actor_critic, 
                                              self.demand_info_for_critic, self.inventory_est_for_critic_tf, self.inventory_est_others_for_critic_tf)
        return self.unwrap(actor_obs), self.unwrap(critic_obs), self.get_avail_actions()

    def reset(self, usage='train'):
        '''
        reset the environment for training
        '''
        self.train_tf = (usage == 'train')
        self.reset_variables(self.train_tf)
        # 通过generator生成数据
        self.demand_dict = self.get_data_from_generators_dict(self.demand_generators)
        self.LT_dict = self.get_data_from_generators_dict(self.LT_generators)
        self.shipping_loss_dict = self.get_data_from_generators_dict(self.shipping_loss_generators, round_int_tf=False)

        self.demand_mean_sta_dict = {}
        self.demand_std_sta_dict = {}
        for agent_name in self.agents:
            d_mean, d_std = self.get_dist_mean_std(self.demand_generators[agent_name])
            self.demand_mean_sta_dict[agent_name] = d_mean
            self.demand_std_sta_dict[agent_name] = d_std

        self.demand_est_dict, self.demand_std_est_dict = self.estimate_demand_mean_std()
        self.LT_est_dict = self.estimate_LT_mean()
        # update MOQ_set
        self.update_MOQ_set()
        # 获取obs
        actor_obs = self.get_step_obs(self.state_normalize, self.actor_obs_type, self.intransit_info_for_actor_critic)
        critic_obs = self.get_step_obs_critic(self.state_normalize, self.critic_obs_type, self.intransit_info_for_actor_critic, 
                                              self.demand_info_for_critic, self.inventory_est_for_critic_tf, self.inventory_est_others_for_critic_tf)
        return self.unwrap(actor_obs), self.unwrap(critic_obs), self.get_avail_actions()
    
    def update_intransit(self, order, intransit, step_num):
        '''
        Return the updated intransit and arrival of items

        Args:
            - order: dict, the order of all agents
        '''
        intransit = copy.deepcopy(intransit)
        arrival_items = {agent_name: [] for agent_name in self.agents}  # agent_name: [{'qty':0, 'avg_cost': 0}, ...]
        for agent_name in self.agents:
            LT_list = [self.LT_dict[agent_name][step_num][frag_index] for frag_index in range(len(self.LT_generators[agent_name]))]
            intransit_frags = intransit[agent_name]
            # 根据order来新添第一段在途
            avg_cost = self.C[agent_name] + (self.K[agent_name] / order[agent_name] if order[agent_name] > 0 else 0)
            if order[agent_name] > 0:
                intransit_frags[0].append({'qty': order[agent_name], 'passed_days': -1, 'arrival_days': LT_list[0], 'avg_cost': avg_cost})
            # 取出对应阶段的所有在途信息
            for fragment_idx in range(len(intransit_frags)):
                remove_items_list = []
                # 更新该阶段的在途
                for intransit_item in intransit_frags[fragment_idx]:
                    intransit_item['passed_days'] += 1
                    # 每个现有的在途passed_days+=1, 如果passed_days>=arrival_days, 就转移到下一段
                    if intransit_item['passed_days'] >= intransit_item['arrival_days']:
                        if fragment_idx >= len(intransit_frags) - 1:
                            # 如果是最后一段，就到达
                            qty_final = round(intransit_item['qty'] * (1 - self.shipping_loss_dict[agent_name][step_num]))
                            arrival_items[agent_name] += [{'qty': qty_final, 
                                                           'avg_cost': intransit_item['avg_cost'] / qty_final * intransit_item['qty']}]
                        else:
                            intransit_frags[fragment_idx + 1].append({'qty': intransit_item['qty'], 
                                                                'passed_days': -1,      # -1是为了在下一步就变成0 
                                                                'arrival_days': round(LT_list[fragment_idx+1]), 
                                                                'avg_cost': intransit_item['avg_cost']})
                        remove_items_list.append(intransit_item)
                # 删除已经到达/到达下一阶段的在途
                for remove_item in remove_items_list:
                    intransit_frags[fragment_idx].remove(remove_item)

        return intransit, arrival_items

    def reset_variables(self, train_tf = True):
        '''
        Reset all the variables 除了demand_dict, LT_dict
        '''
        self.step_num = 0
        self.inventory = {agent_name: self.initial_inventory[agent_name] for agent_name in self.agents}  # 当日库存
        self.inventory_batch = {agent_name: [{'qty': self.initial_inventory[agent_name], 'avg_cost': 0}, ] for agent_name in self.agents}  # 按顺序记录每批库存的成本信息
        self.order = {agent_name: 0 for agent_name in self.agents}  # 当日订货
        # deepcopy
        self.intransit = {agent_name: copy.deepcopy(self.initial_in_transit[agent_name])
                          for agent_name in self.agents}  # 在途具体情况 {agent_name:[[第一段所有在途], [第二段所有在途], ...]}
        # update self.intransit
        self.intransit, _ = self.update_intransit(self.order, self.intransit, self.step_num)
        # transshipment
        self.transship_intend = {agent_name: 0 for agent_name in self.agents}
        self.transship_actual = {agent_name: 0 for agent_name in self.agents}
        self.transship_matrix = np.zeros((self.n_agents, self.n_agents))

        # 验证/测试 阶段存储相关信息以便展示
        self.demand_fulfilled_all = {agent_name: 0 for agent_name in self.agents}
        self.demand_fulfilled_intime = {agent_name: 0 for agent_name in self.agents}
        self.demand_fulfilled_late = {agent_name: 0 for agent_name in self.agents}
        self.shortage = {agent_name: 0 for agent_name in self.agents}
        self.reward_selfish = {agent_name: 0 for agent_name in self.agents}
        self.reward_selfish_cum = {agent_name: 0 for agent_name in self.agents}
        self.reward = {agent_name: 0 for agent_name in self.agents}
        self.reward_cum = {agent_name: 0 for agent_name in self.agents}
        self.shipping_cost_pure = {agent_name: 0 for agent_name in self.agents}
        self.shipping_cost_all = {agent_name: 0 for agent_name in self.agents}
        self.ordering_cost = {agent_name: 0 for agent_name in self.agents}
        self.penalty_cost = {agent_name: 0 for agent_name in self.agents}
        self.holding_cost = {agent_name: 0 for agent_name in self.agents}
        self.selling_revenue = {agent_name: 0 for agent_name in self.agents}
        self.ordering_times = {agent_name: 0 for agent_name in self.agents}
        self.inventory_before_arrive = {agent_name: 0 for agent_name in self.agents}
        self.transship_instantly_used = {agent_name: 0 for agent_name in self.agents}
        self.mechanism_reward = {agent_name: 0 for agent_name in self.agents}
    
    def get_info(self):
        infos=[]
        for agent_id, agent_name in enumerate(self.agents):
            info_dict={}
            info_dict['inventory_before_arrive']= self.inventory_before_arrive[agent_name]
            info_dict['inventory'] = self.inventory[agent_name]
            info_dict['demand'] = self.demand_dict[agent_name][self.step_num - 1] if self.step_num > 0 else 0
            info_dict['order'] = self.order[agent_name]
            info_dict['transship'] = self.transship_actual[agent_name]
            info_dict['transship_intend'] = self.transship_intend[agent_name]
            info_dict['demand_fulfilled_all'] = self.demand_fulfilled_all[agent_name]
            info_dict['demand_fulfilled_intime'] = self.demand_fulfilled_intime[agent_name]
            info_dict['demand_fulfilled_late'] = self.demand_fulfilled_late[agent_name]
            info_dict['shortage'] = self.shortage[agent_name]
            info_dict['reward_selfish'] = self.reward_selfish[agent_name]
            info_dict['reward_selfish_cum'] = self.reward_selfish_cum[agent_name]
            info_dict['reward'] = self.reward[agent_name]
            info_dict['reward_cum'] = self.reward_cum[agent_name]
            info_dict['shipping_cost_all'] = self.shipping_cost_all[agent_name]
            info_dict['shipping_cost_pure'] = self.shipping_cost_pure[agent_name]
            info_dict['penalty_cost'] = self.penalty_cost[agent_name]
            info_dict['holding_cost'] = self.holding_cost[agent_name]
            info_dict['selling_revenue'] = self.selling_revenue[agent_name]
            info_dict['ordering_times'] = self.ordering_times[agent_name]
            info_dict['ordering_cost'] = self.ordering_cost[agent_name]
            info_dict['transship_instantly_used'] = self.transship_instantly_used[agent_name]
            
            infos.append(info_dict)
        return infos
    
    def step(self, actions):
        # 转换格式为订货和意图转运量
        order_amounts, transship_intentions = self.action_map(actions) 
        self.order = order_amounts
        self.transship_intend = transship_intentions
        rewards = self.state_update(order_amounts, transship_intentions)
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

        return [self.unwrap(actor_obs), self.unwrap(critic_obs), rewards_self_interest, agent_done, agent_info, self.get_avail_actions()]
    
    def action_map(self, actions):
        '''
        把actions映射到实际的订货和转运量
        '''
        actions = self.wrap(np.squeeze(actions))

        if self.action_type == 'heu_discrete':
            order_amounts = {key:val[0] for (key,val) in actions.items()}
            if self.transship_type == 'pure_autonomy':
                transship_intentions = {key:val[1] for (key,val) in actions.items()}
            elif self.transship_type == 'half_autonomy':
                transship_intentions = {key:1 if val[1] == 1 else 0 for (key,val) in actions.items()}
            elif self.transship_type == 'no':
                transship_intentions = {key:0 for key in actions.keys()}
            elif self.transship_type == 'force_to_transship':
                transship_intentions = {key:1 for key in actions.keys()}
            else:
                raise Exception('wrong transship_type')
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
            transship_intentions = {agent_name: round(transship_intention_action_ratio_dict[agent_name] * self.demand_est_dict[agent_name])
                                    if transship_intention_action_ratio_dict[agent_name] > 0
                                    else round(transship_intention_action_ratio_dict[agent_name] * max(self.inventory[agent_name], 0))
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
                
        
    def get_step_obs(self, state_normalize, actor_obs_type, intransit_info_for_actor_critic):
        """
        Get step obs (obs for each step)

        Outputs:
            - actors_obs: a dict of list, each list contains the obs for each agent
        """
        actors_obs = {agent_name: [] for agent_name in self.agents}

        full_obs = {agent_name: [] for agent_name in self.agents}  # 包含所有agent的个人obs
        refined_obs = {agent_name: [] for agent_name in self.agents}  # 包含所有agent的提炼后的obs
        for agent_name in self.agents:
            # 当demand_est_method为rolling_mean时，demand_est_mean_scalar = demand_est_scalar
            demand_est_mean_scalar, demand_std_est_mean_scalar = self.demand_mean_sta_dict[agent_name], self.demand_std_sta_dict[agent_name]  # 帮助normalize demand_est_scalar
            demand_est_scalar = self.demand_est_dict[agent_name]  # 帮助normalization, 以及作为obs的一部分
            demand_std_est_scalar = self.demand_std_est_dict[agent_name]  # 帮助normalization, 以及作为obs的一部分
            LT_est_vec = self.LT_est_dict[agent_name]
            inventory_scalar = self.inventory[agent_name]
            demand_scalar = self.demand_dict[agent_name][self.step_num]
            MOQ_scalar = self.MOQ_set[agent_name]
            # 总的在途量
            intransit_all_scalar = 0
            # 每个阶段的在途总量
            intransit_qty_each_frag = []
            # 每个阶段的最大在途passed_days
            intransit_passed_days = []
            # 贝叶斯估计次日到达的量
            intransit_tomorrow_arrival_num = []
            intransit_tomorrow_arrival_prob = []
            for intransit_frag in self.intransit[agent_name]:
                intransit_qty_frag = sum([intransit_item['qty'] for intransit_item in intransit_frag])
                intransit_all_scalar += intransit_qty_frag
                if 'qty_each' in intransit_info_for_actor_critic:
                    intransit_qty_each_frag += [intransit_qty_frag]
                if 'passed_days' in intransit_info_for_actor_critic:
                    intransit_passed_days += [max([intransit_item['passed_days'] for intransit_item in intransit_frag] + [0])]
            if 'tomorrow' in intransit_info_for_actor_critic:
                # 贝叶斯估计次日到达的量
                est_num, est_prob = self.est_next_arrival(agent_name)
                intransit_tomorrow_arrival_num += [est_num]
                intransit_tomorrow_arrival_prob += [est_prob]

            if(state_normalize):
                # 防止MOQ_scalar和demand_est_scalar为0；且当MOQ较小时，使用demand_scalar来normalize
                inventory_intransit_norm_helper = max(demand_est_scalar, 1)
                inventory_scalar_norm = inventory_scalar / inventory_intransit_norm_helper - 1
                inventory_scalar_norm = self.clip(inventory_scalar_norm, self.state_clip_list[0], self.state_clip_list[1])
                demand_scalar_norm = demand_scalar / max(demand_est_scalar, 1e-6) - 1
                demand_scalar_norm = self.clip(demand_scalar_norm, self.state_clip_list[0], self.state_clip_list[1])
                demand_est_scalar_norm = demand_est_scalar / max(demand_est_mean_scalar, 1e-6) - 1
                demand_est_scalar_norm = self.clip(demand_est_scalar_norm, self.state_clip_list[0], self.state_clip_list[1])
                demand_std_est_scalar_norm = demand_std_est_scalar / max(demand_std_est_mean_scalar, 1e-6) - 1
                demand_std_est_scalar_norm = self.clip(demand_std_est_scalar_norm, self.state_clip_list[0], self.state_clip_list[1])
                intransit_all_scalar_norm = intransit_all_scalar / max(inventory_intransit_norm_helper, 1e-6) - 1
                intransit_qty_each_frag_norm = [intransit_qty_each_frag[i] / max(inventory_intransit_norm_helper, 1e-6) - 1 
                                                for i in range(len(intransit_qty_each_frag))]
                intransit_qty_each_frag_norm = self.clip(intransit_qty_each_frag_norm, self.state_clip_list[0], self.state_clip_list[1])
                intransit_passed_days_norm = [intransit_passed_days[i] / max(1/2 * LT_est_vec[i], 1e-6) - 1 
                                              for i in range(len(intransit_passed_days))]
                intransit_passed_days_norm = self.clip(intransit_passed_days_norm, self.state_clip_list[0], self.state_clip_list[1])
                intransit_tomorrow_arrival_num_norm = [intransit_tomorrow_arrival_num[i] / max(demand_est_scalar, 1e-6) - 1 
                                                       for i in range(len(intransit_tomorrow_arrival_num))]
                intransit_tomorrow_arrival_num_norm = self.clip(intransit_tomorrow_arrival_num_norm, self.state_clip_list[0], self.state_clip_list[1])

                intransit_tomorrow_arrival_prob_norm = [intransit_tomorrow_arrival_prob[i]
                                                        for i in range(len(intransit_tomorrow_arrival_prob))]
                intransit_tomorrow_arrival_prob_norm = self.clip(intransit_tomorrow_arrival_prob_norm, self.state_clip_list[0], self.state_clip_list[1])
                arr = [inventory_scalar_norm, demand_scalar_norm, demand_est_scalar_norm, demand_std_est_scalar_norm, intransit_all_scalar_norm] + (
                    intransit_qty_each_frag_norm + intransit_passed_days_norm + intransit_tomorrow_arrival_num_norm + intransit_tomorrow_arrival_prob_norm
                    )
                refined_arr = [inventory_scalar_norm, demand_scalar_norm, demand_est_scalar_norm, demand_std_est_scalar_norm, intransit_all_scalar_norm] + intransit_tomorrow_arrival_num_norm + intransit_tomorrow_arrival_prob_norm
            else:
                arr = [inventory_scalar, demand_scalar, demand_est_scalar, demand_std_est_scalar, intransit_all_scalar] + (
                    intransit_qty_each_frag + intransit_passed_days + intransit_tomorrow_arrival_num + intransit_tomorrow_arrival_prob)
                refined_arr = [inventory_scalar, demand_scalar, demand_est_scalar, demand_std_est_scalar, intransit_all_scalar] + intransit_tomorrow_arrival_num + intransit_tomorrow_arrival_prob
            full_obs[agent_name] = arr
            refined_obs[agent_name] = refined_arr

        # 根据actor_obs_type来选择obs
        ## full: 始终把自己agent的obs排在最前面，方便share_param运行
        if actor_obs_type == 'full':
            for agent_name in self.agents:
                full_arr = []
                full_arr += full_obs[agent_name]
                for agent_name_other in self.agents:
                    if agent_name_other != agent_name:
                        full_arr += full_obs[agent_name_other]
                actors_obs[agent_name] = full_arr
        else:
            for agent_name in self.agents:
                # partial: 每个agent的obs为个人obs和其他agent的refined_obs的拼接
                if actor_obs_type == 'partial':
                    refined_others_arr = []
                    for other_agent_name in self.agents:
                        if agent_name != other_agent_name:
                            refined_others_arr += refined_obs[other_agent_name]
                    actors_obs[agent_name] = refined_others_arr + full_obs[agent_name]
                elif actor_obs_type == 'self':
                    actors_obs[agent_name] = full_obs[agent_name]
                else:
                    raise Exception('wrong actor_obs_type')

        return actors_obs
    
    def est_next_arrival(self, agent_num):
        '''
        预估次日货物到达数目，以及有货物到达的概率
        '''
        est_arrival_num = 0
        est_arrival_prob = 0
        for intransit_item in self.intransit[agent_num][-1]:
            prob = self.get_dist_posterior_prob(self.LT_generators[agent_num][-1], intransit_item['passed_days'], intransit_item['passed_days'] + 1)
            est_arrival_num += prob * intransit_item['qty']
            est_arrival_prob = max(est_arrival_prob, prob)
        return est_arrival_num, est_arrival_prob
    
    def del_and_insert(self, arr, del_num, insert_num):
        '''
        二分查找在有序数组中删除一个数字并插入一个数字
        '''
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == del_num:
                # 找到要删除的数字，将其替换为插入的数字
                arr[mid] = insert_num
                # 从插入数字的位置开始向前遍历，直到找到一个比当前位置小的数或者到达数组的开头
                i = mid
                while i > 0 and arr[i - 1] > insert_num:
                    arr[i], arr[i - 1] = arr[i - 1], arr[i]
                    i -= 1
                # 从插入数字的位置开始向后遍历，直到找到一个比当前位置大的数或者到达数组的结尾
                i = mid
                while i < len(arr) - 1 and arr[i + 1] < insert_num:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    i += 1
                break
            elif arr[mid] < del_num:
                left = mid + 1
            else:
                right = mid - 1
        return arr

    def set_demand_statistics(self):
        '''
        统计需求quantile/mean, 以供critic network使用.
        以self.inventory_est_horizon为界，划分demand为短期与长期，并分别计算各自指标
        '''
        if self.step_num == 0:
            self.demand_dy_sorted_long = {agent_name: np.sort(demand[self.step_num + self.inventory_est_horizon: 
                                                                self.step_num + self.inventory_est_horizon + self.critic_looking_len].copy()) 
                                     for agent_name, demand in self.demand_dict.items()}
            self.demand_long_mean = {agent_name: np.mean(self.demand_dy_sorted_long[agent_name]) for agent_name in self.agents}
            self.demand_short = {agent_name: demand[self.step_num : self.step_num + self.inventory_est_horizon].copy()
                                 for agent_name, demand in self.demand_dict.items()}
            self.demand_short_mean = {agent_name: np.mean(demand) for agent_name, demand in self.demand_short.items()}

        else:
            # 每次统计时，将第一个demand删除，将对应demand插入  
            insert_index=self.critic_looking_len + self.step_num + self.inventory_est_horizon - 1

            self.demand_short = {agent_name: demand[self.step_num : self.step_num + self.inventory_est_horizon].copy()
                                 for agent_name, demand in self.demand_dict.items()}
            
            for agent_name in self.agents:
                del_num = self.demand_dict[agent_name][self.step_num + self.inventory_est_horizon - 1]
                insert_num = self.demand_dict[agent_name][insert_index] if insert_index < len(self.demand_dict[agent_name]) else self.demand_est_dict[agent_name]
                self.demand_dy_sorted_long[agent_name] = self.del_and_insert(self.demand_dy_sorted_long[agent_name], del_num, insert_num)
                self.demand_long_mean[agent_name] = self.demand_long_mean[agent_name] + (
                    (-del_num + insert_num) / self.critic_looking_len)

                del_num_LT = self.demand_dict[agent_name][self.step_num-1]
                insert_num_LT = self.demand_short[agent_name][-1]
                self.demand_short_mean[agent_name] = self.demand_short_mean[agent_name] + (
                    (-del_num_LT + insert_num_LT) / self.inventory_est_horizon) 

        # 计算分位数
        if 'long_quantile' in self.demand_info_for_critic:
            self.demand_q5 = {agent_name: self.demand_dy_sorted_long[agent_name][int(0.05*(len(self.demand_dy_sorted_long[agent_name])-1))]
                            for agent_name in self.agents}
            self.demand_q25 = {agent_name: self.demand_dy_sorted_long[agent_name][int(0.25*(len(self.demand_dy_sorted_long[agent_name])-1))]
                                for agent_name in self.agents}
            self.demand_q50 = {agent_name: self.demand_dy_sorted_long[agent_name][int(0.5*(len(self.demand_dy_sorted_long[agent_name])-1))]
                                for agent_name in self.agents}
            self.demand_q75 = {agent_name: self.demand_dy_sorted_long[agent_name][int(0.75*(len(self.demand_dy_sorted_long[agent_name])-1))]
                                for agent_name in self.agents}
            self.demand_q95 = {agent_name: self.demand_dy_sorted_long[agent_name][int(0.95*(len(self.demand_dy_sorted_long[agent_name])-1))]
                                for agent_name in self.agents}
            
    def get_arrival_qty_list(self, intransit, order):
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
                    LT_left = round(item['arrival_days'] - item['passed_days'])
                    for frag_idx_left in range(frag_idx+1, len(agent_intransit)):
                        LT_left += round(self.LT_dict[agent_name][self.step_num + LT_left][frag_idx_left])
                    qty_final = round(item['qty'] * (1 - self.shipping_loss_dict[agent_name][self.step_num + LT_left]))
                    arrival_day = max(LT_left -1, 0)
                    if arrival_day < self.inventory_est_horizon:
                        est_arrival_qty_list[agent_name][arrival_day] += qty_final

        return est_arrival_qty_list

    
    def get_est_inv(self):
        '''
        假设期间不transship + order, self.inventory_est_horizon时间内, 每天还剩多少库存(+)/缺货多少(-)。
        为避免信息冗余, 不包含当天的库存
        '''
        # 根据self.intransit, self.inventory, self.demand_dict update self.est_inventory
        est_inventory = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        # 记录所有其他agent平均每日缺货和过剩
        shortage_others = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        excess_others = {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        order = self.order
        # arrival_qty_list: {agent_name: [0] * self.inventory_est_horizon for agent_name in self.agents}
        arrival_qty_list = self.get_arrival_qty_list(self.intransit, order)
        for t_shift in range(self.inventory_est_horizon):
            # intransit, arrival_items = self.update_intransit(order, intransit, self.step_num + t_shift)
            order = {agent_name: 0 for agent_name in self.agents}
            for agent_name in self.agents:
                arrival_num = arrival_qty_list[agent_name][t_shift-1] if t_shift > 0 else 0
                inv = self.inventory[agent_name] if t_shift == 0 else est_inventory[agent_name][t_shift - 1] 
                inv = inv if self.backlog_tf else max(0, inv)
                inv += arrival_num
                demand = self.demand_dict[agent_name][self.step_num + t_shift]
                est_inventory[agent_name][t_shift] = inv - demand
        # 计算其他agent的平均每日缺货和过剩
        for agent_name in self.agents:
            other_agents = [a for a in self.agents if a != agent_name]
            for t in range(self.inventory_est_horizon):
                shortages = [max(-est_inventory[other][t], 0) for other in other_agents]
                shortage_others[agent_name][t] = sum(shortages) / len(other_agents)

                excesses = [max(est_inventory[other][t], 0) for other in other_agents]
                excess_others[agent_name][t] = sum(excesses) / len(other_agents)
        return est_inventory, shortage_others, excess_others

    # critic network 专属obs
    def get_step_obs_critic(self, state_normalize, critic_obs_type, intransit_info_for_actor_critic, demand_info_for_critic, 
                            inventory_est_for_critic_tf, inventory_est_others_for_critic_tf):
        actor_agent_obs = self.get_step_obs(state_normalize, critic_obs_type, intransit_info_for_actor_critic)
        # 更新demand statistics
        # 包含demand_q5, demand_q25, demand_q50, demand_q75, demand_q95
        # demand_dy_sorted_long, demand_long_mean, demand_short, demand_short_mean
        self.set_demand_statistics()
        # 估计未来库存
        est_inventory, est_shortage_others, est_excess_others = self.get_est_inv() if inventory_est_for_critic_tf else ({}, {}, {})

        full_obs = {agent_name: [] for agent_name in self.agents}
        refined_obs = {agent_name: [] for agent_name in self.agents}
        critic_agent_obs = {agent_name: [] for agent_name in self.agents}
        # collect the materials for critic
        for agent_name in self.agents:
            # long obs
            long_quantile_arr = [self.demand_q5[agent_name], self.demand_q25[agent_name], self.demand_q50[agent_name],
                                self.demand_q75[agent_name], self.demand_q95[agent_name]] if 'long_quantile' in demand_info_for_critic else []
            long_mean_arr = [self.demand_long_mean[agent_name]] if 'long_mean' in demand_info_for_critic else []
            long_arr = long_mean_arr + long_quantile_arr
            # short obs
            short_mean_arr = [self.demand_short_mean[agent_name]] if 'short_mean' in demand_info_for_critic else []
            short_all_arr = list(self.demand_short[agent_name]) if 'short_all' in demand_info_for_critic else []
            short_arr = short_mean_arr + short_all_arr
            # est_inv_obs
            est_inv_arr = est_inventory[agent_name] if inventory_est_for_critic_tf else []
            est_shortage_arr = est_shortage_others[agent_name] if inventory_est_for_critic_tf and inventory_est_others_for_critic_tf else []
            est_excess_arr = est_excess_others[agent_name] if inventory_est_for_critic_tf and inventory_est_others_for_critic_tf else []
            # normalization
            if state_normalize:
                demand_est_scalar = max(self.demand_est_dict[agent_name], 1)
                inventory_norm_helper = demand_est_scalar
                long_arr_norm = [item / max(demand_est_scalar, 1e-6) - 1 for item in long_arr]
                long_arr_norm = self.clip(long_arr_norm, self.state_clip_list[0], self.state_clip_list[1])
                short_arr_norm = [item / max(demand_est_scalar, 1e-6) - 1 for item in short_arr]
                short_arr_norm = self.clip(short_arr_norm, self.state_clip_list[0], self.state_clip_list[1])
                est_inv_arr_norm = [item / max(inventory_norm_helper, 1e-6) - 1 for item in est_inv_arr]
                est_inv_arr_norm = self.clip(est_inv_arr_norm, self.state_clip_list[0], self.state_clip_list[1])

                fillrate_assumed = 0.8
                est_shortage_arr_norm = [item / ((1 - fillrate_assumed) * max(inventory_norm_helper, 1e-6)) - 1 for item in est_shortage_arr]
                est_shortage_arr_norm = self.clip(est_shortage_arr_norm, self.state_clip_list[0], self.state_clip_list[1])
                est_excess_arr_norm = [item / max(inventory_norm_helper, 1e-6)  - 1 for item in est_excess_arr]
                est_excess_arr_norm = self.clip(est_excess_arr_norm, self.state_clip_list[0], self.state_clip_list[1])
                arr = long_arr_norm + short_arr_norm + est_inv_arr_norm + est_shortage_arr_norm + est_excess_arr_norm
                refined_arr = est_inv_arr_norm
            else:
                arr = long_arr + short_arr + est_inv_arr + est_shortage_arr + est_excess_arr
                refined_arr = est_inv_arr
            full_obs[agent_name] = arr
            refined_obs[agent_name] = refined_arr

        # 根据critic_obs_type来选择组合obs
        if critic_obs_type == 'full':
            for agent_name in self.agents:
                full_arr = []
                for agent_name_other in self.agents:
                    if agent_name_other != agent_name:
                        full_arr += full_obs[agent_name_other]
                critic_agent_obs[agent_name] = full_arr + full_obs[agent_name] + actor_agent_obs[agent_name]
        else:
            for agent_name in self.agents:
                if critic_obs_type == 'partial':
                    refined_others_arr = []
                    for other_agent_name in self.agents:
                        if agent_name != other_agent_name:
                            refined_others_arr += refined_obs[other_agent_name]
                    critic_agent_obs[agent_name] = refined_others_arr + full_obs[agent_name] + actor_agent_obs[agent_name]
                elif critic_obs_type == 'self':
                    critic_agent_obs[agent_name] = full_obs[agent_name] + actor_agent_obs[agent_name]
                else:
                    raise Exception('wrong critic_obs_type')
                
        return critic_agent_obs

    def get_processed_rewards(self, rewards):
        """
        考虑自私(利益共同体)因素的reward, 并更新reward_cum
        """
        processed_rewards = {agent_name: 0 for agent_name in self.agents}
        reward_mean = np.mean([rewards[agent_name] for agent_name in self.agents])
        for agent_name in self.agents:
            processed_rewards[agent_name] = self.self_interest * rewards[agent_name] + (1 - self.self_interest) * reward_mean
        self.reward = processed_rewards
        self.reward_cum = {agent_name: self.reward_cum[agent_name] + self.reward[agent_name] for agent_name in self.agents}
        self.reward_selfish_cum = {agent_name: self.reward_selfish_cum[agent_name] + rewards[agent_name] for agent_name in self.agents}
        
        # 计算mechanism_reward
        mechanism_reward = (self.mechanism_reward_weight_others * reward_mean + 
                            (1 - self.mechanism_reward_weight_others) * sum(self.shipping_cost_pure[agent_name] 
                                                                            for agent_name in self.agents))
        for agent_name in self.agents:
            self.mechanism_reward[agent_name] = mechanism_reward
        
        return processed_rewards
    
    def state_update(self, order_amounts, transship_intentions):
        '''
        更新状态, 计算reward
        '''
        cur_demand = {agent_name: self.demand_dict[agent_name][self.step_num]
                      for agent_name in self.agents}
        rewards = {agent_name: 0 for agent_name in self.agents}
        # 把transship_intentions转换为transship_actual
        ## 防止transship的量多于库存，如果不是pure_autonomy的话，transship_intentions就是是否愿意让central controller支配转运，故不需要这个限制
        if self.reactive_tf and self.transship_type == 'pure_autonomy':
            transship_intentions = {agent_name: max(- max(self.inventory[agent_name], 0), transship_intentions[agent_name])
                                    for agent_name in self.agents}
        else:
            transship_intentions = {agent_name: max(- max(self.inventory[agent_name] - cur_demand[agent_name], 0), transship_intentions[agent_name])
                                    for agent_name in self.agents}
        self.transship_matrix, payment_arr = self.dist_mechanism.get_mechanism_result(self.unwrap(transship_intentions))
        # 表示从transship中获得了多少货物
        self.transship_actual = self.wrap(self.transship_matrix.sum(axis=1))
        # 更新后的在途情况， 以及当天到货的货物
        self.intransit, arrival_items = self.update_intransit(self.order, self.intransit, self.step_num)


        # 更新状态 + 计算reward 
        for agent_idx, agent_name in enumerate(self.agents):
            self.ordering_times[agent_name] += (1 if self.order[agent_name] > 0 else 0)
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
                fill_rate_assumed = 0.8
                norm_drift = cur_demand[agent_name] * (- fill_rate_assumed * self.H[agent_name] 
                                                       + self.P[agent_name] 
                                                       - self.C[agent_name] 
                                                       - (1 - fill_rate_assumed) * self.B[agent_name])
            else:
                norm_drift = 0

            # 是否对reward通过demand_est进行scale, scale之后reward被扭曲了，效果一般
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
    
    def consume_inventory(self, agent_name, demand_to_be_fulfilled):
        '''
        消耗库存，更新inventory_batch里的库存成本信息
        '''
        inv_cost = 0
        remove_items_list = []
        # 记录消耗库存后的成本信息
        for inventory_item in self.inventory_batch[agent_name]:
            if demand_to_be_fulfilled == 0:
                break
            demand_fulfilled_by_this_item = min(inventory_item['qty'], demand_to_be_fulfilled)
            inventory_item['qty'] = max(0, inventory_item['qty'] - demand_fulfilled_by_this_item)
            if inventory_item['qty'] == 0:
                remove_items_list.append(inventory_item)
            demand_to_be_fulfilled = demand_to_be_fulfilled - demand_fulfilled_by_this_item
            inv_cost += demand_fulfilled_by_this_item * inventory_item['avg_cost']
        
        for remove_item in remove_items_list:
            self.inventory_batch[agent_name].remove(remove_item)

        return inv_cost

    def seed(self, seed):
        self._seed = seed
        random.seed(seed)   # Python的随机性
        np.random.seed(seed)   # numpy的随机性

    def get_avail_actions(self):
        # TODO: 后面有需要再改，因为这个项目好像不太适配multi-discrete + available actions
        if self.action_type == 'discrete':
            return np.ones((self.n_agents, self.action_space[0].n))
        elif self.action_type == 'multi_discrete':
            return None
    
    def close(self):
        pass

# ----------------- UTILS -----------------
    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l
    
    def repeat(self, a):
        if isinstance(a, (list, np.ndarray)):
            return [a.copy() for _ in range(self.n_agents)]
        elif isinstance(a, dict):
            return [{k: v.copy() if hasattr(v, 'copy') else v for k, v in a.items()} for _ in range(self.n_agents)]
        else:
            return [a for _ in range(self.n_agents)]
    
    def scalar_to_dict(self, scalar):
        if isinstance(scalar, dict) and ('agent0' in scalar):
            return scalar
        return self.wrap(self.repeat(scalar))
    
    def get_dist_mean_std(self, dist_generator):
        gen = None
        if dist_generator['gen_func'] == 'merton':
            gen = generator.merton(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'poisson':
            gen = generator.poisson(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'normal':
            gen = generator.normal(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'uniform':
            gen = generator.uniform(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'kim':
            gen = generator.kim_dist(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'constant':
            gen = generator.constant_dist(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'kim_merton':
            gen = generator.kim_merton(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'shanshu':
            gen = generator.shanshu(length=self.episode_length * 2, **dist_generator['gen_args'])
        else:
            raise Exception('wrong gen_func')
        d_mean, d_std = gen.get_theoretical_mean_std()
        return d_mean, d_std
    
    def get_dist_posterior_prob(self, dist_generator, condition_num, value):
        gen = None
        if dist_generator['gen_func'] == 'merton':
            gen = generator.merton(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'poisson':
            gen = generator.poisson(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'normal':
            gen = generator.normal(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'uniform':
            gen = generator.uniform(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'kim':
            gen = generator.kim_dist(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'constant':
            gen = generator.constant_dist(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'kim_merton':
            gen = generator.kim_merton(length=self.episode_length * 2, **dist_generator['gen_args'])
        elif dist_generator['gen_func'] == 'shanshu':
            gen = generator.shanshu(length=self.episode_length * 2, **dist_generator['gen_args'])
        else:
            raise Exception('wrong gen_func')
        return gen.get_posterior_prob(condition_num, value)

    def clip(self, arr, lb, ub):
        if not isinstance(arr, list) and (not isinstance(arr, np.ndarray) or arr.ndim == 0):
            return max(min(arr, ub), lb)
        return [max(min(item, ub), lb) for item in arr]
    
    def generate_eval_test_dateset(self, usage, data_type, generators_dict, round_int_tf, num_eval=30):
        if usage == 'test':
            data_dir = TEST_DIR.format(self.dataset_name)
        elif usage == 'eval':
            data_dir = EVAL_DIR.format(self.dataset_name)
        else:
            raise Exception('no such usage except test and eval')

        for i in range(num_eval):
            generated_data_dict = self.get_data_from_generators_dict(generators_dict, round_int_tf)
            for agent_name in self.agents:
                agent_data = generated_data_dict[agent_name]
                agent_data_dir = data_dir + str(agent_name)
                if not os.path.exists(agent_data_dir):
                    os.makedirs(agent_data_dir)
                file_path = agent_data_dir+'/{}_{}.txt'.format(data_type,i)
                agent_data = np.array(agent_data)
                # 保留浮点数
                if round_int_tf:
                    np.savetxt(file_path, agent_data, fmt='%d', delimiter=' ')
                else:
                    np.savetxt(file_path, agent_data, fmt='%f', delimiter=' ')
    
    def generate_eval_test_wrapper(self):
        # 防止每次生成的数据不一样
        self.seed(0)
        for usage in ["test","eval"]:
            self.generate_eval_test_dateset(usage, "demand", self.demand_generators, True)
            self.generate_eval_test_dateset(usage, "LT", self.LT_generators, True)
            self.generate_eval_test_dateset(usage, "shippingloss", self.shipping_loss_generators, False)

if __name__ == '__main__':
    args = {'action_type': "heu_discrete", 
            'stationary_tf': False,
            'k': 0.5,
            'payment_type': 'product_first',
            'transship_type': 'force_to_transship',
            'product_allocation_method': {'allocate_func':'baseline', 'allocate_args':{'ratio_tf': True}}, 
            'reactive_tf': False,
            }
    # {'allocate_func':'anup', 'allocate_args':{'ratio_tf': True}}
    # {'allocate_func':'homo_distance', 'allocate_args':{'ratio_tf': True}}
    # {'allocate_func':'baseline', 'allocate_args':{'ratio_tf': True}}
    env = MultiLtTransshipEnv(args)
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

