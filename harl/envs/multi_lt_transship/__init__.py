import numpy as np
import os
DISTANCE_DIR = 'harl/envs/multi_lt_transship/distance/'
TEST_DIR = 'harl/envs/multi_lt_transship/test_data/{}/'
EVAL_DIR = 'harl/envs/multi_lt_transship/eval_data/{}/'
TRAIN_DIR = 'harl/envs/multi_lt_transship/train_data/{}/'
SHANSHU_GEN_SOURCE = 'harl/envs/multi_lt_transship/shanshu_gen_source/{}/{}.csv'

# 按顺序取DISTANCE_DIR下的文件名，读取文件，转换为int类型，存入DISTANCE列表中
DISTANCE_LIST = [np.loadtxt(DISTANCE_DIR + file, delimiter=',')
            for file in sorted(os.listdir(DISTANCE_DIR))]


# 这里的key 'agent{}'.format(i) 代表的是agent的id,只是为了方便理解，顺序需要严格按照0，1，2，3...来排列
ENV_ARGS_DEFAULT = {
    'num_agents': 3,
    # 以下是gen_func: [*gen_args]形式的说明
    # merton: [std, max_num], poisson: [mean, max_num], 
    # normal: [mean, std, max_num], uniform: [min_num, max_num], 
    # kim: [meanMin, meanMax, stdMin, stdMax, max_num]
    # constant: [value]
    # kim_merton: [mean_std, mean_max, std_max, std_std, max_num]
    # 声明每个agent的LT生成器，若有多个，则表示有多段LT
    # 'LT_generators': {'agent0':[{'gen_func':'uniform', 'gen_args':{'min_num':1, 'max_num':3}}, 
    #                             {'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':4}}, 
    #                             {'gen_func':'constant', 'gen_args':{'value': 1}}], 
    #                   'agent1':[{'gen_func':'uniform', 'gen_args':{'min_num':1, 'max_num':3}}, 
    #                             {'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':4}}, 
    #                             {'gen_func':'constant', 'gen_args':{'value': 1}}],
    #                   'agent2':[{'gen_func':'uniform', 'gen_args':{'min_num':1, 'max_num':3}}, 
    #                             {'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':4}}, 
    #                             {'gen_func':'constant', 'gen_args':{'value': 1}}]},
    'LT_generators': [{'gen_func':'uniform', 'gen_args':{'min_num':1, 'max_num':3}}, 
                        {'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':4}}, 
                        {'gen_func':'constant', 'gen_args':{'value': 1}}],

    # 声明每个agent的demand生成器
    # 'demand_generators': {'agent0':{'gen_func':'merton', 'gen_args':{'std':4, 'max_num':20}},
    #                       'agent1':{'gen_func':'merton', 'gen_args':{'std':4, 'max_num':20}}, 
    #                       'agent2':{'gen_func':'merton', 'gen_args':{'std':4, 'max_num':20}}},
    'demand_generators': {'gen_func':'kim_merton', 'gen_args':{'mean_std': 4, 'mean_max': 30, 'std_max': 5, 'std_std': 1, 'max_num': 50}},
    # 声明每个agent的shipping loss生成器，若不考虑shipping loss，则{'gen_func':'constant', 'gen_args':{'value':0}}
    # 'shipping_loss_generators': {'agent0':{'gen_func':'uniform', 'gen_args':{'min_num':0, 'max_num':0.2}}, 
    #                              'agent1':{'gen_func':'uniform', 'gen_args':{'min_num':0, 'max_num':0.2}},
    #                              'agent2':{'gen_func':'uniform', 'gen_args':{'min_num':0, 'max_num':0.2}}},
    'shipping_loss_generators': {'gen_func':'uniform', 'gen_args':{'min_num':0, 'max_num':0.2}},

    # backlog 还是 lost sales
    'backlog_tf': True, 

    # cost parameters，若为标量，则所有agent的cost参数相同
    # ## holding cost,
    # 'H': {'agent0':0.1, 'agent1':0.1, 'agent2':0.1},
    # ## backlog/penalty cost
    # 'B': {'agent0':0.2, 'agent1':0.2, 'agent2':0.2},
    # ## selling price
    # 'P': {'agent0':0, 'agent1':0, 'agent2':0},
    # ## ordering cost
    # 'C': {'agent0':0, 'agent1':0, 'agent2':0},
    # ## fixed ordering cost
    # 'K': {'agent0':0, 'agent1':0, 'agent2':0}, 
    'H': 0.1,
    'B': 0.2,
    'P': 0,
    'C': 0,
    'K': 0,
    ## shipping cost per distance
    'shipping_cost_per_distance': 5e-5, 

    # initialization
    # 'initial_inventory': {'agent0':20, 'agent1':20, 'agent2':20}, 
    'initial_inventory': 20,
    ## initial in-transit inventory, 表示每个agent不同阶段LT的初始在途库存
    # 'initial_in_transit': {'agent0':[[{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},]],
    #                        'agent1':[[{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},]],
    #                        'agent2':[[{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},], 
    #                                  [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},]]}, # 在途具体情况 {agent_name:[[第一段所有在途], [第二段所有在途], ...]}
    'initial_in_transit': [[{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},],
                            [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},],
                            [{'qty':10, 'passed_days':0, 'arrival_days': 2, 'avg_cost': 0},]],

    
    # distance matrix index, 会自动取出对应的distance matrix
    'distance_matrix_index': 1,

    # transshipment mechanism
    'reactive_tf': True, # 是否允许reactive transship
    'transship_type': 'pure_autonomy', # 转运类型 'pure_autonomy': 自己决策transship的量, 'half_autonomy': 只决策是否参与transship, 'no': 不参与transship, 'force_to_transship': 强制参与转运
    ## 以下是product_allocation_method的allocate_func对应的allocate_args的说明
    ## homo_distance: ratio_tf: 按比例分配/砍一刀
    ## hier: how(even/ratio), threshold(距离)
    'product_allocation_method': {'allocate_func':'least_distance'}, # 'homo_distance', 'least_distance', 'hier', 'anup', 'no', 'mechanism_agent', 'baseline'
    'revenue_allocation_method': 'constant', # 'constant' or 'ratio'
    'constant_transshipment_price': 0.15, # only used when revenue_allocation_method is 'constant'

    # reward calculation
    ## 表示reward计算的自私程度，0表示完全合作，1表示完全自私
    'self_interest': 1.0, # self-interest parameter
    ## 结算订货费用的方式
    'payment_type': 'selling_first', # 先款后货: 'pay_first'，先货后款: 'product_first'，先销后款: 'selling_first'
    # reward的计算方式
    'reward_type': 'profit', # 'profit', 'norm_profit'
    'reward_scale_tf': False, # 是否对reward进行scale

    # observation
    'state_normalize': True,            # 是否把state normalize 到均值为0，但无法保证服从N(0, 1)
    'state_clip_list': [-5, 5],         # 强制控制norm后的state在[-5, 5]范围，防止出现过于破坏分布的数据点，从而阻碍训练
    # 'intransit_info_for_actor_critic': ['qty_each', 'tomorrow', 'passed_days'], # tomorrow: 基于贝叶斯预测次日抵达库存, passed_days: 每阶段货物对应已经经过的最大天数
    'intransit_info_for_actor_critic': ['tomorrow'], # tomorrow: 基于贝叶斯预测次日抵达库存, passed_days: 每阶段货物对应已经经过的最大天数
    ## actor observation
    'actor_obs_type': 'full', # 'full', 'partial', 'self'
    ## critic observation
    'critic_obs_type': 'full', # 'full', 'partial', 'self'
    'demand_info_for_critic': ['long_quantile', 'long_mean', 'short_all', 'short_mean'], # choices: 'long_quantile': 远期需求的分位数, 
    # 'long_mean': 远期需求的均值, 'short_all': 近期需求的所有值, 'short_mean': 近期需求的均值
    'inventory_est_for_critic_tf': True, # True/False 是否将inventory估计值作为critic的输入
    'inventory_est_others_for_critic_tf': True, # True/False, 是否把其他agent的库存多寡作为critic的输入
    'inventory_est_horizon': 4, # inventory估计值的时间跨度, 同时也是区分长短期的时间跨度

    # ordering frequency
    # 'MOQ': {'agent0':20, 'agent1':20, 'agent2':20}, # minimum order quantity
    'MOQ': 20, # minimum order quantity
    # 'average_frequency': {'agent0':1, 'agent1':1, 'agent2':1}, # average order frequency
    'average_frequency': 1, # average order frequency
    
    # action space
    'action_type': 'multi_discrete', # 'discrete', 'continuous' or 'multi_discrete'
    'order_step_size': 0.1, # 非continuous的order的action space的步长
    'transship_step_size': 0.05, # 非continuous的transship的action space的步长
    # 'order_ub': {'agent0':3, 'agent1':3, 'agent2':3}, # 最高几倍的MOQ(或0.8*(average_frequency-1)*demand_mean)\
    'order_ub': 3, # 最高几倍的MOQ(或0.8*(average_frequency-1)*demand_mean)
    # 'transship_lb': {'agent0':-1, 'agent1':-1, 'agent2':-1}, # 最高转运出去是几倍的demand_mean
    # 'transship_ub': {'agent0':2, 'agent1':2, 'agent2':2}, # 最高转运进来是几倍的demand_mean
    'transship_lb': -1, # 最高转运出去是几倍的库存
    'transship_ub': 1.5, # 最高转运进来是几倍的demand_mean
    'estimate_demand_method': 'ema',  # 'mean', 'rolling_mean', 'ema'
    'rolling_window': 3, # rolling window size, 用于估计demand_mean
    'alpha': 0.8, # ema的alpha值

    # eval/test dataset name
    'dataset_name': 'demand_kim_merton',
    'scenario': 'default', 
    
    'gamma': 0.99,   # 和algo的参数里设置一致
    'episode_length': 200,   # 和algo的参数里设置一致

    # only for mechanism_agent
    "actor_obs_type_mechanism": 'intentions_only', # 'intentions_only', 'same_as_critic'
    "critic_obs_type_mechanism": 'full', # 'intentions_only', 'full'
}



