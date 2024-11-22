import json
import os

import numpy as np
from scipy import stats

from harl.envs.multi_lt_transship.transship_multi_lt import MultiLtTransshipEnv
from harl.envs.multi_lt_transship.baseline import Baseline

def search_save(default_args, result_path):
    # 以num_agents为文件名txt，写入default_args
    with open(result_path, 'w') as f:
        f.write(str(default_args))
        f.write('\n')
        
    eval_episodes = 20
    episode_length = 200

    # 第一层搜索
    eval_episodes_raw = 5
    k_range = np.arange(-2, 10, 0.3)
    reward_cum_dict = {k: np.zeros(default_args['num_agents']) for k in k_range}
    reward_cum_scalar_dict = {k: -1e7 for k in k_range}
    best_reward = float('-inf')
    consecutive_worse = 0
    for k in k_range:
        args = default_args.copy()
        args['k'] = k
        current_reward = 0
        for eval_idx in range(eval_episodes_raw):
            env = MultiLtTransshipEnv(args)
            baseline = Baseline(args)
            obs = env.reset_for_dataset('eval', eval_idx)[0]
            while(True):  
                actions = baseline.get_action(env, obs) 
                obs, _, rewards, _, _, _ = env.step(actions)
                reward_cum_dict[k] += np.squeeze(rewards)/eval_episodes_raw
                current_reward += np.sum(rewards)/eval_episodes_raw/default_args['num_agents']
                if env.step_num >= episode_length:
                    break
        reward_cum_scalar_dict[k] = current_reward
        
        if current_reward > best_reward:
            best_reward = current_reward
            consecutive_worse = 0
        else:
            consecutive_worse += 1
            if consecutive_worse >= 2:
                break
    
    print(reward_cum_dict)
    with open(result_path, 'a') as f:
        f.write(str(reward_cum_dict))
        f.write('\n')

    # 第二层搜索，选择其中最好的，在这范围精搜
    best_k = max(reward_cum_scalar_dict, key=reward_cum_scalar_dict.get)
    k_range = np.arange(best_k - 0.55, best_k + 0.55, 0.05)
    reward_cum_dict_2 = {k: np.zeros(default_args['num_agents']) for k in k_range}
    reward_cum_scalar_dict_2 = {k: 0 for k in k_range}
    for k in k_range:
        args = default_args.copy()
        args['k'] = k
        for eval_idx in range(eval_episodes):
            env = MultiLtTransshipEnv(args)
            baseline = Baseline(args)
            obs = env.reset_for_dataset('eval', eval_idx)[0]
            while(True):  
                actions = baseline.get_action(env, obs) 
                obs, _, rewards, _, _, _ = env.step(actions)
                reward_cum_dict_2[k] += np.squeeze(rewards)/eval_episodes
                reward_cum_scalar_dict_2[k] += np.sum(rewards)/eval_episodes/default_args['num_agents']
                if env.step_num >= episode_length:
                    break
    with open(result_path, 'a') as f:
        f.write(str(reward_cum_dict_2))
        f.write('\n')

    # print最优的k
    best_k = max(reward_cum_scalar_dict_2, key=reward_cum_scalar_dict_2.get)
    print(best_k)
    print(reward_cum_dict_2[best_k])

    # 用找到最优的k再跑一遍test
    reward_test = np.zeros(default_args['num_agents'])
    for eval_idx in range(eval_episodes):
        args = default_args.copy()
        args['k'] = best_k
        env = MultiLtTransshipEnv(args)
        baseline = Baseline(args)
        obs = env.reset_for_dataset('test', eval_idx)[0]
        while(True):  
            actions = baseline.get_action(env, obs) 
            obs, _, rewards, _, _, _ = env.step(actions)
            reward_test += np.squeeze(rewards)/eval_episodes
            if env.step_num >= episode_length:
                break
    print(reward_test)
    with open(result_path, 'a') as f:
        f.write(str(best_k))
        f.write(str(reward_test))
        f.write(str(np.average(reward_test)))
        f.write('\n')

def cal_quantile_k_result_save(default_args, result_path):
    with open(result_path, 'w') as f:
        f.write(str(default_args))
        f.write('\n')
        
    eval_episodes = 20
    episode_length = 200

    # 定义环境
    env = MultiLtTransshipEnv(default_args)

    # get b/h+b quantile(standard norm)
    args = default_args.copy()
    h, b = env.H['agent0'], env.B['agent0']
    k = stats.norm.ppf(b/(b+h))
    args['k'] = k

    reward_cum_val = np.zeros(default_args['num_agents'])
    for eval_idx in range(eval_episodes):
        # 初始化策略/环境
        baseline = Baseline(args)
        obs = env.reset_for_dataset('eval', eval_idx)[0]
        while(True):  
            actions = baseline.get_action(env, obs) 
            obs, _, rewards, _, _, _ = env.step(actions)
            reward_cum_val += np.squeeze(rewards)/eval_episodes
            if env.step_num >= episode_length:
                break
    print(reward_cum_val)
    with open(result_path, 'a') as f:
        f.write(str(reward_cum_val))
        f.write('\n')
        f.write(str(np.average(reward_cum_val)))
        f.write('\n')

    # 用找到最优的k再跑一遍test
    reward_test = np.zeros(default_args['num_agents'])
    for eval_idx in range(eval_episodes):
        args = default_args.copy()
        args['k'] = k
        baseline = Baseline(args)
        obs = env.reset_for_dataset('test', eval_idx)[0]
        while(True):  
            actions = baseline.get_action(env, obs) 
            obs, _, rewards, _, _, _ = env.step(actions)
            reward_test += np.squeeze(rewards)/eval_episodes
            if env.step_num >= episode_length:
                break
    print(reward_test)
    with open(result_path, 'a') as f:
        f.write(str(k))
        f.write(str(reward_test))
        f.write(str(np.average(reward_test)))
        f.write('\n')

if __name__ == '__main__':
    # NOTICE: 模式选择
    # 是否reactive
    reactive_tf = False
    # 是否根据b/h+b的quantile来选择k
    use_quantile_k = False
    # 是否认为需求是stationary
    consider_stationary = False
    # num_agents
    num_agents = 3
    # allocate_func
    allocate_func = 'anup'  # 'baseline'

    # 保存结果的文件夹,若不存在则创建
    result_dir = 'harl/envs/multi_lt_transship/caogao_result_{}{}{}{}'.format(
        'quantile_k' if use_quantile_k else 'search_k',
        '_sta' if consider_stationary else '',
        '' if reactive_tf else '_proactive',
        '_'+allocate_func if allocate_func != 'baseline' else ''
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 默认参数，一般不用动
    default_args = {'action_type': "heu_discrete", 
                    'stationary_tf': consider_stationary,
                    'k': 0.629,
                    'reactive_tf': reactive_tf,
                    'payment_type': 'selling_first',
                    'num_agents': num_agents, 
                    'dataset_name': 'demand_kim_merton',
                    'transship_type': 'force_to_transship',
                    'product_allocation_method': {'allocate_func':allocate_func, 'allocate_args':{'ratio_tf': True}}
                    }
    # NOTICE: 先跑默认的
    result_path = result_dir + '/{}_{}.txt'.format('default', default_args['num_agents'])
    if use_quantile_k:
        cal_quantile_k_result_save(default_args, result_path)
    else:
        search_save(default_args, result_path)

    # # NOTICE: 单次计算
    # update_args_path = 'tuned_configs/multi_lt_transship/env_test/lt/default1.json'
    # with open(update_args_path, 'r') as f:
    #     args = json.load(f)
    #     update_args = args['env_args']
    #     exp_name = args['main_args']['exp_name']
    # default_args.update(update_args)
    # result_path = result_dir + '/{}_{}.txt'.format(exp_name, default_args['num_agents'])
    # if use_quantile_k:
    #     cal_quantile_k_result_save(default_args, result_path)
    # else:
    #     search_save(default_args, result_path)


    # NOTICE: 批量计算
    update_args_path_dir_list = ['tuned_configs/multi_lt_transship/env_test/backlog_price',
                                    'tuned_configs/multi_lt_transship/env_test/fixed_ordering_cost',
                                    'tuned_configs/multi_lt_transship/env_test/holding_cost',
                                    'tuned_configs/multi_lt_transship/env_test/order_freq',
                                    'tuned_configs/multi_lt_transship/env_test/shipping_cost_per_unit',
                                    'tuned_configs/multi_lt_transship/env_test/pure_agent_num',
                                    'tuned_configs/multi_lt_transship/env_test/lt',
                                    'tuned_configs/multi_lt_transship/env_test/demand',
                                    'tuned_configs/multi_lt_transship/env_test/moq',
                                    'tuned_configs/multi_lt_transship_mechanism/env_test/shanshu_demand']
    # update_args_path_dir_list = ['tuned_configs/multi_lt_transship_mechanism/env_test/shanshu_demand']
    
    for update_args_path_dir in update_args_path_dir_list:
        for root, _, files in os.walk(update_args_path_dir):
            for file in files:
                if file.endswith(".json"):
                    update_args_path = os.path.join(root, file).replace("\\", "/")
                    with open(update_args_path, 'r') as f:
                        args = json.load(f)
                        update_args = args['env_args']
                        exp_name = args['main_args']['exp_name']
                    used_args = default_args.copy()
                    used_args.update(update_args)
                    result_path = result_dir + '/{}_{}.txt'.format(exp_name, used_args['num_agents'])
                    if use_quantile_k:
                        cal_quantile_k_result_save(used_args, result_path)
                    else:
                        search_save(used_args, result_path)                                 
    