import numpy as np
import math 
import torch 
import coptpy as cp
from coptpy import COPT
import itertools

S_I = 10
S_O = 10

class Baseline:
    def __init__(self,args):
        # heuristic parameters
        # k1,k2 = -2.4, -0.8
        self.k = args.get('k', 0)
        # self.k2 = args.get('k2', 0)
        self.pre_ema = [0] * 30 # for each agent
        self.pre_ema_d_sqr = [0] * 30
        self.stationary_tf = args['stationary_tf']
        self.args = args
        self.mini_pooling = {"flag": False, "threshold": 200, "how": "even"}

    # @staticmethod
    # def get_order_revenue_list(env,agent_name,demand_pred):
    #     revenue_list = []
    #     revenue_list.append(-demand_pred * env.B[agent_name])
    #     for k in range(1, 30):
    #         revenue_k = (k * demand_pred * env.P[agent_name] - sum(
    #             range(k)) * demand_pred * env.H[agent_name] - env.shipping_cost_per_distance) / k
    #         revenue_list.append(revenue_k)
    #     return revenue_list
    
    @staticmethod
    def calculate_mu_and_d_sqr(alpha, t, ema, ema_d_sqr, pre_ema, pre_ema_d_sqr, lt_sqr):
        tmp_ema = ema
        tmp_pre_ema = pre_ema
        # 未来t天的需求总值均值
        mu = ema
        # 未来t天的需求总值方差
        d_sqr = ema_d_sqr
        # rolling地去计算未来t天每天的均值和方差
        for i in range(int(t - 1)):
            tmp1 = tmp_ema
            tmp_ema = alpha * tmp_ema + (1 - alpha) * tmp_pre_ema
            tmp_pre_ema = tmp1
            mu += tmp_ema
            d_sqr += ema_d_sqr
        # 增加对lt随机性的考量
        d_sqr += lt_sqr * tmp_ema **2
        return mu, d_sqr
    
    @staticmethod
    def get_hist_demand(env):
        return [env.demand_dict[agent_name][:env.step_num] for agent_name in (env.agents)]

    def get_order_action(self, env, obs):
        self.hist_demand = self.get_hist_demand(env)

        order_amounts = []
        for (i, agent_name) in enumerate(env.agents):
            inv, demand, orders_num = (env.inventory[agent_name], 
                                   env.demand_dict[agent_name][env.step_num-1], 
                                   sum(dc['qty'] for ldc in env.intransit[agent_name] for dc in ldc))
            total_inv = inv + orders_num
            # 计算总的lead time mean/sqr
            lead_time_mean, lead_time_sqr = 0, 0
            for lt_gen in env.LT_generators[agent_name]:
                gen_mean, gen_std = env.get_dist_mean_std(lt_gen)
                lead_time_mean += gen_mean
                lead_time_sqr += gen_std**2

            # s对应的时间窗口，r + lt
            t_plus_l = lead_time_mean + 1

            # 如果补货策略stationary，则直接用分布的均值和方差
            if self.stationary_tf:
                demand_mean_est, demand_sqr_est = env.get_dist_mean_std(env.demand_generators[agent_name])
                # s计算
                mu_t_plus_l = demand_mean_est * t_plus_l
                sig_t_plus_l = math.sqrt(demand_sqr_est * t_plus_l + lead_time_sqr * demand_mean_est**2)
                s = mu_t_plus_l + self.k * sig_t_plus_l

            # 否则用EMA来估计均值和方差
            else:
                window = lead_time_mean 
                alpha = env.alpha
                if len(self.hist_demand) == 0:
                    demand_mean_est = demand
                    demand_sqr_est = 0
                else:
                    demand_mean_est = alpha * demand + (1 - alpha) * self.pre_ema[i]
                    demand_sqr_est = alpha * (demand - demand_mean_est)**2 + \
                    (1 - alpha) * (self.pre_ema_d_sqr[i])
                # 基于ema估计未来t+lt天的需求均值和方差
                mu_t_plus_l, d_sqr_t_plus_l = self.calculate_mu_and_d_sqr(
                    alpha, t_plus_l, demand_mean_est, demand_sqr_est, self.pre_ema[i], self.pre_ema_d_sqr[i], lead_time_sqr)
                sig_t_plus_l = math.sqrt(d_sqr_t_plus_l)

                s = mu_t_plus_l + self.k * sig_t_plus_l

                # 更新EMA需要维护的list
                self.pre_ema[i] = demand_mean_est
                self.pre_ema_d_sqr[i] = demand_sqr_est

            if total_inv < s:
                order_heu = round(max(env.MOQ[agent_name], s - total_inv))
            else:
                order_heu = 0
            order_amounts.append(order_heu)
        return order_amounts
    
    def get_action(self, env, obs):
        order_amounts = self.get_order_action(env, obs)
        # transship_amounts = self.get_transship_actions(env, order_amounts)
        # 目前已把transship优化分配放在baseline_mechanism.py中, 故这里无需再次计算transship
        transship_amounts = [0 for _ in range(env.n_agents)]
        actions = [k for k in zip(order_amounts, transship_amounts)]
        return actions
    
    def est_future_arrival(self, env, agent_name, order_amount, look_ahead_days):
        '''
        预估未来look_ahead_days天的到货情况
        TODO: 这里的到货时间是按照LT的均值来算的，后续可以考虑加入随机性,比如把货物贝叶斯平均到每一天上
        '''
        future_arrival = [0 for _ in range(int(look_ahead_days))]
        lt_mean_list = [env.get_dist_mean_std(lt_gen)[0] for lt_gen in env.LT_generators[agent_name]]
        residual_days_today_order = int(sum(lt_mean_list) - 0)
        if residual_days_today_order < look_ahead_days:
            future_arrival[residual_days_today_order] += order_amount
        for (frag_idx, intransit_list) in enumerate(env.intransit[agent_name]):
            for intransit_dict in intransit_list:
                residual_days = int(sum(lt_mean_list[frag_idx:]) - intransit_dict['passed_days'])
                if residual_days < look_ahead_days:
                    future_arrival[residual_days] += intransit_dict['qty']
        return future_arrival
        