import numpy as np
from scipy.optimize import linprog
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class BaselineDistribution(BaseMechanism):
    '''
    该类用于复现baseline的分配计划，就是直接用线性规划求解预估环境下的最优转运量
    '''
    def __init__(self, distance_matrix, allocate_args=None, others_info = None):
        super().__init__(distance_matrix, allocate_args, others_info)
        self.pre_ema = [0] * self.env.n_agents  

    def predict_demand(self, agent_name):
        i = self.env.agents.index(agent_name)
        lead_time_mean = sum(self.env.get_dist_mean_std(lt_gen)[0] for lt_gen in self.env.LT_generators[agent_name])
        window = lead_time_mean
        alpha = self.env.alpha
        
        if self.env.step_num == 0:
            return self.env.demand_dict[agent_name][0], self.env.demand_dict[agent_name][0]  # 使用第一个需求作为初始预测
        
        demand = self.env.demand_dict[agent_name][self.env.step_num]
        demand_mean_est = alpha * demand + (1 - alpha) * self.pre_ema[i]
        self.pre_ema[i] = demand_mean_est
        return demand, demand_mean_est
    
    def get_transship_matrix(self, transship_intentions):
        '''
        transship_intentions: list, 每个agent是否愿意参与transship
        '''
        # 参与transshipment的agent索引
        active_agents = [i for i in range(len(transship_intentions)) if transship_intentions[i] == 1]

        # 如果没有agent愿意参与transshipment，那么就不需要转运
        if len(active_agents) <= 1:
            return np.zeros((self.agent_num, self.agent_num))
        # 初始库存
        order_amounts = self.env.unwrap(self.env.order)
        next_start_inventory = [self.env.inventory[agent_name] for agent_name in self.env.agents]
        next_start_inventory = [next_start_inventory[i] for i in active_agents]

        # 预估的到货list
        arrival_list = []
        look_ahead_days = 10000

        # 确定look ahead days的范围
        for (i, agent_name) in enumerate(self.env.agents):
            lt_agent = 0
            for lt_gen in self.env.LT_generators[agent_name]:
                gen_mean, _ = self.env.get_dist_mean_std(lt_gen)
                lt_agent += gen_mean
            look_ahead_days = int(min(look_ahead_days, lt_agent))

        # 假设未来的需求就是这个了
        next_demand = []
        for agent_name in self.env.agents:
            demand, demand_mean_est = self.predict_demand(agent_name)
            next_demand.append([demand] + [demand_mean_est] * (look_ahead_days - 1))
        next_demand = [next_demand[i] for i in active_agents]
        for (i, agent_name) in enumerate(self.env.agents):
            future_arrival = self.est_future_arrival(agent_name, order_amounts[i], look_ahead_days)
            arrival_list.append(future_arrival)
        arrival_list = [arrival_list[i] for i in active_agents]

        # 创建优化变量
        I = active_agents  # 只考虑愿意参与的agent
        J = active_agents  # 只考虑愿意参与的agent
        T = list(range(look_ahead_days))
        
        N = len(I)  # Number of agents
        T_len = len(T)  # Number of time periods
        
        # Variable indices
        tran_indices = {}
        index = 0
        for t in range(T_len):
            for i in range(N):
                for j in range(N):
                    tran_indices[(i, j, t)] = index
                    index += 1

        s_inv_indices = {}
        for t in range(T_len):
            for i in range(N):
                s_inv_indices[(i, t)] = index
                index += 1

        inv_max_0_indices = {}
        for t in range(T_len):
            for i in range(N):
                inv_max_0_indices[(i, t)] = index
                index += 1

        inv_indices = {}
        for t in range(T_len):
            for i in range(N):
                inv_indices[(i, t)] = index
                index += 1

        shortage_indices = {}
        for t in range(T_len):
            for i in range(N):
                shortage_indices[(i, t)] = index
                index += 1

        # Total number of variables
        V = index
        bounds = []

        # Bounds for tran variables (non-negative)
        for _ in range(N*N*(T_len)):
            bounds.append((0, None))

        # Bounds for s_inv variables (unbounded)
        for _ in range(N*(T_len)):
            bounds.append((None, None))

        # Bounds for inv_max_0 variables (non-negative)
        for _ in range(N*(T_len)):
            bounds.append((0, None))

        # Bounds for inv variables (unbounded)
        for _ in range(N*T_len):
            bounds.append((None, None))

        # Bounds for shortage variables (non-negative)
        for _ in range(N*T_len):
            bounds.append((0, None))
        
        # equality constraints
        A_eq = []
        b_eq = []

        for i in range(N):
            # model.addConstr(s_inv[i, 0] == next_start_inventory[i])
            row = np.zeros(V)
            idx = s_inv_indices[(i, 0)]
            row[idx] = 1
            A_eq.append(row)
            b_eq.append(next_start_inventory[i])

        for t in range(T_len):
            for i in range(N):
                # reactive:
                # model.addConstr(inv[i, t] == s_inv[i, t] - tran.sum(i, "*", t) + tran.sum("*", i, t) - next_demand[i][t])
                # proactive:
                # model.addConstr(inv[i, t] == s_inv[i, t] - next_demand[i][t]
                row = np.zeros(V)
                row[s_inv_indices[(i, t)]] = 1
                row[inv_indices[(i, t)]] = -1
                if self.env.reactive_tf:
                    for j in range(N):
                        row[tran_indices[(i, j, t)]] = -1
                        row[tran_indices[(j, i, t)]] = 1
                A_eq.append(row)
                b_eq.append(next_demand[i][t])
                # reactive:
                # model.addConstr(s_inv[i, t+1] == inv[i, t] + arrival_list[i][t])
                # proactive:
                # model.addConstr(s_inv[i, t+1] == inv[i, t] - tran.sum(i, "*", t) + tran.sum("*", i, t) + arrival_list[i][t])
                if t < T_len - 1:
                    row = np.zeros(V)
                    row[inv_indices[(i, t)]] = 1
                    row[s_inv_indices[(i, t+1)]] = -1
                    if not self.env.reactive_tf:
                        for j in range(N):
                            row[tran_indices[(i, j, t)]] = -1
                            row[tran_indices[(j, i, t)]] = 1
                    A_eq.append(row)
                    b_eq.append(-arrival_list[i][t])

        # # inequality constraints
        A_ub = []
        b_ub = []
        for i in range(N):
            # proactive:
            # model.addConstr(tran.sum(i, "*", t) <= inv_max_0[i, t])
            if not self.env.reactive_tf:
                row = np.zeros(V)
                for j in range(N):
                    row[tran_indices[(i, j, 0)]] = 1
                row[inv_max_0_indices[(i, 0)]] = -1
                A_ub.append(row)
                b_ub.append(0)


        for i in range(N):
            for t in range(T_len):
                # model.addConstr(inv_max_0[i, t] >= inv[i, t])
                row = np.zeros(V)
                row[inv_max_0_indices[(i, t)]] = -1
                row[inv_indices[(i, t)]] = 1
                A_ub.append(row)
                b_ub.append(0)
                # model.addConstr(shortage[i, t] >= -inv[i, t])
                row = np.zeros(V)
                row[inv_indices[(i, t)]] = -1
                row[shortage_indices[(i, t)]] = -1
                A_ub.append(row)
                b_ub.append(0)

        c = np.zeros(V)
        # Holding costs for inv_max_0 variables
        for t in range(T_len):
            for i in range(N):
                idx = inv_max_0_indices[(i, t)]
                c[idx] = (self.env.H[self.env.agents[i]]) 

        # Shortage costs for shortage variables
        for t in range(T_len):
            for i in range(N):
                idx = shortage_indices[(i, t)]
                c[idx] = (self.env.B[self.env.agents[i]])

        # Transportation costs for tran variables
        for t in range(T_len):
            for i in range(N):
                for j in range(N):
                    idx = tran_indices[(i, j, t)]
                    c[idx] = self.env.distance_matrix[i][j] * self.env.shipping_cost_per_distance

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'time_limit': 60.0})
        
        # 初始化transship_amounts，只更新参与transshipment的agent
        transship_amounts = [0 for _ in range(self.env.n_agents)]

        if result.success:
            x = result.x
            for (i, j, t), idx in tran_indices.items():
                if t == 0 and i!=j:
                    i_agent = I[i]
                    j_agent = J[j]
                    transship_amount = round(x[idx] - 0.1)

                    # 最大转运量阈值
                    tran_max_thres = max(0, self.env.inventory[self.env.agents[i_agent]] 
                                      if self.env.reactive_tf else 
                                      self.env.inventory[self.env.agents[i_agent]] - self.env.demand_dict[self.env.agents[i_agent]][self.env.step_num])
                    # 检查是否超过库存可接受范围
                    if transship_amounts[i_agent] - transship_amount >= -tran_max_thres:
                        transship_amounts[i_agent] -= transship_amount
                        transship_amounts[j_agent] += transship_amount
                    else:
                        # 如果超过可用库存，则只转运可用的最大数量
                        available_amount = tran_max_thres + transship_amounts[i_agent]
                        transship_amounts[i_agent] = -tran_max_thres
                        transship_amounts[j_agent] += available_amount
        else:
            print("Optimization failed:", result.message)
        
        transship_matrix = self.find_min_distance_matrix_alternative(transship_amounts)
        return transship_matrix