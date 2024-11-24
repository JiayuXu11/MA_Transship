import numpy as np
# import coptpy as cp
# from coptpy import COPT

class BaseMechanism:
    def __init__(self, distance_matrix, allocate_args=None, others_info = None):
        # TODO: others_info先直接把Env扔进来了，后面有时间再改的正规一点
        self.distance_matrix = distance_matrix
        self.agent_num = distance_matrix.shape[0]
        self.allocate_args = allocate_args

        self.env = others_info
        self.revenue_allocation_method = self.env.revenue_allocation_method
        self.constant_transshipment_price = self.env.constant_transshipment_price
        self.shipping_cost_matrix = self.env.shipping_cost_matrix

    def get_transship_matrix(self, transship_intentions):
        '''
        根据每个人的intentions生成转运矩阵, (i,j)表示j转运给i的量(正数表示j转给i, 负数表示i转给j)
        如果该mechanism是不参考其他agent的intentions而生成转运矩阵的, 
        那么这里的transship_intentions就是表达每个人是否愿意接受transship支配, 1表示愿意, 0表示不愿意。
        '''
        return np.zeros((self.agent_num, self.agent_num))
    
    def get_payment(self, transship_matrix):
        '''
        根据转运矩阵生成每个agent的payment: [agent0_payment, agent1_payment, ...] 
        正数表示agent需要支付, 负数表示agent需要收到payment
        '''
        if self.revenue_allocation_method == 'constant':
            shipping_cost_list = (self.shipping_cost_matrix * np.maximum(transship_matrix, 0)).sum(axis=1)
            payment = transship_matrix.sum(axis=1) * self.constant_transshipment_price + shipping_cost_list
        else:
            raise NotImplementedError
        return payment
    
    def get_mechanism_result(self, transship_intentions, **kwargs):
        '''
        生成mechanism的支配结果，包含转运矩阵和payment
        '''
        transship_matrix = self.get_transship_matrix(transship_intentions)
        payment = self.get_payment(transship_matrix)
        return transship_matrix, payment
    
    def est_future_arrival(self, agent_name, order_amount, look_ahead_days):
        '''
        预估未来look_ahead_days天的到货情况
        '''
        future_arrival = [0 for _ in range(int(look_ahead_days))]
        lt_mean_list = [self.env.get_dist_mean_std(lt_gen)[0] for lt_gen in self.env.LT_generators[agent_name]]
        residual_days_today_order = int(sum(lt_mean_list) - 0)
        if residual_days_today_order < look_ahead_days:
            future_arrival[residual_days_today_order] += order_amount
        for (frag_idx, intransit_list) in enumerate(self.env.intransit[agent_name]):
            for intransit_dict in intransit_list:
                residual_days = int(sum(lt_mean_list[frag_idx:]) - intransit_dict['passed_days'])
                if residual_days < look_ahead_days:
                    residual_days = max(residual_days, 0)
                    future_arrival[residual_days] += intransit_dict['qty']
        return future_arrival
    
    def find_min_distance_matrix_alternative(self, transship_actual):
        '''
        按照潜在路径（pos -> neg）从小到大排列然后依次 fulfill transship matrix
        '''
        # 初始化transship_matrix
        transship_matrix = np.zeros((self.agent_num, self.agent_num))
        
        # 记录未满足的供需
        demand = np.array([max(val, 0) for val in transship_actual])
        supply = np.array([max(-val, 0) for val in transship_actual])

        # 找出所有潜在的供需配对，按照距离从小到大排序
        potential_paths = []
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i != j and supply[i] > 0 and demand[j] > 0:
                    potential_paths.append((i, j, self.distance_matrix[i, j]))

        # 按距离从小到大排序
        potential_paths.sort(key=lambda x: x[2])

        # 依次fulfill每个供需对
        for i, j, _ in potential_paths:
            if supply[i] == 0 or demand[j] == 0:
                continue

            # 选择可运送的最大值，满足供需约束
            transfer_amount = min(supply[i], demand[j])
            transship_matrix[i][j] = transfer_amount
            transship_matrix[j][i] = -transfer_amount

            # 更新剩余供需
            supply[i] -= transfer_amount
            demand[j] -= transfer_amount

            # 如果供需已经满足，跳过
            if supply[i] == 0:
                continue
            if demand[j] == 0:
                continue

        return -transship_matrix


