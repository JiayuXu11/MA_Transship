import numpy as np
import random
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class HierDistribution(BaseMechanism):
    '''
    该类用于生成基于阶梯原则的分配计划，就是根据距离将不同的响应可能划入不同的阶梯，然后按照代表距离从小到大的阶梯顺序响应匹配
    '''
    # 获取最近一阶梯的转运对
    def get_pooling_pairs(distance_matrix, transship_intentions, thres):
        n = len(transship_intentions)
        possible_pairs = []

        # Find possible transship pairs
        for i in range(n):
            for j in range(n):
                if i != j and transship_intentions[i] < 0 and transship_intentions[j] > 0:
                    possible_pairs.append((i, j, distance_matrix[i][j]))

        # Sort pairs by distance
        possible_pairs.sort(key=lambda x: x[2])

        # Filter pairs within the distance threshold
        mini_pool_pairs = [pair[:2] for pair in possible_pairs if pair[2] - possible_pairs[0][2] <= thres]

        if not mini_pool_pairs:
            return False
        
        return mini_pool_pairs
    
    def get_transship_matrix(self, transship_intentions): 
        threshold, how = self.allocate_args.get('threshold', 400), self.allocate_args.get('how', 'ratio')
        transship_matrix = np.zeros((self.agent_num, self.agent_num))
        transship_intentions = transship_intentions.copy()

        # 不断对各个阶梯进行转运匹配
        while True:
            # 获取最近一阶梯的转运对
            mini_pool_pairs = self.get_pooling_pairs(self.distance, transship_intentions, threshold)
            
            # 如果没有可匹配的转运对，结束
            if not mini_pool_pairs:
                break
            
            # 对最近一阶梯的转运对进行转运匹配
            while mini_pool_pairs:
                # 随机选择一对转运对匹配
                if how == 'even':
                    curr_pair = random.choice(mini_pool_pairs)
                # 按照转运意向的量，加权随机选择转运对
                elif how == 'ratio':
                    weights = [transship_intentions[pair[1]] for pair in mini_pool_pairs]
                    curr_pair = random.choice(mini_pool_pairs, weights=weights, k=1)
                transship_matrix[curr_pair[0]][curr_pair[1]] -= 1
                transship_matrix[curr_pair[1]][curr_pair[0]] += 1
                transship_intentions[curr_pair[0]] += 1
                transship_intentions[curr_pair[1]] -= 1
                if transship_intentions[curr_pair[0]] == 0:
                    mini_pool_pairs = list(filter(lambda x: curr_pair[0] not in x, mini_pool_pairs))
                if transship_intentions[curr_pair[1]] == 0:
                    mini_pool_pairs = list(filter(lambda x: curr_pair[1] not in x, mini_pool_pairs))
        
        return transship_matrix

    # def get_payment(self, transship_intentions, transship_matrix):

    #     return [0] * self.agent_num
    
