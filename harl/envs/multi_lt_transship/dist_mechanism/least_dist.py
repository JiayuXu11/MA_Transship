import numpy as np
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class LeastDistanceDistribution(BaseMechanism):
    '''
    该类用于生成基于最小距离原则的分配计划，就是优先响应距离最近的intention
    '''
    def get_shipping_order(self, matrix):
        indices_sorted_by_value = np.argsort(matrix, axis=None)
        # reshape the sorted indices to match the original matrix shape
        indices_sorted_by_value = np.unravel_index(indices_sorted_by_value, matrix.shape)

        # exclude diagonal elements and symmetric elements
        indices_sorted_by_value = list(zip(indices_sorted_by_value[0], indices_sorted_by_value[1]))
        indices_sorted_by_value = [index for index in indices_sorted_by_value if index[0] < index[1]]
        return indices_sorted_by_value
    
    def get_transship_matrix(self, transship_intentions):
        '''
        该函数用于生成转运矩阵

        Args:
            transship_intentions: list, 每个agent的预期转运量
        '''
        transship_matrix = np.zeros((self.agent_num, self.agent_num))
        transship_intentions = transship_intentions.copy()
        shipping_order = self.get_shipping_order(self.distance_matrix)
        for a1, a2 in shipping_order:
            if a1 < self.agent_num and a2 < self.agent_num:
                # 表示一个想要货，一个想出货
                if transship_intentions[a1] * transship_intentions[a2] < 0:
                    tran_amount = min(abs(transship_intentions[a1]), abs(transship_intentions[a2]))
                    transship_matrix[a1][a2] = tran_amount if transship_intentions[a1] > 0 else -tran_amount
                    transship_matrix[a2][a1] = -transship_matrix[a1][a2]
                    transship_intentions[a1] -= transship_matrix[a1][a2]
                    transship_intentions[a2] -= transship_matrix[a2][a1]

        return transship_matrix
