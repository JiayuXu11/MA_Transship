import numpy as np
from scipy.optimize import linprog
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class AnupindiDistribution(BaseMechanism):
    '''
    该类用于复现Anupindi等人的分配机制
    '''
    def get_transship_matrix(self, transship_intentions):
        '''
        Generates the transshipment matrix.
        Args:
            transship_intentions: list, indicates whether each agent is willing to participate in transshipment

        Returns:
            transship_matrix: np.array, the transshipment matrix where (i,j) represents the amount agent j transships to agent i
        '''
        cur_demand = {agent_name: self.env.demand_dict[agent_name][self.env.step_num]
                      for agent_name in self.env.agents}
        transship_matrix = np.zeros((self.agent_num, self.agent_num))

        if self.env.reactive_tf and self.env.transship_type in ['force_to_transship', 'half_autonomy']:
            transship_intentions_qty = [-self.env.inventory[agent_name] + cur_demand[agent_name] 
                                    if transship_intentions[agent_id]==1 else 0
                                        for agent_id, agent_name in enumerate(self.env.agents)]
        elif self.env.transship_type in ['force_to_transship', 'half_autonomy']:
            transship_intentions_qty = [max(0, -self.env.inventory[agent_name] + cur_demand[agent_name] 
                                        + self.env.demand_est_dict[agent_name] - self.est_future_arrival(agent_name, 0, 1)[0]) + 
                                        min(0, -self.env.inventory[agent_name] + cur_demand[agent_name] 
                                        + self.env.demand_est_dict[agent_name])
                                    if transship_intentions[agent_id]==1 else 0
                                        for agent_id, agent_name in enumerate(self.env.agents)]
        else:
            transship_intentions_qty = transship_intentions.copy()
            
        transship_intentions_qty = np.array(transship_intentions_qty)

        num_vars = self.agent_num * self.agent_num

        # Objective coefficients c (negated because linprog performs minimization)
        c = []
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                rev = (self.env.P[self.env.agents[j]] 
                       + self.env.H[self.env.agents[i]] 
                       + self.env.B[self.env.agents[j]]
                       - self.env.C[self.env.agents[i]] 
                       - self.distance_matrix[i,j] * self.env.shipping_cost_per_distance)
                c.append(-rev)  # Negate for minimization

        # Bounds on variables (all variables >= 0)
        bounds = [(0, None)] * num_vars

        # Constraints: A_ub x <= b_ub
        A_ub = []
        b_ub = []

        # Column sum constraints (transship in quantities)
        for j in range(self.agent_num):
            A_row = [0.0] * num_vars
            for i in range(self.agent_num):
                idx = i * self.agent_num + j
                A_row[idx] = 1.0
            A_ub.append(A_row)
            b_ub.append(max(transship_intentions_qty[j], 0))

        # Row sum constraints (transship out quantities)
        for i in range(self.agent_num):
            A_row = [0.0] * num_vars
            for j in range(self.agent_num):
                idx = i * self.agent_num + j
                A_row[idx] = 1.0
            A_ub.append(A_row)
            b_ub.append(max(-transship_intentions_qty[i], 0))

        # Suppress output by setting options
        options = {'disp': False}

        # Solve the LP using linprog with HiGHS
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)

        if res.success:
            # Extract solution
            Q_sol = np.array(res.x).reshape((self.agent_num, self.agent_num))
            transship_matrix = Q_sol.T - Q_sol
            fun = -res.fun  # Negate to get the original maximized value

            # Get dual variables (shadow prices)
            duals = res.ineqlin.marginals  # Corrected line
            # The order of duals corresponds to the constraints in A_ub
            # First the column sum constraints, then the row sum constraints
            in_price = -duals[:self.agent_num]
            out_price = -duals[self.agent_num:2*self.agent_num]
            self.prices = list(out_price) + list(in_price)

            # 初始化transship_amounts，只更新参与transshipment的agent
            transship_amounts = [0 for _ in range(self.env.n_agents)]

            # 检查transship_matrix是否满足库存限制，并更正为整数
            for i in range(self.agent_num):
                for j in range(self.agent_num):
                    if i != j and transship_matrix[i, j] < 0:
                        transship_amount = round(-transship_matrix[i, j] - 0.1)

                        # 最大转运量阈值
                        tran_max_thres = max(0, self.env.inventory[self.env.agents[i]] 
                                          if self.env.reactive_tf else 
                                          self.env.inventory[self.env.agents[i]] - cur_demand[self.env.agents[i]])
                        # 检查是否超过库存可接受范围
                        if transship_amounts[i] - transship_amount >= -tran_max_thres:
                            transship_amounts[i] -= transship_amount
                            transship_amounts[j] += transship_amount

                        else:
                            # 如果超过可用库存，则只转运可用的最大数量
                            available_amount = tran_max_thres + transship_amounts[i]
                            transship_amounts[i] = -tran_max_thres
                            transship_amounts[j] += available_amount

        else:
            print("Optimization was stopped with status:", res.message)
        
        transship_matrix = self.find_min_distance_matrix_alternative(transship_amounts)

        return transship_matrix

    def get_payment(self, transship_matrix):
        '''
        Generates the payment for each agent based on the transshipment matrix.
        Positive values indicate payments to be made, negative values indicate payments to be received.
        '''
        payment_type = self.allocate_args.get('payment_type', 'dual')
        if payment_type != 'dual':
            # 使用父类的get_payment方法
            return super().get_payment(transship_matrix)
        else:
            out_price = self.prices[0:self.agent_num]
            in_price = self.prices[self.agent_num:]
            transship_qty = np.sum(transship_matrix, axis=1)
            # Calculate revenue allocation
            rev_allocation = np.multiply(out_price, np.maximum(-transship_qty, 0)) + np.multiply(in_price, np.maximum(transship_qty, 0))
            # Calculate revenue without payment
            rev_no_pay = (np.multiply([self.env.P[self.env.agents[i]] + self.env.B[self.env.agents[i]] for i in range(self.agent_num)], np.maximum(transship_qty, 0)) +
                        np.multiply([self.env.H[self.env.agents[i]] - self.env.C[self.env.agents[i]] for i in range(self.agent_num)], np.maximum(-transship_qty, 0)))
            payment = rev_no_pay - rev_allocation
            return payment