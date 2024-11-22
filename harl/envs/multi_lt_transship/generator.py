import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

from harl.envs.multi_lt_transship import SHANSHU_GEN_SOURCE

INF = 1e9

class base_generator(object):
    def __init__(self, length, max_num=INF):
        self.length = length
        self.max_num = max_num
        self.num_list = []
        self.theo_mean, self.theo_std = None, None
    
    def __getitem__(self, key = 0):
        return self.num_list[key]
    
    def get_data(self, round_int_tf = True):
        if round_int_tf:
            return np.round(self.num_list)
        return self.num_list
    
    def get_posterior_prob(self, condition_num, value):
        '''
        获取后验概率P(x = value|x > condition_num), 给不出来就返回0.5敷衍一下
        '''
        return 0.5
    
    def get_theoretical_mean_std(self):
        '''
        返回生成器的理论均值和标准差
        '''
        raise NotImplementedError("Subclass must implement abstract method")

class merton(base_generator):

    def __init__(self, length, std=4, max_num=INF):
        super(merton, self).__init__(length, max_num)
        self.std = std
    
    def get_data(self, round_int_tf = True):
        base = self.max_num / 2
        base = int(base) if round_int_tf else base
        start = 0
        var = self.std**2
        delta = 0.01
        delta_t = var
        u = 0.5*delta*delta
        a = 0
        b = 0.01
        lamda = var
        
        while(True):
            self.num_list = []
            self.drump = []
            self.no_drump = []
            self.no_drump.append(start)
            self.num_list.append(start)
            self.drump.append(0)
            for i in range(self.length):
                Z = np.random.normal(0, 1)
                N = np.random.poisson(lamda)
                Z_2 = np.random.normal(0, 2)
                M = a*N + b*(N**0.5)*Z_2
                new_X = self.num_list[-1] + u - 0.5*delta*delta + (delta_t**0.5)*delta*Z + M
                self.num_list.append(new_X)
            if round_int_tf:
                self.num_list = [int(math.exp(i)*base) for i in self.num_list]
            else:
                self.num_list = [math.exp(i)*base for i in self.num_list]
            
            if(np.mean(self.num_list)>0 and np.mean(self.num_list)<self.max_num):
                break
            
        for i in range(len(self.num_list)):
            self.num_list[i] = min(self.max_num, self.num_list[i])  
        return self.num_list
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:  
            self.theo_mean = self.max_num / 2
            self.theo_std = self.std
        return self.theo_mean, self.theo_std

class poisson(base_generator):

    def __init__(self, length, mean=10, max_num=INF):
        super(poisson, self).__init__(length, max_num)
        self.mean = mean

    def get_data(self, round_int_tf = True):
        self.num_list = np.random.poisson(self.mean, self.length)
        self.num_list = np.clip(self.num_list, 0, self.max_num)
        self.num_list = np.round(self.num_list) if round_int_tf else self.num_list
        return self.num_list
    
    def get_posterior_prob(self, condition_num, value):
        '''
        获取后验概率P(x = value|x > condition_num)
        '''
        P_value = stats.poisson.pmf(value, self.mean)
        P_ge_condition_num = 1 - stats.poisson.cdf(condition_num, self.mean)
        return (P_value/P_ge_condition_num if P_ge_condition_num != 0 else 1)
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:
            self.theo_mean = self.mean
            self.theo_std = self.mean**0.5
        return self.theo_mean, self.theo_std

class normal(base_generator):
    def __init__(self, length, mean=10, std=2, max_num=INF):
        super(normal, self).__init__(length, max_num)
        self.mean = mean
        self.std = std
    
    def get_data(self, round_int_tf = True):
        self.num_list = np.random.normal(self.mean, self.std**2, self.length)
        self.num_list = np.clip(self.num_list, 0, self.max_num)
        self.num_list = np.round(self.num_list) if round_int_tf else self.num_list
        return self.num_list
    
    def get_posterior_prob(self, condition_num, value):
        '''
        获取后验概率P(x = value|x > condition_num)
        '''
        P_value = stats.norm.pdf(value, self.mean, self.std)
        P_ge_condition_num = 1 - stats.norm.cdf(condition_num, self.mean, self.std)
        return (P_value/P_ge_condition_num if P_ge_condition_num != 0 else 1)
    
    def get_theoretical_mean_std(self):
        # 考虑截断在0和max_num之间的正态分布的均值和标准差
        if self.theo_mean is None:
            a = (0 - self.mean) / self.std
            b = (self.max_num - self.mean) / self.std
            mean = self.mean + self.std * (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
            var = (1 + (a*norm.pdf(a) - b*norm.pdf(b))/(norm.cdf(b) - norm.cdf(a)) 
                - ((norm.pdf(a) - norm.pdf(b))/(norm.cdf(b) - norm.cdf(a)))**2) * self.std**2
            self.theo_mean = mean
            self.theo_std = var**0.5
        return self.theo_mean, self.theo_std

class uniform(base_generator):
    def __init__(self, length, min_num=0, max_num=20):
        super(uniform, self).__init__(length, max_num)
        self.min_num = min_num

    def get_data(self, round_int_tf = True):
        self.num_list = np.random.uniform(self.min_num, self.max_num, self.length)
        self.num_list = np.clip(self.num_list, self.min_num, self.max_num)
        self.num_list = np.round(self.num_list) if round_int_tf else self.num_list
        return self.num_list
    
    def get_posterior_prob(self, condition_num, value):
        '''
        获取后验概率P(x = value|x > condition_num)
        '''
        P_value = 1/(self.max_num - self.min_num)
        P_ge_condition_num = 1/(self.max_num - self.min_num) * (self.max_num - condition_num)
        return (P_value/P_ge_condition_num if P_ge_condition_num != 0 else 1)
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:  
            self.theo_mean = (self.min_num + self.max_num) / 2
            self.theo_std = ((self.max_num - self.min_num)**2 / 12)**0.5
        return self.theo_mean, self.theo_std

class kim_dist(base_generator):
    def __init__(self, length, meanMin = 5, meanMax = 30, stdMin = 1, stdMax = 5, max_num = INF):
        super(kim_dist, self).__init__(length, max_num)
        self.meanMin = meanMin
        self.meanMax = meanMax
        self.stdMin = stdMin
        self.stdMax = stdMax

    def get_data(self, round_int_tf = True):
        self.meanList = []
        self.num_list = np.zeros(shape=[self.length])
        self.sigmaList = []
        
        self.meanList = np.random.randint(self.meanMin, self.meanMax, self.length)
        self.sigmaList = np.random.randint(self.stdMin, self.stdMax, self.length)

        for i in range(self.length):
            realizednumValue = self.getRealizednum(i, random.random())
            realizednumValue = round(realizednumValue) if round_int_tf else realizednumValue
            self.num_list[i] = min(realizednumValue, self.max_num) if realizednumValue >= 0 else 0
        return self.num_list    
    
    def getRealizednum(self, t, randomNumber):
        value = norm.ppf(randomNumber, loc=self.meanList[t], scale = self.sigmaList[t])
        return value
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:
            self.theo_mean = (self.meanMin + self.meanMax) / 2
            self.theo_std = (self.stdMin + self.stdMax) / 2
        return self.theo_mean, self.theo_std

class constant_dist(base_generator):
    def __init__(self, length, value = 10):
        super(constant_dist, self).__init__(length, value)
        self.value = value  
        
    def get_data(self, round_int_tf = True):
        self.num_list = np.ones(shape=[self.length]) * self.value
        self.num_list = np.round(self.num_list) if round_int_tf else self.num_list
        return self.num_list
    
    def get_posterior_prob(self, condition_num, value):
        '''
        获取后验概率P(x = value|x > condition_num)
        '''
        if value == self.value:
            return 1
        return 0
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:
            self.theo_mean = self.value
            self.theo_std = 0
        return self.theo_mean, self.theo_std

class kim_merton(base_generator):
    def __init__(self, length, mean_std=4, mean_max=30, std_std=1, std_max=5, max_num=INF):
        super(kim_merton, self).__init__(length, max_num)
        self.mean_std = mean_std
        self.mean_max = mean_max
        self.std_std = std_std
        self.std_max = std_max

    def get_data(self, round_int_tf=True):
        # Generate mean and std using Merton process
        mean_generator = merton(self.length, std=self.mean_std, max_num=self.mean_max)
        std_generator = merton(self.length, std=self.std_std, max_num=self.std_max)
        
        self.meanList = mean_generator.get_data(round_int_tf=False)
        self.sigmaList = std_generator.get_data(round_int_tf=False)
        
        self.num_list = np.zeros(shape=[self.length])
        
        for i in range(self.length):
            realizednumValue = self.getRealizednum(i, random.random())
            realizednumValue = round(realizednumValue) if round_int_tf else realizednumValue
            self.num_list[i] = min(max(realizednumValue, 0), self.max_num)
        
        return self.num_list

    def getRealizednum(self, t, randomNumber):
        value = norm.ppf(randomNumber, loc=self.meanList[t], scale=self.sigmaList[t])
        return value
    
    def get_theoretical_mean_std(self):
        if self.theo_mean is None:
            self.theo_mean = self.mean_max / 2
            self.theo_std = self.mean_std / 2
        return self.theo_mean, self.theo_std
    
class shanshu(base_generator):
    def __init__(self, length, SKU_id, agent_idx, max_num=INF):
        super(shanshu, self).__init__(length, max_num)
        self.cur_data_path = SHANSHU_GEN_SOURCE.format(SKU_id, agent_idx)
        self.cur_df = pd.read_csv(self.cur_data_path)

    def get_data(self, round_int_tf = True):
        # 随机生成start
        assert len(self.cur_df) >= self.length, "Data length is less than the required length"
        start = random.randint(0, len(self.cur_df)-self.length)
        cur_df_eval = self.cur_df.iloc[start:start+self.length].copy()
        self.num_list = cur_df_eval['sale'].tolist()
        self.num_list = np.clip(self.num_list, 0, self.max_num)
        self.num_list = np.round(self.num_list) if round_int_tf else self.num_list
        return self.num_list

    def get_theoretical_mean_std(self):
        # 根据cur_df计算理论均值和标准差
        if self.theo_mean is None:
            self.theo_mean = np.mean(self.cur_df['sale'])
            self.theo_std = np.std(self.cur_df['sale'])
        return self.theo_mean, self.theo_std

if __name__ == '__main__':
    gen = shanshu(100, 'SKU006', 0)
    print(gen.get_data())
    print(np.mean(gen.num_list))
    print(gen.get_theoretical_mean_std())

