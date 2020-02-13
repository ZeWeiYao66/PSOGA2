# ------------------------------
# PGComponent.py: Individual类、Population类
# -----------------------------
import numpy as np
import copy
from PGCloudlet import Cloudlet, Cloudlets
from Utils import *


# 表征个体（即粒子）
class Individual:
    # 粒子参数初始化
    def __init__(self):
        self.solution = None  # 粒子所代表的解（也即粒子的位置）
        self.velocity = None  # 粒子的速度
        self.fitness = None  # 粒子的适应度
        self.pbest = None  # 粒子的个体极值
        self.pbestFitness = None  # 粒子的个体极值适应度

    # 初始化粒子的位置，该解为K × K的矩阵，并求适应度
    def initSolution(self, K, cloudlets, delayMatrix):
        """
        :param K: 微云的数目
        :param cloudlets: 微云集合
        :param delayMatrix: 网络延时矩阵
        """
        np.set_printoptions(suppress=True)  # 取消科学计数法
        sol = []  # 随机生成的解
        # 生成解
        '''problem2：粒子的初始化得修改，初始化的时间占用过大'''
        for i in range(K):
            temp = np.round(np.random.uniform(0, cloudlets[i].arrivalRate, size=K), decimals=5)
            sol.append(temp)
        # 转换成numpy.ndarray
        sol = np.array(sol)
        for i in range(K):
            sol[i][i] = 0
        # 对生成的解进行检查
        self.solution = CheckSolution(sol, K, cloudlets)
        # 更新个体极值
        self.pbest = sol
        # 更新个体的适应度值
        cloudlets_copy = copy.deepcopy(cloudlets)
        self.Calculate_fitness(cloudlets_copy, K, delayMatrix)
        self.pbestFitness = self.fitness

    # 初始化粒子的速度，为K × K的矩阵
    def initVelocity(self, K, cloudlets):
        """
        :param K: 微云的数目
        :param cloudlets: 微云集合
        """
        speed = []
        # 随机生成速度
        for i in range(K):
            # 微云的到达率
            arr_rate = cloudlets[i].arrivalRate
            # 速度区间取任务到达率的10%
            temp = np.random.uniform(-arr_rate * 0.1, arr_rate * 0.1, size=K)
            speed.append(temp)
        self.velocity = np.round(np.array(speed), decimals=5)

    # 计算粒子对应的适应度值
    def Calculate_fitness(self, cloudlets, K, delayMatrix):
        """ 
        :param K: 微云的数目
        :param cloudlets: 微云集合
        :param delayMatrix: 网络延时矩阵
        :return: 粒子的适应度值
        """""
        # 1.对微云的任务到达率进行更新,先减少后增加
        for index in range(K):
            cloudlets[index].arrivalRate -= self.solution[index, :].sum()
            cloudlets[index].arrivalRate += self.solution[:, index].sum()
        # 2.计算任务等待时间与网络延迟
        responseTime = []
        append = responseTime.append
        # 3.计算过载微云的任务响应时间
        for i in range(K):
            WaitTime = cloudlets[i].CalWaitTime()
            DelayTime = 0
            for j in range(K):
                DelayTime += self.solution[i][j] * delayMatrix[i][j]
            append(WaitTime + DelayTime)
        pbest_fitness = np.round(max(responseTime), decimals=5)
        self.fitness = pbest_fitness


# 表征种群
class Population:
    def __init__(self, individual, cloudlets, size, w=0.8, c1=2, c2=2):
        self.individual = individual  # 个体模板
        self.cloudlets = cloudlets  # 微云集合(包括微云数目、延迟时间)
        self.size = size  # 种群大小
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.r1 = np.round(np.random.rand(), decimals=5)  # [0,1]上的随机数
        self.r2 = np.round(np.random.rand(), decimals=5)
        self.gbest = None  # 全局极值
        self.gbestFitness = None  # 全局极值对应的适应度值
        self.individuals = None  # 种群

    # 初始化粒子群
    def initialize(self):
        IndCls = self.individual.__class__
        # 声明individual对象
        self.individuals = np.array([IndCls() for i in range(self.size)], dtype=IndCls)
        # 初始化粒子的位置，并更新适应度值
        self.initSolu()
        # 更新全局极值
        fitnesses = [self.individuals[i].pbestFitness for i in range(self.size)]
        gbest_index = np.argmin(fitnesses)
        self.gbest = self.individuals[gbest_index].pbest
        self.gbestFitness = min(fitnesses)
        # 初始化粒子的速度
        self.initVelo()

    # 初始化粒子的解
    def initSolu(self):
        for i in range(self.size):
            self.individuals[i].initSolution(self.cloudlets.K, self.cloudlets.cloudlets, self.cloudlets.C)

    # 初始化粒子的速度
    def initVelo(self):
        for i in range(self.size):
            self.individuals[i].initVelocity(self.cloudlets.K, self.cloudlets.cloudlets)

    # 更新粒子的位置和速度
    def update_position(self, index):
        """
        :param index: 更新第index个粒子
        """
        """problem2: 速度需要考虑界限，更新粒子的解时会出现负值，要进行调整"""
        V_t_plus_1 = self.w * self.individuals[index].velocity \
                     + self.c1 * self.r1 * (self.individuals[index].pbest - self.individuals[index].solution) \
                     + self.c2 * self.r2 * (self.gbest - self.individuals[index].solution)
        # 检查粒子速度是否符合条件
        CheckSpeed(V_t_plus_1, self.cloudlets.K, self.cloudlets.cloudlets)
        # 更新位置与速度
        X_t_plus_1 = self.individuals[index].solution + np.round(V_t_plus_1, decimals=5)
        # print(X_t_plus_1)
        # 检查粒子位置是否符合条件
        self.individuals[index].solution = CheckSolution(X_t_plus_1, self.cloudlets.K, self.cloudlets.cloudlets)
        # print(self.individuals[index].solution)
        self.individuals[index].velocity = np.round(V_t_plus_1, decimals=5)

    # 更新个体极值
    def update_pbest(self):
        # 对微云集合进行深复制，这样子对复制集合操作不会对原集合产生影响
        for k in range(self.size):
            cloudlets_copy = copy.deepcopy(self.cloudlets.cloudlets)
            # 计算个体的适应度值
            self.individuals[k].Calculate_fitness(cloudlets_copy, self.cloudlets.K, self.cloudlets.C)
            # 如果更新过的粒子的适应度值比之前好，就对个体极值进行更新
            if self.individuals[k].pbestFitness > self.individuals[k].fitness:
                self.individuals[k].pbestFitness = self.individuals[k].fitness
                self.individuals[k].pbest = self.individuals[k].solution

    # 更新全局极值
    def update_gbest(self):
        # 获取粒子群的适应度值
        fitnesses = [self.individuals[i].pbestFitness for i in range(self.size)]
        # 选取适应度值最小的粒子作为全局极值
        gbest_fit = min(fitnesses)
        gbest_index = np.argmin(fitnesses)
        if self.gbestFitness > gbest_fit:
            self.gbestFitness = gbest_fit
            self.gbest = self.individuals[gbest_index].pbest
