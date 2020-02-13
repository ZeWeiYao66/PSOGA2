import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from PGCloudlet import Cloudlet, Cloudlets
from PGComponent import Individual, Population
from PGOperator import Mutation
from PSOGA import Psoga
import time

'''Step1：参数设置'''
K = 40  # 微云个数
N = 1000  # 迭代数目
Pnum = 100  # 种群数目
Rmut = 0.02  # 变异概率
w = 0.8  # 惯性权重
c1 = 2  # 学习因子
c2 = 2

'''Step2：对每个微云i的服务率，服务器数量，任务到达率,网络延时进行初始化'''
cloudlet = Cloudlet()  # 单个微云
cloudlets = Cloudlets(cloudlet, K)  # 微云集合
serNum, serRate, arrRate, delayMat = cloudlets.initialize()  # 初始化微云集合
np.set_printoptions(threshold=1e6)
np.set_printoptions(suppress=True)
print('服务器数目：', serNum.tolist())
print('服务率：', serRate.tolist())
print('任务到达率：', arrRate.tolist())
print('网络延时：', delayMat.tolist())
'''Step2：计算每个微云的本地任务响应时间'''
waitTimes = cloudlets.CalWaitTimes()
print('所有微云的本地任务响应时间: ', waitTimes)
print('其中最大的本地任务响应时间: ', max(waitTimes))
waitTimes_sorted = np.sort(waitTimes)  # 按照微云的本地任务响应时间进行排序
'''Step3：按照微云的本地任务响应时间，将K个微云划分成过载和不过载的微云.
          从第二个微云开始，到倒数第二个结束.'''
fitness = []  # 存放适应度值
result = []  # 存放每一次划分的结果
'''Step4：初始化粒子的解（也即任务流g）,计算适应值，并进行迭代'''
I = Individual()  # 单个粒子
P = Population(I, cloudlets, Pnum)  # 种群(粒子集合)
print(P.gbestFitness)
M = Mutation(Rmut)  # 变异操作
psoga = Psoga(P, M)
start_time = time.process_time()
bestResult, fit = psoga.run(N)  # 运行主算法，获得每一次划分的最优解及其适应度
end_time = time.process_time()
print('total time outside:', end_time - start_time)
'''Step5：从result中计算最优解'''
