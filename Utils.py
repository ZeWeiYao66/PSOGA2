# ------------------------------
# 工具库：主要是计算微云响应时间,对粒子位置与速度进行检查的函数
# ------------------------------
import math
import numpy as np
import time
from PGCloudlet import *


# Erlang C 公式
def ErlangC(n, p):
    """
    :param n: 微云的服务器个数n
    :param p: 任务到达率/微云的服务率
    :return: ErlangC公式计算出的值
    """
    # 给p加上一个很小的数，避免出现除0的现象
    p = p + 1e-5
    L = (n * p) ** n / math.factorial(n)
    R = 1 / (1 - p)
    M = L * R
    sum_ = 0
    for k in range(n):
        sum_ += (n * p) ** k / math.factorial(k)
    # print(p)
    return M / (sum_ + M)


# 对粒子的速度进行检查
def CheckSpeed(velocity, K, cloudlets):
    """
    :param K: 微云的数目
    :param velocity:粒子的速度
    :param cloudlets: 微云集合
    """
    for i in range(K):
        arrRate = cloudlets[i].arrivalRate * 0.1  # 任务到达率
        for j in range(K):
            # 如果粒子的某一个分量速度超过取值范围，则设为边界值
            if velocity[i][j] > arrRate:
                velocity[i][j] = arrRate
            elif velocity[i][j] < -arrRate:
                velocity[i][j] = -arrRate


# 对粒子的位置进行检查
def CheckSolution(solution, K, cloudlets):
    """
    :param K: 微云的数目
    :param solution: 粒子的解
    :param cloudlets: 微云集合
    """
    # 首先检查解当中是否有负数存在，存在则置为0，保证解中的每个数都>=0
    new_solution = np.maximum(solution, 0)
    # 接下来对解的每行进行检查，防止每行值之和超过过载微云i的任务到达率
    for i in range(K):
        while new_solution[i, :].sum() > cloudlets[i].arrivalRate:
            row = new_solution[i, :]
            row_max_index = np.argmax(row)
            # 如果该行中最大的值大于任务到达率的一半，就将其除以2
            if new_solution[i][row_max_index] >= cloudlets[i].arrivalRate / 2:
                new_solution[i][row_max_index] = new_solution[i][row_max_index] / 2
            new_solution[i][row_max_index] -= np.random.rand()
            if new_solution[i][row_max_index] < 0:
                new_solution[i][row_max_index] = 0
    # 接下来对解的每列进行检查，防止每列值之和超过不过载微云j的总任务接受率(⭐⭐⭐要考虑不过载微云本身的任务到达率)
    for j in range(K):
        arg = cloudlets[j].serverNum * cloudlets[j].serverRate - cloudlets[j].arrivalRate
        while new_solution[:, j].sum() >= arg:
            col = new_solution[:, j]
            col_max_index = np.argmax(col)  # 获取该列中的最大值下标
            new_solution[col_max_index][j] -= np.random.rand()
            if new_solution[col_max_index][j] < 0:
                new_solution[col_max_index][j] = 0
    return np.round(new_solution, decimals=5)


if __name__ == '__main__':
    t1 = time.time()
    for _ in range(100000):
        arr = np.random.randint(-5, 5, (20, 20))
        CheckSolution(arr, 20, cloudlets.cloudlets)
    t2 = time.time()
    print(t2 - t1)
