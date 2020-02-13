# ------------------------------
# PGOperator.py: Mutation类
# ------------------------------
import numpy as np
import Utils


class Mutation:
    # 初始化
    def __init__(self, rate):
        """
        :param rate:变异概率
        """
        self.rate = rate

    # 变异操作（随机选取行和列，对这些行和列的值进行重新初始化）
    def mutate(self, individual, K, cloudlets):
        """
        :param K: 微云的数目
        :param individual: 需要进行变异的个体
        :param cloudlets: 微云集合
        """
        # 随机选取行和列
        row_rand_array = np.arange(K)
        col_rand_array = np.arange(K)
        np.random.shuffle(row_rand_array)  # 对行下标进行重新排列
        np.random.shuffle(col_rand_array)  # 对列下标进行重新排列
        # 相当于对|len1|×|len2|的矩阵重新初始化
        lens = int(np.ceil(K / 2))  # 选取其中1/3的行
        len1 = np.random.randint(1, lens)
        len2 = np.random.randint(1, lens)
        row_rand_array = row_rand_array[:len1]
        # len2 = Utils.calculate_len(len_Vt)  # 随机选取其中的一些列,len2为随机选取的个数
        col_rand_array = col_rand_array[:len2]
        # 重新初始化
        for i in range(len1):
            arriveRate = cloudlets[row_rand_array[i]].arrivalRate  # 取对应行的过载微云的任务到达率
            # 重新初始化
            for j in range(len2):
                if row_rand_array[i] != col_rand_array[j]:
                    individual[row_rand_array[i]][col_rand_array[j]] = np.round(np.random.uniform(0, arriveRate),
                                                                                decimals=5)
