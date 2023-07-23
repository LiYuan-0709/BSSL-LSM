import numpy as np
import time
from numpy import random as rd
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000, suppress=True)
from aimFunction import function_wsy

def objective_func2(x):
    # 这也是一个用来测试的目标函数
    return x ** 2

def GetTime(func_name):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func_name(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print("pso算法运行时间为%f秒" % run_time)
        file_handle=open('recorder_PSO.txt',mode='a')
        file_handle.write("PSO run_time1: %s \n"%(run_time))
        file_handle.close()
        return ret
    return wrapper
 
 
class PSO(object):
    def __init__(self, alpha, times, obj_type, particle_num, w, c1, c2, obj_func, **kwargs):
        """
        :param alpha: 约束因子，用来控制速度的权重
        :param times: 算法的迭代次数
        :param obj_type: 目标函数的类型，传入"Max"表示目标函数为最大化问题，传入"Min"表示目标函数为最小化问题
        :param particle_num: 粒子数量
        :param w: 惯性因子
        :param c1: 源于自身经验的加速常数
        :param c2: 源于其他粒子的加速常数
        :param obj_func: 被优化的目标函数
        :param kwargs: 目标函数决策变量及取值范围，给定方式例如：x1=(x1_min, x1_max), x2=(x2_min, x2_max), ...,
                        注意此处决策变量及其范围给定的顺序需和目标函数obj_func中决策变量参数顺序一致
        """
        self.alpha = alpha
        self.times = times
        self.obj_type = obj_type
        self.particle_num = particle_num
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.obj_func = obj_func
        self.kwargs = kwargs

        # self.v_max确定决策变量每一维的最大飞翔速度，根据最大飞翔速度为决策变量取值范围的百分之10到百分之20之间来取值
        self.v_max = (np.array(list(kwargs.values()))[:, 1] - np.array(list(kwargs.values()))[:, 0]) * 0.15
        assert np.all(self.v_max > 0), "输入决策变量的最小值大于最大值"

        self.particles = np.empty((self.particle_num, len(self.kwargs)))

        # for循环初始化初代粒子群
        for dimension in range(len(self.kwargs)):
            var_range = list(self.kwargs.values())[dimension]
            self.particles[:, dimension:dimension + 1] = rd.uniform(var_range[0], var_range[1], (self.particle_num, 1))
        assert self.obj_type.lower() in ["max", "min"], "目标函数类型填写错误, 请填写'Max'或'Min'"

        if self.obj_type.lower() == "max":
            self.individual_best_fit_value = [-float("inf")] * self.particle_num
        else:
            self.individual_best_fit_value = [float("inf")] * self.particle_num

        self.individual_best_fit_value = np.array(self.individual_best_fit_value)
        self.individual_best_position = np.empty((self.particle_num, len(self.kwargs)))

        # 确定初始速度
        self.v = np.array([rd.uniform(-v_max_value, v_max_value, self.particle_num) for v_max_value in self.v_max]).T

        # self.bests_pop_fit_values用于存放每次迭代群体最优适应度, self.bests_pop_positions用于存放每次迭代最优群体适应度对应的位置
        self.bests_pop_fit_values = []
        self.bests_pop_positions = []
 
    def get_fit_value(self):
        # 求取每个粒子的适应度
        self.fit_value = np.empty(self.particle_num)
        for particle_index in range(self.particle_num):
            decision_var = self.particles[particle_index]
            decision_var_max = np.array(list(self.kwargs.values()))[:, 1]
            decision_var_min = np.array(list(self.kwargs.values()))[:, 0]
            # 一下用于限制决策变量在设定的取值范围之内，如果出了取值范围则设置其适应度为负无穷或正无穷
            if np.all((decision_var_max - decision_var) >= 0) and np.all((decision_var - decision_var_min) > 0):
                self.fit_value[particle_index] = self.obj_func(*tuple(self.particles[particle_index]))
            else:
                if self.obj_type.lower() == "max":
                    self.fit_value[particle_index] = -float("inf")
                else:
                    self.fit_value[particle_index] = float("inf")
 
    def get_best(self):
        # 获取个体最优以及群体最优粒子位置
        # replace_index为个体最优位置需要替换的粒子的索引
        if self.obj_type.lower() == "max":
            replace_index = self.fit_value > self.individual_best_fit_value
        else:
            replace_index = self.fit_value < self.individual_best_fit_value
        self.individual_best_position[replace_index] = self.particles[replace_index]
        self.individual_best_fit_value[replace_index] = self.fit_value[replace_index]

        # 根据个体最优适应度获得群体最优适应度，以及群体最优粒子位置
        if self.obj_type.lower() == "max":
            self.pop_bset_fit_value = np.max(self.individual_best_fit_value)
        else:
            self.pop_bset_fit_value = np.min(self.individual_best_fit_value)
        best_index = list(self.individual_best_fit_value).index(self.pop_bset_fit_value)
        self.pop_best_position = self.individual_best_position[best_index]
        self.bests_pop_fit_values.append(self.pop_bset_fit_value)
        self.bests_pop_positions.append(self.pop_best_position)
 
    def calc_v_position(self):
        # 得到粒子群移动后的位置
        self.v = self.v + self.c1 * rd.random() * (self.individual_best_position - self.particles) + self.c2 * rd.random() * (self.pop_best_position - self.particles)

        # 以下两步用于限制速度在[-self.v_max, self.v_max]之内
        self.v = np.array([self.v[r, c] if self.v[r, c] < self.v_max[c] else self.v_max[c] for r in range(self.particle_num) for c in range(len(self.kwargs))]).reshape(self.particle_num, len(self.kwargs))
        self.v = np.array([self.v[r, c] if self.v[r, c] > -self.v_max[c] else -self.v_max[c] for r in range(self.particle_num) for c in range(len(self.kwargs))]).reshape(self.particle_num, len(self.kwargs))

        # 根据计算出的速度获取粒子群移动后的位置
        self.particles = self.particles + self.alpha * self.v
 
    @GetTime
    def run(self):
        for i in range(self.times):
            print("PSO搜索的第 %s 次"%(i))
            self.get_fit_value()
            self.get_best()
            self.calc_v_position()
        print("目标函数最优值为：", self.bests_pop_fit_values[-1])
        print("最优解为：", self.bests_pop_positions[-1])
        self.draw()
 
    def draw(self):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(np.arange(1, self.times + 1), self.bests_pop_fit_values, label="fit_value trend", color="r")
        ax.set_xlabel("iterate times")
        ax.set_ylabel("fit_value")
        ax.grid()
        ax.legend()
        ax.set_title("PSO")
        plt.savefig("PSO-iter_fig.png")
 
 
def main():
    pso = PSO(0.8, 30, "Max", 30, 0.5, 2, 2, function_wsy, x1=(800, 800.01), x2=(0.1, 0.9),x3=(0.1, 0.9),
              x4=(0.1, 0.9),x5=(0.1, 0.9),x6=(0.1, 0.9),x7=(32,640),x8=(0.0001,0.01),x9=(10,35),x10=(0.05,0.5),x11=(10,1000))

    start_time2 = time.time()

    pso.run()
    
    end_time2 = time.time()
    run_time2 = end_time2 - start_time2

    file_handle=open('recorder_PSO.txt',mode='a')
    file_handle.write("PSO run_time2: %s \n"%(run_time2))
    file_handle.close()
    # (self, alpha, times, obj_type, particle_num, w, c1, c2, obj_func, **kwargs)
    """
        :param alpha: 约束因子，用来控制速度的权重
        :param times: 算法的迭代次数
        :param obj_type: 目标函数的类型，传入"Max"表示目标函数为最大化问题，传入"Min"表示目标函数为最小化问题
        :param particle_num: 粒子数量
        :param w: 惯性因子
        :param c1: 源于自身经验的加速常数
        :param c2: 源于其他粒子的加速常数
        :param obj_func: 被优化的目标函数
        :param kwargs: 目标函数决策变量及取值范围，给定方式例如：x1=(x1_min, x1_max), x2=(x2_min, x2_max), ...,
                        注意此处决策变量及其范围给定的顺序需和目标函数obj_func中决策变量参数顺序一致
        """

if __name__ == "__main__":
    main()
