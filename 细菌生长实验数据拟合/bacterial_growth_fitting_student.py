import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_bacterial_data(file_path):
    """
    从文件中加载细菌生长实验数据
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: 包含时间和酶活性测量值的元组
    """
    # TODO: 实现数据加载功能 (大约3行代码)
    # [STUDENT_CODE_HERE]
    data = np.loadtxt(file_path, delimiter=',')  # 加载数据文件
    t = data[:, 0]  # 第一列为时间
    activity = data[:, 1]  # 第二列为酶活性测量值
    return t, activity

def V_model(t, tau):
    """
    V(t)模型函数
    
    参数:
        t (float or numpy.ndarray): 时间
        tau (float): 时间常数
        
    返回:
        float or numpy.ndarray: V(t)模型值
    """
    # TODO: 根据V(t) = 1 - e^(-t/τ)实现模型函数 (1行代码)
    # [STUDENT_CODE_HERE]
    
    return 1 - np.exp(-t / tau)  # V(t) = 1 - e^(-t/τ)

def W_model(t, A, tau):
    """
    W(t)模型函数
    
    参数:
        t (float or numpy.ndarray): 时间
        A (float): 比例系数
        tau (float): 时间常数
        
    返回:
        float or numpy.ndarray: W(t)模型值
    """
    # TODO: 根据W(t) = A(e^(-t/τ) - 1 + t/τ)实现模型函数 (1行代码)
    # [STUDENT_CODE_HERE]
    return A * (np.exp(-t / tau) - 1 + t / tau)  # W(t) = A(e^(-t/τ) - 1 + t/τ)

def fit_model(t, data, model_func, p0):
    """
    使用curve_fit拟合模型
    
    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        p0 (list): 初始参数猜测
        
    返回:
        tuple: 拟合参数及其协方差矩阵
    """
    # TODO: 使用scipy.optimize.curve_fit进行拟合 (1行代码)
    # [STUDENT_CODE_HERE]
    popt, pcov = curve_fit(model_func, t, data, p0=p0)  # 使用curve_fit进行拟合
    return popt, pcov

def plot_results(t, data, model_func, popt, title):
    """
    绘制实验数据与拟合曲线
    
    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        popt (numpy.ndarray): 拟合参数
        title (str): 图表标题
    """
    # TODO: 实现绘图功能 (约10行代码)
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(8, 6))
    plt.scatter(t, data, label='Experimentla Data', color='blue')  # 绘制实验数据点
    t_fit = np.linspace(min(t), max(t), 500)  # 生成拟合曲线的时间点
    plt.plot(t_fit, model_func(t_fit, *popt), label='Fitted Curve', color='red')  # 绘制拟合曲线
    plt.title(title)
    plt.xlabel('Time (t)')
    plt.ylabel('Enzyme Activity')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid()
    # 在图中标注拟合参数
    param_text = '\n'.join([f'{param} = {value:.3f}' for param, value in zip(['A', 'τ'][:len(popt)], popt)])
    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据
    data_dir = "https://github.com/Xianghua-YNU/cp2025-practices-week12-ftyy/tree/main/细菌生长实验数据拟合" # 请替换为你的数据目录
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")
    
    # 拟合V(t)模型
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0])
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f}")
    
    # 拟合W(t)模型
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0])
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f}, τ = {popt_W[1]:.3f}")
    
    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, 'V(t) Model Fit')
    plot_results(t_W, W_data, W_model, popt_W, 'W(t) Model Fit')
