import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_sequence_length_cdf(sequence_map, save_path='sequence_length_cdf.png'):
    """
    根据给定的序列长度 Map 绘制累积分布图 (CDF) 并保存为文件。
    
    参数:
    sequence_map (dict): 键为序列长度 (int)，值为该长度下序列数量 (int) 的字典。
    save_path (str): 保存图像的文件路径，默认为 'sequence_length_cdf.png'。
    """
    # 提取序列长度并排序
    lengths = sorted(sequence_map.keys())
    # print(lengths)
    # 根据排序后的长度获取对应的数量
    counts = [sequence_map[length] for length in lengths]
    
    # 计算总序列数
    total = sum(counts)
    
    # 计算累积分布
    cumulative_counts = np.cumsum(counts)
    cdf = cumulative_counts / total  # 累积比例（0到1之间）
    
    # 创建图形
    fig, ax = plt.subplots()
    
    # 绘制CDF曲线
    ax.plot(lengths, cdf, color='black', linewidth=2)
    
    # 填充曲线下侧区域为黑色
    ax.fill_between(lengths, cdf, color='black', alpha=0.8)
    
    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 将 0-1 转换为 0%-100%
    
    # 设置标签
    ax.set_xlabel('Sequence Lengths')
    ax.set_ylabel('Cumulative Distribution')
    
    # 添加网格
    ax.grid(True, which="both", ls="--", alpha=0.7)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_sequence_length_distribution(sequence_map, save_path='sequence_length_distribution.png'):

        
        # 提取序列长度并排序
        lengths = sorted(sequence_map.keys())
        # print(lengths)
       
        # print(lengths)

        # print(lengths)
        # 根据排序后的长度获取对应的数量
        counts = [sequence_map[length] for length in lengths]

        # 确保输入列表长度一致
        if len(lengths) != len(counts):
            raise ValueError("lengths 和 counts 的长度必须一致！")

        # 计算总序列数量
        total = sum(counts)
        
        # 计算每个序列长度的百分比
        percentages = [count / total * 100 for count in counts]  # 百分比（0% 到 100%）
        # print(percentages)
        # 创建图形
        fig, ax = plt.subplots()
        
        # 绘制分布图（折线图）
        ax.plot(lengths, percentages, color='black', linewidth=2)
        
        # 填充曲线下侧区域为黑色
        ax.fill_between(lengths, percentages, color='green', alpha=0.8)
        
        # 设置 Y 轴为百分比格式
        ax.yaxis.set_major_formatter(PercentFormatter())
        
        # 设置标签
        ax.set_xlabel('Sequence Lengths')
        ax.set_ylabel('Percentage of Sequences')
        
        # 添加网格
        ax.grid(True, which="both", ls="--", alpha=0.7)
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()