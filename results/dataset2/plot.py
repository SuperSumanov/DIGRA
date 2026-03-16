import matplotlib.pyplot as plt
import numpy as np
import re

# 解析数据
def parse_data(filename, content):
    data = {}
    current_threshold = None
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('----------'):
            # 提取阈值
            threshold = float(line.strip('-').split('---')[-1])
            current_threshold = threshold
            data[current_threshold] = {'ef': [], 'recall': [], 'time': [], 'qps': []}
            i += 1
        elif line.startswith('ef:'):
            ef = int(line.split(':')[1])
            recall = float(lines[i+1].split(':')[1])
            time = float(lines[i+2].split(':')[1])
            qps = float(lines[i+3].split(':')[1])
            
            if current_threshold is not None:
                data[current_threshold]['ef'].append(ef)
                data[current_threshold]['recall'].append(recall)
                data[current_threshold]['time'].append(time)
                data[current_threshold]['qps'].append(qps)
            i += 4
        else:
            i += 1
    
    return data

# 读取文件内容
with open('cluster.txt', 'r') as f:
    cluster_content = f.read()

with open('origin.txt', 'r') as f:
    origin_content = f.read()

# 解析数据
cluster_data = parse_data('cluster.txt', cluster_content)
origin_data = parse_data('origin.txt', origin_content)

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('HNSW Performance Comparison: Cluster vs Origin', fontsize=16, fontweight='bold')

# 阈值列表
thresholds = [0.01, 0.05, 0.1, 0.2, 0.4]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 为每个阈值创建子图
for idx, threshold in enumerate(thresholds):
    row = idx // 2
    col = idx % 2
    
    ax = axes[row, col]
    
    if threshold in cluster_data and threshold in origin_data:
        cluster_ef = cluster_data[threshold]['ef']
        cluster_qps = cluster_data[threshold]['qps']
        
        origin_ef = origin_data[threshold]['ef']
        origin_qps = origin_data[threshold]['qps']
        
        # 绘制折线图
        ax.plot(cluster_ef, cluster_qps, 'o-', color=colors[0], linewidth=2, markersize=4, 
                label=f'Cluster (threshold={threshold})', alpha=0.8)
        ax.plot(origin_ef, origin_qps, 's-', color=colors[1], linewidth=2, markersize=4, 
                label=f'Origin (threshold={threshold})', alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(f'Recall Threshold = {threshold}', fontsize=12, fontweight='bold')
        ax.set_xlabel('EF Parameter', fontsize=10)
        ax.set_ylabel('QPS (Queries Per Second)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 设置y轴为对数刻度以更好地显示差异
        ax.set_yscale('log')
        
        # 添加性能对比注释
        avg_cluster_qps = np.mean(cluster_qps)
        avg_origin_qps = np.mean(origin_qps)
        improvement = (avg_cluster_qps - avg_origin_qps) / avg_origin_qps * 100
        
        ax.text(0.02, 0.98, f'Avg QPS Improvement: {improvement:.1f}%', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 创建汇总图表
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Performance Summary: Average QPS by Threshold', fontsize=14, fontweight='bold')

# 左图：平均QPS对比
threshold_labels = [str(t) for t in thresholds]
avg_cluster_qps_list = []
avg_origin_qps_list = []

for threshold in thresholds:
    if threshold in cluster_data and threshold in origin_data:
        avg_cluster_qps_list.append(np.mean(cluster_data[threshold]['qps']))
        avg_origin_qps_list.append(np.mean(origin_data[threshold]['qps']))

x = np.arange(len(threshold_labels))
width = 0.35

ax2[0].bar(x - width/2, avg_cluster_qps_list, width, label='Cluster', color=colors[0], alpha=0.8)
ax2[0].bar(x + width/2, avg_origin_qps_list, width, label='Origin', color=colors[1], alpha=0.8)
ax2[0].set_xlabel('Recall Threshold')
ax2[0].set_ylabel('Average QPS')
ax2[0].set_title('Average QPS by Threshold')
ax2[0].set_xticks(x)
ax2[0].set_xticklabels(threshold_labels)
ax2[0].legend()
ax2[0].grid(True, alpha=0.3)

# 右图：性能提升百分比
improvements = [(c - o) / o * 100 for c, o in zip(avg_cluster_qps_list, avg_origin_qps_list)]
colors_bar = ['green' if imp > 0 else 'red' for imp in improvements]
ax2[1].bar(threshold_labels, improvements, color=colors_bar, alpha=0.7)
ax2[1].set_xlabel('Recall Threshold')
ax2[1].set_ylabel('Performance Improvement (%)')
ax2[1].set_title('QPS Improvement (Cluster vs Origin)')
ax2[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2[1].grid(True, alpha=0.3)

# 添加数值标签
for i, v in enumerate(improvements):
    ax2[1].text(i, v + (5 if v > 0 else -10), f'{v:.1f}%', 
                ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# 打印详细统计信息
print("=" * 80)
print("DETAILED PERFORMANCE COMPARISON")
print("=" * 80)

for threshold in thresholds:
    print(f"\nThreshold = {threshold}")
    print("-" * 60)
    print(f"{'EF':<10} {'Cluster QPS':<15} {'Origin QPS':<15} {'Difference':<15} {'Improvement %':<15}")
    print("-" * 60)
    
    cluster_ef = cluster_data[threshold]['ef']
    cluster_qps = cluster_data[threshold]['qps']
    origin_qps = origin_data[threshold]['qps']
    
    for i in range(len(cluster_ef)):
        diff = cluster_qps[i] - origin_qps[i]
        imp = (cluster_qps[i] - origin_qps[i]) / origin_qps[i] * 100
        print(f"{cluster_ef[i]:<10} {cluster_qps[i]:<15.2f} {origin_qps[i]:<15.2f} {diff:<15.2f} {imp:<15.2f}")
    
    avg_imp = (np.mean(cluster_qps) - np.mean(origin_qps)) / np.mean(origin_qps) * 100
    print("-" * 60)
    print(f"{'AVERAGE':<10} {np.mean(cluster_qps):<15.2f} {np.mean(origin_qps):<15.2f} {(np.mean(cluster_qps)-np.mean(origin_qps)):<15.2f} {avg_imp:<15.2f}")