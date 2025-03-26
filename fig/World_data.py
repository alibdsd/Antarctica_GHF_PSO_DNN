

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 主可视化函数（平面投影，加大label字体到18号）
def visualize_global_ghf(csv_file, n_points_per_dim=50):
    """
    在全球平面地图上绘制从CSV读取的地热通量（GHF）分布，颜色随heat flow变化
    参数:
    csv_file: str - CSV文件路径
    n_points_per_dim: int - 用于网格参考的点数（可选）
    """
    # 读取CSV文件
    try:
        data = pd.read_csv(csv_file)
        required_columns = ['lon', 'lat', 'heat flow']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please check the path.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

    # 设置画布
    plt.figure(figsize=(15, 8))  # 适合全球平面的比例
    
    # 创建全球范围的Basemap实例，使用等矩形投影（平面）
    m = Basemap(projection='cyl',  # 使用圆柱投影，平面显示
                resolution='l',
                llcrnrlon=-180,  # 西侧 -180°
                llcrnrlat=-90,   # 南侧 -90°
                urcrnrlon=180,   # 东侧 180°
                urcrnrlat=90)    # 北侧 90°
    
    # 绘制地图特征
    m.drawlsmask(land_color="#ffffff", ocean_color="#e8f4f8", resolution='l')
    m.drawcoastlines(linewidth=0.5)
    # 经纬度网格，字体大小设为18
    m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0], linewidth=0.5, fontsize=16)
    m.drawmeridians(np.arange(-180, 181, 30), labels=[0,0,0,1], linewidth=0.5, fontsize=16)
    
    # 提取数据并转换为地图坐标
    lons = data['lon'].values
    lats = data['lat'].values
    heat_flow = data['heat flow'].values
    
    # 过滤数据，确保在全球范围内
    mask = (lons >= -180) & (lons <= 180) & (lats >= -90) & (lats <= 90)
    lons = lons[mask]
    lats = lats[mask]
    heat_flow = heat_flow[mask]
    
    x, y = m(lons, lats)
    
    # 使用颜色映射绘制点，颜色随heat flow变化
    vmin = np.min(heat_flow) if len(heat_flow) > 0 else 20
    vmax = np.max(heat_flow) if len(heat_flow) > 0 else 150
    scat = m.scatter(x, y, c=heat_flow, s=10, cmap='jet', vmin=vmin, vmax=vmax, alpha=0.7, edgecolor='none')
    
    # 添加颜色条，字体大小设为18
    cbar = plt.colorbar(scat, label='Heat Flow (mW/m²)', orientation='horizontal', pad=0.1)
    cbar.set_label('Heat Flow (mW/m²)', fontsize=18)
    cbar.ax.tick_params(labelsize=18)  # 颜色条刻度字体大小
    
    # 设置标题，字体大小设为18
    plt.title('Global GHF Distribution', fontsize=18, pad=10)
    
    # 加粗画布四周边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 调整画布边距
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('global_ghf_map_flat.png', dpi=300, bbox_inches='tight')
    plt.show()

# 单独绘制GHF分布直方图
def plot_ghf_histogram(csv_file):
    """
    绘制GHF分布的直方图，颜色与地图的jet颜色映射一致
    参数:
    csv_file: str - CSV文件路径
    """
    # 读取CSV文件
    try:
        data = pd.read_csv(csv_file)
        required_columns = ['lon', 'lat', 'heat flow']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please check the path.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

    # 提取热通量数据
    heat_flow = data['heat flow'].values
    
    # 过滤数据，确保在合理范围内
    mask = (heat_flow >= 0) & (heat_flow <= 300)  # 与地图的颜色范围一致
    heat_flow = heat_flow[mask]
    
    # 创建画布
    plt.figure(figsize=(10, 6))  # 单独直方图的画布大小
    
    # 绘制直方图，先使用默认颜色
    hist, bins, patches = plt.hist(heat_flow, bins=50, range=(0, 200), density=True, edgecolor='none')
    
    # 使用jet颜色映射为直方图柱子着色
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=200)  # 与地图的颜色范围一致
    bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个bin的中心值
    
    # 为每个柱子设置颜色
    for i, patch in enumerate(patches):
        color = cmap(norm(bin_centers[i]))
        patch.set_facecolor(color)
    
    # 设置直方图的标签和标题，字体大小为18
    #plt.xlabel('Heat Flow (mW/m²)', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('GHF Distribution Histogram', fontsize=18,pad=10)
    plt.tick_params(axis='both', labelsize=18)  # 刻度字体大小
    
    # 加粗画布四周边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('ghf_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例运行
if __name__ == "__main__":
    # 设置CSV文件路径（请替换为你的实际文件路径）
    csv_file = 'GHF_continent.csv'  # 替换为你的CSV文件路径
    
    # 调用函数，绘制全球GHF分布
    visualize_global_ghf(csv_file, n_points_per_dim=50)
    plot_ghf_histogram(csv_file)




