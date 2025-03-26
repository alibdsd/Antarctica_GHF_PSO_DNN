import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# [保留原始的shoot函数不变]
def shoot(lon, lat, azimuth, maxdist=None):
    """Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS = 0.00000000005
    if ((np.abs(np.cos(glat1)) < EPS) and not (np.abs(np.sin(faz)) < EPS)):
        print("Only N-S courses are meaningful, starting at a pole!")

    a = 6378.13 / 1.852
    f = 1 / 298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf == 0):
        b = 0.
    else:
        b = 2. * np.arctan2(tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs(y - c) > EPS):
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
             d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)

# 定义绘制圆形函数
def equi(m, centerlon, centerlat, radius, *args, **kwargs):
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])
    X, Y = m(X, Y)
    plt.plot(X, Y, **kwargs)

# 主可视化函数
def visualize_data_on_custom_map(csv_file, center, radius, n_points_per_dim=50):
    """
    在指定范围(90W-130W, 28N-55N)地图上绘制从CSV读取的数据点，颜色随heat flow变化
    参数:
    csv_file: str - CSV文件路径
    center: 元组 (longitude, latitude) - 圆心坐标
    radius: float - 半径（单位：km）
    n_points_per_dim: int - 用于网格参考的点数（可选）
    """
    # 读取CSV文件
    try:
        data = pd.read_csv(csv_file)
        # 确保必要的列存在
        required_columns = ['lon', 'lat', 'heat flow']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please check the path.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

    # 设置正方形画布
    plt.figure(figsize=(12, 12))  # 正方形画布
    
    # 创建Basemap实例，范围为 130W-90W, 28N-55N
    center_lon, center_lat = center
    m = Basemap(projection='merc',
                resolution='l',
                llcrnrlon=-127,  # 西侧 130°W
                llcrnrlat=30,    # 南侧 28°N
                urcrnrlon=-100,   # 东侧 90°W
                urcrnrlat=50)    # 北侧 55°N
    
    # 绘制地图特征
    m.drawlsmask(land_color="#ffffff", ocean_color="#e8f4f8", resolution='l')
    m.drawcoastlines(linewidth=0.5)
    # 经纬度网格
    m.drawparallels(np.arange(28, 56, 5.), labels=[1,0,0,0], linewidth=0.5, fontsize=18)
    m.drawmeridians(np.arange(-130, -89, 5.), labels=[0,0,0,1], linewidth=0.5, fontsize=18)
    
    # 提取数据并转换为地图坐标
    lons = data['lon'].values
    lats = data['lat'].values
    heat_flow = data['heat flow'].values
    
    # 过滤数据，确保在指定范围内
    mask = (lons >= -130) & (lons <= -90) & (lats >= 28) & (lats <= 55)
    lons = lons[mask]
    lats = lats[mask]
    heat_flow = heat_flow[mask]
    
    x, y = m(lons, lats)
    
    # 使用颜色映射绘制点，颜色随heat flow变化
    vmin = np.min(heat_flow) if len(heat_flow) > 0 else 20
    vmax = np.max(heat_flow) if len(heat_flow) > 0 else 150
    scat = m.scatter(x, y, c=heat_flow, s=30, cmap='jet', vmin=vmin, vmax=vmax, alpha=0.7, edgecolor='none')
    
    # 绘制圆形
    equi(m, center_lon, center_lat, radius,
         lw=1, linestyle='-', color='black')
    
    # 标记圆心
    x_center, y_center = m(center_lon, center_lat)
    m.plot(x_center, y_center, 'ko', markersize=4, label='Center')
    
    # 添加颜色条
    cbar = plt.colorbar(scat, label='Heat Flux (mW/m²)', orientation='horizontal', pad=0.05)
    cbar.set_label('Heat Flux (mW/m²)', fontsize=18)  # 设置颜色条标签的字体大小
    cbar.ax.tick_params(labelsize=18)  # 设置颜色条刻度字体大小
    
    # 设置图例
    plt.legend(loc='upper right', fontsize=18)  # 将图例放置在右上角
    
    # 设置标题
    plt.title(f'GHF validation set with Circle Radius {radius} km', fontsize=19, pad=10)
    
    # 加粗画布四周边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 设置边框粗细
    
    # 调整画布边距
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('data_map_with_circle.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例运行
if __name__ == "__main__":
    # 设置CSV文件路径（请替换为你的实际文件路径）
    csv_file = 'World_train_data.csv'  # 替换为你的CSV文件路径
    # 设置圆心（位于指定范围中心）
    center = (-113, 40.5)  # 中心点，位于范围 (-130, -90, 28, 55) 内部
    radius = 700  # km（适合范围）
    
    visualize_data_on_custom_map(csv_file, center, radius, n_points_per_dim=50)
