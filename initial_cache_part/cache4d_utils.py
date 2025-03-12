import torch
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os

def inverse_sigmoid(y):
    return torch.log(y / (1 - y))

def generate_inscribed_ellipsoid(rect_min, rect_max, num_points):
    """
    生成内切椭球体上的均匀点。
    """
    # 计算椭球体的中心和半径
    center = (rect_min + rect_max) / 2.0
    radii = (rect_max - rect_min) / 2.0

    # 使用均匀采样生成单位球面上的点
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # 缩放到椭球体
    x *= radii[0]
    y *= radii[1]
    z *= radii[2]

    # 平移到中心
    x += center[0]
    y += center[1]
    z += center[2]

    points = np.stack((x, y, z), axis=1)

    return points

def assign_random_colors(num_points):
    """
    为每个点赋予随机RGB颜色。
    """
    colors = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)
    return colors

def save_ply(filename, points, colors, max_sh_degree=2):
    """
    将点云数据保存为PLY文件，包括位置、颜色和GaussianModel所需的其他属性。
    """
    num_points = points.shape[0]
    num_f_dc = 3  # RGB对应的f_dc_*属性
    num_f_rest = 3 * (max_sh_degree + 1) ** 2 - num_f_dc  # 其他f_rest_*属性

    # 初始化所有属性
    data_dict = {
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'nx': np.zeros(num_points, dtype=np.float32),
        'ny': np.zeros(num_points, dtype=np.float32),
        'nz': np.zeros(num_points, dtype=np.float32),
    }

    # f_dc_*属性：简单地将RGB颜色作为前3个f_dc_*系数
    for i in range(num_f_dc):
        data_dict[f'f_dc_{i}'] = (colors[:, i].astype(np.float32) / 255.0)  # 归一化颜色

    # f_rest_*属性：设置为零
    for i in range(num_f_rest):
        data_dict[f'f_rest_{i}'] = np.zeros(num_points, dtype=np.float32)

    # opacity属性：逆Sigmoid(0.1)
    opacity_value = inverse_sigmoid(torch.tensor(0.1)).item()
    data_dict['opacity'] = np.full(num_points, opacity_value, dtype=np.float32)

    # scale_*属性：log(distance) - 为x, y, z方向各一个
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale_values = np.log(distances + 1e-8)  # 防止log(0)
    for i in range(3):  # 保存scale_0, scale_1, scale_2
        data_dict[f'scale_{i}'] = scale_values.astype(np.float32)

    # rot_*属性：单位四元数 [1, 0, 0, 0]
    for i in range(4):
        rot_values = np.ones(num_points, dtype=np.float32) if i == 0 else np.zeros(num_points, dtype=np.float32)
        data_dict[f'rot_{i}'] = rot_values

    # 将所有属性组合成结构化数组
    dtype = []
    for key in data_dict:
        if key.startswith('f_dc_') or key.startswith('f_rest_') or key.startswith('scale_') or key.startswith('rot_'):
            dtype.append((key, 'f4'))
        elif key in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
            dtype.append((key, 'f4'))
        elif key == 'opacity':
            dtype.append((key, 'f4'))
        else:
            raise ValueError(f"Unknown attribute: {key}")

    structured_array = np.empty(num_points, dtype=dtype)
    for key in data_dict:
        structured_array[key] = data_dict[key]

    # 定义PLY元素
    el = PlyElement.describe(structured_array, 'vertex')

    # 保存为PLY文件
    PlyData([el], text=True).write(filename)
    print(f"已保存PLY文件，包含{num_points}个点到 {filename}")


def main():
    parser = argparse.ArgumentParser(description='初始化GaussianModel的点云数据。')
    parser.add_argument('--output', type=str, default='initial_point_cloud.ply', help='输出PLY文件路径。')
    parser.add_argument('--num_points', type=int, default=20000, help='生成的点数量。')
    parser.add_argument('--rect_min', type=float, nargs=3, default=[-1.0, -1.0, -1.0], help='矩形的最小坐标 (x, y, z)。')
    parser.add_argument('--rect_max', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='矩形的最大坐标 (x, y, z)。')
    parser.add_argument('--sh_degree', type=int, default=2, help='球谐函数的最大阶数。')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

    # 生成点云
    points = generate_inscribed_ellipsoid(np.array(args.rect_min), np.array(args.rect_max), args.num_points)
    colors = assign_random_colors(args.num_points)

    # 保存为PLY文件
    save_ply(args.output, points, colors, max_sh_degree=args.sh_degree)

if __name__ == '__main__':
    main()


"""
python 4d_cache_utils.py --output my_point_cloud.ply --num_points 30000
python 4d_cache_utils.py --rect_min -2.0 -1.0 -1.5 --rect_max 2.0 1.0 1.5
"""
