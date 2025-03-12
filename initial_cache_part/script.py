import numpy as np
import torch
from initial_render import render
import matplotlib.pyplot as plt
from cache4d_utils import save_ply, generate_inscribed_ellipsoid, assign_random_colors
from gaussian_model import GaussianModel


class ViewpointCamera:
    def __init__(self, FoVx=60.0, FoVy=45.0, image_height=800, image_width=600, camera_center=(0.0, 0.0, 0.0)):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = image_height
        self.image_width = image_width
        self.camera_center = torch.tensor(camera_center, dtype=torch.float32, device="cuda")
        # 假设world_view_transform和full_proj_transform是4x4变换矩阵
        self.world_view_transform = torch.eye(4, device="cuda")
        self.full_proj_transform = torch.eye(4, device="cuda")
        self.image_name = "initial_view"


class Pipe:
    def __init__(self, debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=True):
        self.debug = debug
        self.antialiasing = antialiasing
        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python


def render_init_result():

    rect_min = [0.1, 0.1, 0.1]
    rect_max = [0.2, 0.6, 0.4]
    num_points = 20000

    # 生成点云
    points = generate_inscribed_ellipsoid(np.array(rect_min), np.array(rect_max), num_points)
    colors = assign_random_colors(num_points)

    # 保存为PLY文件
    output_ply = "initial_point_cloud.ply"
    save_ply(output_ply, points, colors, max_sh_degree=2)

    # 加载PLY文件到GaussianModel
    model = GaussianModel(sh_degree=2)
    model.load_ply(output_ply)

    # 设置相机参数
    viewpoint_camera = ViewpointCamera(
        FoVx=60.0, 
        FoVy=45.0, 
        image_height=800, 
        image_width=600, 
        camera_center=(0.0, 0.0, 5.0)  # 假设相机位于z=5的位置
    )

    # 设置渲染管道参数
    pipe = Pipe(
        debug=False,
        antialiasing=True,
        compute_cov3D_python=True,
        convert_SHs_python=True
    )

    # 设置背景颜色 (例如，黑色)
    bg_color = torch.zeros(3, device="cuda")  # RGB黑色

    # 调用渲染函数
    render_output = render(
        viewpoint_camera=viewpoint_camera,
        pc=model,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=1.0,
        separate_sh=False,
        override_color=None,
        use_trained_exp=False
    )

    # 获取渲染图像
    rendered_image = render_output["render"].detach().cpu().numpy()
    
    # 转换为可视化格式 (H, W, C)
    rendered_image = np.transpose(rendered_image, (1, 2, 0))

    # 可选：保存渲染图像
    plt.imsave("rendered_initial_point_cloud.png", rendered_image)
    print("已保存渲染图像到 rendered_initial_point_cloud.png")

if __name__ == "__main__":
    render_init_result()
