import torch
import copy
from torch import nn
from scene.gaussian_model import GaussianModel


def copy_gaussian_model(original_model, indices, hyperparam):
    """
    根据提供的索引，将原始 GaussianModel 中指定的点复制到一个新的 GaussianModel 实例中。

    Args:
        original_model (GaussianModel): 要复制的原始模型实例。
        indices (list 或 torch.Tensor): 要复制的点的索引。
        args: 初始化 GaussianModel 所需的参数（包括 training_args）。

    Returns:
        GaussianModel: 包含指定点的新 GaussianModel 实例。
    """
    # 确保 indices 是一个 1D 的 LongTensor，并且在与原模型相同的设备上
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.long, device=original_model._xyz.device)
    elif isinstance(indices, torch.Tensor):
        indices = indices.long().to(original_model._xyz.device)
    else:
        raise TypeError("indices 必须是 list 或 torch.Tensor 类型")

    # 初始化新的 GaussianModel 实例
    new_model = GaussianModel(original_model.max_sh_degree, hyperparam)

    # 复制激活的 SH 度数
    new_model.active_sh_degree = original_model.active_sh_degree

    # 深拷贝变形网络，确保新模型拥有独立的网络参数
    new_model._deformation = copy.deepcopy(original_model._deformation)
    new_model._deformation_table = original_model._deformation_table[indices].clone().detach()

    # 复制点云数据并创建新的 nn.Parameter
    new_model._xyz = nn.Parameter(original_model._xyz[indices].clone().detach(), requires_grad=True)
    new_model._features_dc = nn.Parameter(original_model._features_dc[indices].clone().detach(), requires_grad=True)
    new_model._features_rest = nn.Parameter(original_model._features_rest[indices].clone().detach(), requires_grad=True)
    new_model._scaling = nn.Parameter(original_model._scaling[indices].clone().detach(), requires_grad=True)
    new_model._rotation = nn.Parameter(original_model._rotation[indices].clone().detach(), requires_grad=True)
    new_model._opacity = nn.Parameter(original_model._opacity[indices].clone().detach(), requires_grad=True)

    # 复制其他属性
    new_model.max_radii2D = original_model.max_radii2D[indices].clone().detach()
    new_model.xyz_gradient_accum = original_model.xyz_gradient_accum.clone().detach()
    new_model.denom = original_model.denom.clone().detach()

    # 初始化新的模型的函数和优化器
    new_model.setup_functions()
    # new_model.training_setup(args.training_args)  # 假设 args 包含 training_args

    # # 如果需要，可以复制优化器的状态（可选）
    # if original_model.optimizer is not None:
    #     # 深拷贝优化器的参数组
    #     new_optimizer_state = copy.deepcopy(original_model.optimizer.state_dict())
    #     new_model.optimizer.load_state_dict(new_optimizer_state)

    return new_model


def merge_gaussian_models(gs_model1, gs_model2, hyperparam):
    """
    将两个 GaussianModel 实例合并为一个新的 GaussianModel 实例，使用 gs_model1 的 deformation_network。

    Args:
        gs_model1 (GaussianModel): 第一个 GaussianModel 实例。
        gs_model2 (GaussianModel): 第二个 GaussianModel 实例。
        hyperparam: 初始化新的 GaussianModel 所需的超参数。

    Returns:
        GaussianModel: 合并后的 GaussianModel 实例。
    """
    # 检查两个模型的兼容性
    if gs_model1.max_sh_degree != gs_model2.max_sh_degree:
        raise ValueError("两个模型的 max_sh_degree 必须相同。")
    
    # 初始化新的 GaussianModel 实例
    new_model = GaussianModel(gs_model1.max_sh_degree, hyperparam)
    
    # 复制激活的 SH 度数（假设两个模型相同）
    new_model.active_sh_degree = gs_model1.active_sh_degree  # 或根据需要处理

    # 使用 gs_model1 的 deformation_network
    new_model._deformation = copy.deepcopy(gs_model1._deformation)
    
    # 合并 deformation_table
    new_deformation_table = torch.cat([gs_model1._deformation_table, gs_model2._deformation_table], dim=0).clone().detach()
    new_model._deformation_table = nn.Parameter(new_deformation_table, requires_grad=False)
    
    # 合并点云数据
    new_model._xyz = nn.Parameter(torch.cat([gs_model1._xyz, gs_model2._xyz], dim=0).clone().detach(), requires_grad=True)
    new_model._features_dc = nn.Parameter(torch.cat([gs_model1._features_dc, gs_model2._features_dc], dim=0).clone().detach(), requires_grad=True)
    new_model._features_rest = nn.Parameter(torch.cat([gs_model1._features_rest, gs_model2._features_rest], dim=0).clone().detach(), requires_grad=True)
    new_model._scaling = nn.Parameter(torch.cat([gs_model1._scaling, gs_model2._scaling], dim=0).clone().detach(), requires_grad=True)
    new_model._rotation = nn.Parameter(torch.cat([gs_model1._rotation, gs_model2._rotation], dim=0).clone().detach(), requires_grad=True)
    new_model._opacity = nn.Parameter(torch.cat([gs_model1._opacity, gs_model2._opacity], dim=0).clone().detach(), requires_grad=True)
    
    # 合并其他属性
    new_model.max_radii2D = torch.cat([gs_model1.max_radii2D, gs_model2.max_radii2D], dim=0).clone().detach()
    new_model.xyz_gradient_accum = gs_model1.xyz_gradient_accum.clone().detach()
    new_model.denom = gs_model1.denom.clone().detach()
    
    # 初始化新的模型的函数
    new_model.setup_functions()
    
    # # 如果需要，可以复制优化器的状态（可选）
    # if gs_model1.optimizer is not None and gs_model2.optimizer is not None:
    #     # 这里假设两个优化器的参数组相同，具体情况需要根据实际优化器类型处理
    #     new_optimizer_state = copy.deepcopy(gs_model1.optimizer.state_dict())
    #     new_model.optimizer.load_state_dict(new_optimizer_state)
    #     # 注意：如果两个模型的优化器有不同的状态，这里需要进一步处理
    # elif gs_model1.optimizer is not None:
    #     # 仅复制 gs_model1 的优化器状态
    #     new_optimizer_state = copy.deepcopy(gs_model1.optimizer.state_dict())
    #     new_model.optimizer.load_state_dict(new_optimizer_state)
    # elif gs_model2.optimizer is not None:
    #     # 仅复制 gs_model2 的优化器状态
    #     new_optimizer_state = copy.deepcopy(gs_model2.optimizer.state_dict())
    #     new_model.optimizer.load_state_dict(new_optimizer_state)
    # else:
    #     # 如果两个模型都没有优化器，则无需处理
    #     pass
    
    return new_model


def move_indices_to_end(model, indices):
    """
    将 GaussianModel 实例中指定的点索引移动到最后。

    Args:
        model (GaussianModel): 要修改的 GaussianModel 实例。
        indices (list 或 torch.Tensor): 要移动的点的索引。
    """
    # 确保 indices 是 1D 的 LongTensor，并且与模型相同设备
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.long, device=model._xyz.device)
    elif isinstance(indices, torch.Tensor):
        indices = indices.long().to(model._xyz.device)
    else:
        raise TypeError("indices 必须是 list 或 torch.Tensor 类型")

    total_points = model._xyz.shape[0]
    all_indices = torch.arange(total_points, device=model._xyz.device)

    # 创建新的索引顺序：先是未选择的点，再是选择的点
    mask = torch.ones(total_points, dtype=torch.bool, device=model._xyz.device)
    mask[indices] = False
    non_selected_indices = all_indices[mask]
    selected_indices = all_indices[indices]
    new_order = torch.cat((non_selected_indices, selected_indices), dim=0)

    # 重新排序所有相关的张量
    def reorder_tensor(tensor):
        if tensor.dim() == 1:
            return tensor[new_order].clone().detach()
        else:
            return tensor[new_order].clone().detach()

    model._xyz.data = reorder_tensor(model._xyz)
    model._features_dc.data = reorder_tensor(model._features_dc)
    model._features_rest.data = reorder_tensor(model._features_rest)
    model._scaling.data = reorder_tensor(model._scaling)
    model._rotation.data = reorder_tensor(model._rotation)
    model._opacity.data = reorder_tensor(model._opacity)
    model.max_radii2D = reorder_tensor(model.max_radii2D)
    # model.xyz_gradient_accum = reorder_tensor(model.xyz_gradient_accum)
    # model.denom = reorder_tensor(model.denom)
    model._deformation_table = reorder_tensor(model._deformation_table)

    # 更新优化器中的参数顺序
    if model.optimizer is not None:
        for group in model.optimizer.param_groups:
            for i, param in enumerate(group['params']):
                # 重新排序参数数据
                param.data = reorder_tensor(param.data)
                
                # 如果优化器有动量等状态，也需要相应地重新排序
                state = model.optimizer.state.get(param)
                if state is not None:
                    if 'exp_avg' in state:
                        state['exp_avg'] = reorder_tensor(state['exp_avg'])
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'] = reorder_tensor(state['exp_avg_sq'])

    # 清理 CUDA 缓存以优化内存使用
    torch.cuda.empty_cache()
