# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def position_target_asset_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    target_pos_w = target_asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    return torch.norm(curr_pos_w - target_pos_w, dim=1)

def object_approach_reward(
    env, asset_cfg: SceneEntityCfg, target_asset_cfg: SceneEntityCfg, std: float = 0.5
) -> torch.Tensor:
    # 1. 获取机器人手部位置
    asset = env.scene[asset_cfg.name]
    # 注意：这里假设 body_ids[0] 是正确的，通常需要确保 cfg 里 body_names 填对
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    
    # 2. 获取方块位置
    target_asset = env.scene[target_asset_cfg.name]
    target_pos_w = target_asset.data.root_pose_w[:, :3] # 直接取 root pose 通常更稳
    
    # 3. 计算欧氏距离
    distance = torch.norm(curr_pos_w - target_pos_w, dim=1)
    
    # 4. 转换为 [0, 1] 的奖励 (Log 形式或 tanh 形式)
    # 使用 1 / (1 + dist^2) 形式，std 控制宽容度
    reward = 1.0 / (1.0 + (distance ** 2) / (std ** 2))
    
    return reward

def object_is_lifted(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """如果方块高度超过阈值，返回 1.0，否则返回 0.0"""
    # 获取方块状态
    obj_root_pose = env.scene[asset_cfg.name].data.root_pose_w
    # obj_root_pose[:, 2] 是 Z 轴高度
    is_lifted = torch.where(obj_root_pose[:, 2] > minimum_height, 1.0, 0.0)
    return is_lifted

def reaching_reward_shaping(env, std: float, asset_cfg: SceneEntityCfg, target_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """使用 tanh 核函数计算距离奖励，范围 [0, 1]，防止梯度爆炸"""
    # 获取末端执行器和方块的位置
    ee_pos = env.scene.rigid_bodies[asset_cfg.name].data.root_pose_w[:, 0:3]
    # 注意：如果 robot 是 Articulation，通常用 bodies 索引，这里简化处理
    # 更严谨的写法需通过 asset_cfg.body_ids 获取特定 link 位置
    
    # 获取目标位置
    target_pos = env.scene.rigid_bodies[target_asset_cfg.name].data.root_pose_w[:, 0:3]
    
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    # 距离越近，奖励越接近 1
    return 1.0 / (1.0 + (distance ** 2) / std)

def gripper_vertical_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 1. 获取夹爪刚体对象
    asset = env.scene[asset_cfg.name]
    
    # 2. 获取夹爪在世界坐标系下的四元数 (Orientation)
    # body_ids[0] 对应你传入的 body_names (即 panda_hand)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    
    # 3. 计算夹爪的 Z 轴方向向量
    # 对于 Franka Panda，夹爪的 Z 轴通常是手指伸出的方向
    # 我们定义一个局部 Z 轴向量 (0, 0, 1)
    vec_z = torch.zeros((env.num_envs, 3), device=env.device)
    vec_z[:, 2] = 1.0
    
    # 将局部 Z 轴旋转到世界坐标系，得到当前夹爪指向
    gripper_dir = quat_apply(curr_quat_w, vec_z)
    
    # 4. 计算与“垂直向下” (0, 0, -1) 的相似度
    # 目标向量: (0, 0, -1)
    target_dir = torch.zeros_like(vec_z)
    target_dir[:, 2] = -1.0
    
    # 计算点积 (Dot Product): 范围 [-1, 1]
    # 1.0 表示完全垂直向下，-1.0 表示完全朝上
    dot_prod = torch.sum(gripper_dir * target_dir, dim=1)
    
    # 5. 映射到奖励
    # 只要点积大于 0 (大致朝下) 就给奖励，越垂直奖励越高
    # 使用 torch.clamp 把负值截断为 0，防止惩罚
    return torch.clamp(dot_prod, min=0.0)

def object_height_continuous_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float
) -> torch.Tensor:
    """
    连续的高度奖励：引导机器人把方块举到目标高度。
    """
    # 1. 获取方块当前位置
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.root_state_w[:, :3]
    
    # 2. 获取方块的高度 (Z轴)
    curr_height = curr_pos_w[:, 2]
    
    # 3. 计算奖励：使用 Tanh 函数，让高度越接近 target_height 分数越高
    # 比如：当前高度 0.02 -> 0分；高度 0.3 -> 接近1分
    # 这里的 distance 是 "当前高度" 与 "目标高度" 的差
    distance = torch.abs(target_height - curr_height)
    
    # 我们希望 distance 越小越好 (越接近目标高度)
    return 1.0 - torch.tanh(distance / std)

def object_keep_xy_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_pos_xy: tuple[float, float] = (0.5, 0.0) # 这里可以设为初始生成的中心区域
) -> torch.Tensor:
    """
    水平位移惩罚：防止机器人在举起过程中乱甩方块。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_xy = asset.data.root_state_w[:, :2] # 只取 X 和 Y
    
    # 目标 XY (例如方块生成的平均中心)
    target = torch.tensor(target_pos_xy, device=env.device).repeat(env.num_envs, 1)
    
    # 计算当前 XY 和目标 XY 的距离
    error = torch.norm(curr_pos_xy - target, dim=1)
    
    # 返回距离作为惩罚 (因为 weight 会设为负数，所以这里返回正的距离)
    return error

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    # 获取资产（机器人）
    asset: Articulation = env.scene[asset_cfg.name]

    # [关键] 只获取配置中指定的关节 (例如只选 panda_joint.*，排除夹爪)
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    
    # 计算功率 P = sum(|v * tau|)
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)