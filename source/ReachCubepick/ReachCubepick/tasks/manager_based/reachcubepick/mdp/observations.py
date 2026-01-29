from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """计算目标物体（Cube）在机器人基座坐标系下的相对位置"""
    robot: RigidObject = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 获取机器人的位置和旋转（四元数）
    robot_root_pos = robot.data.root_pos_w
    robot_root_quat = robot.data.root_quat_w
    
    # 获取目标的位置
    target_root_pos = target.data.root_pos_w
    
    # 1. 计算世界坐标系下的向量差
    diff_w = target_root_pos - robot_root_pos
    
    # 2. 将向量旋转到机器人的局部坐标系 (乘以机器人四元数的逆)
    # 这样即使机器人底座转动了，相对位置也能算对
    return math_utils.quat_apply(math_utils.quat_inv(robot_root_quat), diff_w)


def link_pos_in_robot_root_frame(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """计算机器人某个部件（如手）在机器人基座坐标系下的位置"""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取基座状态
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    
    # 获取末端执行器（手）的世界坐标
    # body_ids[0] 对应配置里写的 body_names=["panda_hand"]
    ee_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    
    # 计算相对位置
    diff_w = ee_pos_w - root_pos
    return math_utils.quat_apply(math_utils.quat_inv(root_quat), diff_w)