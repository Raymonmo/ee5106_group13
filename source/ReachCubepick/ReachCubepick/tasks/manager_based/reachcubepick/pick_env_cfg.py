# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
import torch

import isaaclab.sim as sim_utils
import isaaclab.assets
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from . import mdp

# [修改 1] 导入 Franka 配置
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

##
# Scene definition
##

ENV_SPACING = 2.5

def get_random_translation():
    # 稍微调整了生成范围，适配 Franka 的工作空间
    x = random.uniform(0.3, 0.5)
    y = random.uniform(-0.2, 0.2)
    z = 0.025
    return (x, y, z)

@configclass
class ReachcubepickSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    _robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 2. 强制开启接触传感器支持 (PhysX 需要这个标志才会上报碰撞数据)
    _robot_cfg.spawn.activate_contact_sensors = True
    
    # [修改 2] 替换为 Franka Panda 机器人
    robot = _robot_cfg
    
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.04), # 稍微调小一点，方便夹取
            mass_props=sim_utils.schemas.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # 给个绿色方便看
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=get_random_translation())
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_(link[1-9]|hand).*", 
        history_length=3, 
        track_air_time=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,      # 亮度，觉得不够亮可以调到 3000.0
            color=(1.0, 1.0, 1.0), # 白光
            # texture_file=...     # 如果你想用HDR图片做背景（比如蓝天白云），可以在这里指定路径
        )
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=2500.0       # 太阳强度
        ),
        # 设置太阳的角度，让阴影好看一点
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

##
# MDP settings
##

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        target_object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, 
            params={"target_cfg": SceneEntityCfg("cube"), "robot_cfg": SceneEntityCfg("robot")}
        )

        ee_pos = ObsTerm(
            func=mdp.link_pos_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])}
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    # [修改 3] Franka 手臂控制 (7个自由度)
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"], # 使用正则匹配所有 7 个关节
        scale=0.5,                     # 稍微降低一点动作缩放，训练更稳定
        use_default_offset=True,
    )

    # [修改 4] 新增夹爪控制 (1个自由度，控制开合)
    # BinaryJointPositionActionCfg 会把输出映射为 "开" 或 "关"
    # 或者使用 JointPositionActionCfg 进行连续控制
    gripper_action: ActionTerm = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"], # 同时控制两个指头
        open_command_expr={"panda_finger_joint.*": 0.04}, # 张开位置
        close_command_expr={"panda_finger_joint.*": 0.0}, # 闭合位置
    )


@configclass
class RewardsCfg:
    # reaching_reward = RewTerm(
    #     func=mdp.position_target_asset_error, # 计算 ||pos - target||
    #     weight=-1.0, # 保持负数，代表惩罚距离
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), 
    #         "target_asset_cfg": SceneEntityCfg("cube")
    #     },
    # )

    reaching_reward = RewTerm(
        func=mdp.object_approach_reward, # 使用上面修改后的函数
        weight=1.5,                  # 改为正数！因为函数返回的是[0,1]的好感度
        params={
            "std": 0.25,             # 调节参数，越小要求精度越高
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), 
            "target_asset_cfg": SceneEntityCfg("cube")
        },
    )
    
    # [新增] 靠近奖励 (Shaped Reward)
    # 相比单纯的距离惩罚，这个奖励在非常接近时给分更高，引导性更强
    approaching_reward = RewTerm(
        func=mdp.position_target_asset_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), 
            "target_asset_cfg": SceneEntityCfg("cube"),
            # 注意：IsaacLab 的 mdp.position_target_asset_error 默认返回的是距离本身
            # 如果想用 exp(-dist)，需要自定义函数。
            # 为了简单，我们先加大上面的 weight=-1.0 的权重，
            # 或者把 action_rate 的惩罚调小，让它敢于移动。
        },
    )

    # [关键修复] 必须正确引用 Lifting 函数
    object_lifted = RewTerm(
        func=mdp.object_is_lifted,  # <--- 使用我们在上面定义的 Python 函数
        weight=100.0, # 提高奖励权重，一旦举起给大分
        params={"minimum_height": 0.06, "asset_cfg": SceneEntityCfg("cube")},
    )

    # lifting_reward = RewTerm(
    #     func=mdp.object_height_continuous_reward, # 使用上面的自定义函数
    #     weight=3.0, # 权重比 reaching 大，让它拿到后更有动力往上提
    #     params={
    #         "asset_cfg": SceneEntityCfg("cube"),
    #         "target_height": 0.4, # 目标高度
    #         "std": 0.15           # 敏感度
    #     },
    # )

    # keep_vertical = RewTerm(
    #     func=mdp.object_keep_xy_penalty,
    #     weight=-1.0, # 负分惩罚
    #     params={
    #         "asset_cfg": SceneEntityCfg("cube"),
    #         # 这里的 target_pos_xy 最好设为你的生成中心，比如 (0.4, 0.0)
    #         "target_pos_xy": (0.4, 0.0) 
    #     },
    # )
    
    # 4. [保留] 任务完成的大奖 (二值奖励)
    # 当高度真的达到标准时，给一个巨大的额外奖励，作为最终目标的确认
    # task_success = RewTerm(
    #     func=mdp.object_is_lifted,
    #     weight=20.0, 
    #     params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("cube")},
    # )

    vertical_alignment = RewTerm(
        func=mdp.gripper_vertical_reward,  # <--- 使用上面定义的函数
        weight=1.0,                    # 权重不用太大，辅助引导即可
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # [调整] 降低惩罚项
    # 如果惩罚太高，机器人就不敢动了
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001) # 降低惩罚
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-5)
    energy = RewTerm(
        func=mdp.energy,
        weight=-2e-7,
        params={
            # "robot" 是你在 SceneCfg 中定义的变量名
            # joint_names 使用正则匹配，只选 "panda_joint" 开头的（即手臂）
            # 从而排除了 "panda_finger_joint" 开头的（即夹爪）
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        }
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"), 
            "threshold": 0.1,  # 只要传感器检测到任何力 > 0.1N 就重置
        },
    )


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # 每次重置时随机化方块位置
    reset_cube_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.5), "y": (-0.2, 0.2), "z": (0.025, 0.0025)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##

@configclass
class ReachcubepickEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ReachcubepickSceneCfg = ReachcubepickSceneCfg(num_envs=2000, env_spacing=ENV_SPACING)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0 # 稍微延长一点时间给夹取动作
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.sim.dt = 1.0 / 60.0

@configclass
class ReachcubepickEnvCfg_PLAY(ReachcubepickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False