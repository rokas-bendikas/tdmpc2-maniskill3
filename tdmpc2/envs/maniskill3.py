import gym.spaces
import gymnasium
import gym
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks import (
    PushCubeEnv,
    PickCubeEnv as PickCubeEnvOriginal,
    StackCubeEnv,
    PickSingleYCBEnv as PickSingleYCBEnvOriginal,
    PegInsertionSideEnv,
)
from mani_skill.utils.wrappers.gymnasium import ManiSkillCPUGymWrapper
from .maniskill3_multiview import MultiViewEnv
from envs.wrappers.time_limit import TimeLimit
import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.pose import Pose


class PickCubeEnv(PickCubeEnvOriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._hidden_objects.remove(self.goal_site)


class PickSingleYCBEnv(PickSingleYCBEnvOriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]

            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)


MANISKILL_MULTIVIEW_TASKS = {
    "push-cube-multiview": dict(
        env="PushCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "pick-cube-multiview": dict(
        env="PickCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "stack-cube-multiview": dict(
        env="StackCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "pick-ycb-multiview": dict(
        env="PickSingleYCB-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "peg-insertion-side-multiview": dict(
        env="PegInsertionSide-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
}


@register_env("PushCube-v1-multiview")
class PushCubeEnvMultiView(MultiViewEnv, PushCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]
    ADDITIONAL_CAMERA_SAMPLING_CONFIG = {
        "radius_limits": [0.4, 0.5],
        "phi_limits": [-0.3 * np.pi, 0.3 * np.pi],
        "theta_limits": [0.0, 1.0],
    }

    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PushCubeEnv.__init__(
            self,
            *args,
            reconfiguration_freq=1 if self._randomize_cameras else 0,
            **kwargs
        )


@register_env("PickCube-v1-multiview")
class PickCubeEnvMultiView(MultiViewEnv, PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]
    ADDITIONAL_CAMERA_SAMPLING_CONFIG = {
        "radius_limits": [0.5, 0.7],
        "phi_limits": [-0.3 * np.pi, 0.3 * np.pi],
        "theta_limits": [0.0, 1.0],
    }

    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PickCubeEnv.__init__(
            self,
            *args,
            reconfiguration_freq=1 if self._randomize_cameras else 0,
            **kwargs
        )


@register_env("StackCube-v1-multiview")
class StackCubeEnvMultiView(MultiViewEnv, StackCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]
    ADDITIONAL_CAMERA_SAMPLING_CONFIG = {
        "radius_limits": [0.6, 0.8],
        "phi_limits": [-0.3 * np.pi, 0.3 * np.pi],
        "theta_limits": [0.0, 1.0],
    }

    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        StackCubeEnv.__init__(
            self,
            *args,
            reconfiguration_freq=1 if self._randomize_cameras else 0,
            **kwargs
        )


@register_env("PickSingleYCB-v1-multiview")
class PickSingleYCBEnvMultiView(MultiViewEnv, PickSingleYCBEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]
    ADDITIONAL_CAMERA_SAMPLING_CONFIG = {
        "radius_limits": [0.5, 0.6],
        "phi_limits": [-0.3 * np.pi, 0.3 * np.pi],
        "theta_limits": [0.0, 1.0],
    }

    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PickSingleYCBEnv.__init__(
            self,
            *args,
            reconfiguration_freq=1 if self._randomize_cameras else 0,
            **kwargs
        )


@register_env("PegInsertionSide-v1-multiview")
class PegInsertionSideEnvMultiView(MultiViewEnv, PegInsertionSideEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]
    ADDITIONAL_CAMERA_SAMPLING_CONFIG = {
        "radius_limits": [0.3, 0.4],
        "phi_limits": [-1.0 * np.pi, 0.0 * np.pi],
        "theta_limits": [0.0, 1.0],
    }

    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PegInsertionSideEnv.__init__(
            self,
            *args,
            reconfiguration_freq=1 if self._randomize_cameras else 0,
            **kwargs
        )


class Gymnasium2GymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset(*args, **kwargs)
        return obs

    def step(self, action):
        obs, reward, terminate, truncate, info = self.env.step(action)
        done = terminate or truncate
        info.update({"success": terminate})
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render()


class MultiViewManiSkillWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
                ),
                "wrist_rgb": gym.spaces.Box(
                    low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
                ),
                "cam_additional_0_rgb": gym.spaces.Box(
                    low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
                ),
                "cam_additional_1_rgb": gym.spaces.Box(
                    low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def _get_obs(self, obs):
        state = np.concatenate([obs["agent"]["qpos"], obs["agent"]["qvel"]])
        obs = {
            "state": state,
            "wrist_rgb": obs["sensor_data"]["cam_wrist"]["rgb"],
            "cam_additional_0_rgb": obs["sensor_data"]["cam_additional_0"]["rgb"],
            "cam_additional_1_rgb": obs["sensor_data"]["cam_additional_1"]["rgb"],
        }
        return obs

    def reset(self, *args, **kwargs):
        obs = self._get_obs(self.env.reset(*args, **kwargs))
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._get_obs(obs)
        return obs, reward, False, info


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=2):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        reward = 0
        for _ in range(self.repeat):
            obs, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(cfg):
    """
    Make ManiSkill3 environment.
    """
    if cfg.task not in MANISKILL_MULTIVIEW_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "rgb_multiview", "This task only supports rgb observations."
    task_cfg = MANISKILL_MULTIVIEW_TASKS[cfg.task]
    env = gymnasium.make(
        task_cfg["env"],
        obs_mode="rgbd",
        robot_uids="panda_wristcam",
        control_mode=task_cfg["control_mode"],
        randomize_cameras=True,
        num_additional_cams=2,
        near_far=[0.00001, 2.0],
        cam_resolution=[64, 64],
        render_mode=cfg.render_mode if cfg.render_mode else None,
    )
    env = ManiSkillCPUGymWrapper(env)
    env = Gymnasium2GymWrapper(env)
    env = MultiViewManiSkillWrapper(env)
    env = ActionRepeatWrapper(env, repeat=2)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
