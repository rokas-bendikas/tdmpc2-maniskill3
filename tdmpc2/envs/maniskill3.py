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
from envs.wrappers.time_limit import TimeLimit
import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from collections import deque
from tensordict import TensorDict


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


class SingleViewEnv:

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.table_scene.ground.remove_from_scene()
        self.table_scene.ground = self._build_fake_ground(
            self.table_scene.scene, name="fake-ground"
        )

    def _build_fake_ground(self, scene, floor_width=20, altitude=0, name="ground"):
        ground = scene.create_actor_builder()
        ground.add_plane_collision(
            pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        )
        ground.add_plane_visual(
            pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
            scale=(floor_width, floor_width, floor_width),
            material=sapien.render.RenderMaterial(
                base_color=[0.9, 0.9, 0.93, 0], metallic=0.5, roughness=0.5
            ),
        )
        return ground.build_static(name=name)


class Maniskill3PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with Maniskill3 environments.
    """

    def __init__(self, cfg, env, num_frames=3, render_size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_frames * 3, render_size, render_size),
            dtype=np.uint8,
        )
        self._maxlen = num_frames
        obs_space = dict()
        self._frames = dict()
        for k, v in self.env.observation_space.spaces.items():
            if k == "state":
                obs_space[k] = v
            else:
                obs_space[k] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3 * num_frames, render_size, render_size),
                    dtype=np.uint8,
                )
                self._frames[k] = deque([], maxlen=num_frames)

        self.observation_space = gym.spaces.Dict(obs_space)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._maxlen):
            for k in self._frames.keys():
                self._frames[k].append(obs[k].permute(2, 0, 1))
        out = {
            "state": obs["state"],
        }
        for k in self._frames.keys():
            out[k] = torch.concatenate(list(self._frames[k]))
        return TensorDict(out)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_obs = {
            "state": obs["state"],
        }
        for k in self._frames.keys():
            self._frames[k].append(obs[k].permute(2, 0, 1))
            new_obs[k] = torch.concatenate(list(self._frames[k]))
        return TensorDict(new_obs), reward, done, info


MANISKILL_TASKS = {
    "ms3-push-cube": dict(
        env="MS3-PushCube-v1",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-pick-cube": dict(
        env="MS3-PickCube-v1",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-stack-cube": dict(
        env="MS3-StackCube-v1",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-pick-ycb": dict(
        env="MS3-PickSingleYCB-v1",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-peg-insertion-side": dict(
        env="MS3-PegInsertionSide-v1",
        control_mode="pd_ee_delta_pose",
    ),
}


@register_env("MS3-PushCube-v1")
class PushCubeEnvSingleView(SingleViewEnv, PushCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]

    def __init__(self, *args, **kwargs):
        self._cam_resolution = kwargs.pop("cam_resolution")
        self._near_far = kwargs.pop("near_far")
        PushCubeEnv.__init__(self, *args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._cam_resolution[0],
                width=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            CameraConfig(
                "cam_additional_0",
                pose=pose,
                width=self._cam_resolution[0],
                height=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
            ),
        ]


@register_env("MS3-PickCube-v1")
class PickCubeEnvSingleView(SingleViewEnv, PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]

    def __init__(self, *args, **kwargs):
        self._cam_resolution = kwargs.pop("cam_resolution")
        self._near_far = kwargs.pop("near_far")
        PickCubeEnv.__init__(self, *args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._cam_resolution[0],
                width=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            CameraConfig(
                "cam_additional_0",
                pose,
                self._cam_resolution[0],
                self._cam_resolution[1],
                np.pi / 2,
                self._near_far[0],
                self._near_far[1],
            ),
        ]


@register_env("MS3-StackCube-v1")
class StackCubeEnvSingleView(SingleViewEnv, StackCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]

    def __init__(self, *args, **kwargs):
        self._cam_resolution = kwargs.pop("cam_resolution")
        self._near_far = kwargs.pop("near_far")
        StackCubeEnv.__init__(self, *args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._cam_resolution[0],
                width=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            CameraConfig(
                "cam_additional_0",
                pose,
                self._cam_resolution[0],
                self._cam_resolution[1],
                np.pi / 2,
                self._near_far[0],
                self._near_far[1],
            ),
        ]


@register_env("MS3-PickSingleYCB-v1")
class PickSingleYCBEnvSingleView(SingleViewEnv, PickSingleYCBEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]

    def __init__(self, *args, **kwargs):
        self._cam_resolution = kwargs.pop("cam_resolution")
        self._near_far = kwargs.pop("near_far")
        PickSingleYCBEnv.__init__(self, *args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._cam_resolution[0],
                width=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            CameraConfig(
                "cam_additional_0",
                pose,
                self._cam_resolution[0],
                self._cam_resolution[1],
                np.pi / 2,
                self._near_far[0],
                self._near_far[1],
            ),
        ]


@register_env("MS3-PegInsertionSide-v1")
class PegInsertionSideEnvSingleView(SingleViewEnv, PegInsertionSideEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "xmate3_robotiq", "fetch"]

    def __init__(self, *args, **kwargs):
        self._cam_resolution = kwargs.pop("cam_resolution")
        self._near_far = kwargs.pop("near_far")
        PegInsertionSideEnv.__init__(self, *args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._cam_resolution[0],
                width=self._cam_resolution[1],
                fov=np.pi / 2,
                near=self._near_far[0],
                far=self._near_far[1],
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            CameraConfig(
                "cam_additional_0",
                pose,
                self._cam_resolution[0],
                self._cam_resolution[1],
                np.pi / 2,
                self._near_far[0],
                self._near_far[1],
            ),
        ]


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


class ManiSkillWrapper(gym.Wrapper):
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
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "rgb_maniskill3", "This task only supports rgb observations."
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gymnasium.make(
        task_cfg["env"],
        obs_mode="rgbd",
        robot_uids="panda_wristcam",
        control_mode=task_cfg["control_mode"],
        near_far=[0.00001, 2.0],
        cam_resolution=[64, 64],
        render_mode=cfg.render_mode if cfg.render_mode else None,
    )
    env = ManiSkillCPUGymWrapper(env)
    env = Gymnasium2GymWrapper(env)
    env = ManiSkillWrapper(env)
    env = ActionRepeatWrapper(env, repeat=2)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
