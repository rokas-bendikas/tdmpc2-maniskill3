import gym.spaces
import gymnasium
import gym
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks import (
    PushCubeEnv,
    PickCubeEnv as PickCubeEnvOriginal,
    StackCubeEnv,
    PickSingleYCBEnv,
)
from mani_skill.utils.wrappers.gymnasium import ManiSkillCPUGymWrapper
from .maniskill3_multiview import MultiViewEnv


class PickCubeEnv(PickCubeEnvOriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._hidden_objects.remove(self.goal_site)


MANISKILL_MULTIVIEW_TASKS = {
    "push-cube-multiview": dict(
        env="PushCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "lift-cube-multiview": dict(
        env="LiftCube-v1-multiview",
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
}


@register_env("PushCube-v1-multiview", max_episode_steps=100)
class PushCubeEnvMultiView(MultiViewEnv, PushCubeEnv):
    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PushCubeEnv.__init__(self, *args, **kwargs)


@register_env("PickCube-v1-multiview", max_episode_steps=100)
class PickCubeEnvMultiView(MultiViewEnv, PickCubeEnv):
    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PickCubeEnv.__init__(self, *args, **kwargs)


@register_env("StackCube-v1-multiview", max_episode_steps=100)
class StackCubeEnvMultiView(MultiViewEnv, StackCubeEnv):
    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        StackCubeEnv.__init__(self, *args, **kwargs)


@register_env("PickSingleYCB-v1-multiview", max_episode_steps=100)
class PickSingleYCBEnvMultiView(MultiViewEnv, PickSingleYCBEnv):
    def __init__(self, *args, **kwargs):
        MultiViewEnv.__init__(
            self,
            kwargs.pop("randomize_cameras"),
            kwargs.pop("num_additional_cams"),
            kwargs.pop("cam_resolution"),
            kwargs.pop("near_far"),
        )
        PickSingleYCBEnv.__init__(self, *args, **kwargs)


class Gymnasium2GymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.env._max_episode_steps

    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset(*args, **kwargs)
        return obs

    def step(self, action):
        obs, reward, terminate, truncate, info = self.env.step(action)
        done = terminate or truncate
        info.update({"success": terminate})
        return obs, reward, done, info


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
        self.max_episode_steps = self.env.max_episode_steps

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
        return obs, reward, done, info


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

    def reset(self):
        return self.env.reset()

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

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
    )
    env = ManiSkillCPUGymWrapper(env)
    env = Gymnasium2GymWrapper(env)
    env = ManiSkillWrapper(env)
    env = ActionRepeatWrapper(env, repeat=2)
    return env
