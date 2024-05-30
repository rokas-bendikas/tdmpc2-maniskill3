import gym.spaces
import gymnasium
import gym
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.utils.wrappers.gymnasium import ManiSkillCPUGymWrapper
from envs.wrappers.time_limit import TimeLimit
import sapien
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from typing import Dict, List
from envs.maniskill3 import (
    PushCubeEnv,
    PickCubeEnv,
    StackCubeEnv,
    PickSingleYCBEnv,
    PegInsertionSideEnv,
    Gymnasium2GymWrapper,
    ActionRepeatWrapper,
)


class MultiViewEnv:
    ADDITIONAL_CAMERA_SAMPLING_CONFIG: Dict[str, List[float]]

    def __init__(
        self,
        randomize_cameras,
        num_additional_cams,
        cam_resolution,
        near_far,
        scene_center=[0, 0, 0.1],
    ):
        self._randomize_cameras = randomize_cameras
        self._num_additional_cams = num_additional_cams
        self._camera_resolutions = cam_resolution
        self._near_far = near_far
        self.cam_names = ["cam_wrist"] + [
            f"cam_additional_{i}" for i in range(num_additional_cams)
        ]
        self._center = scene_center

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

    def _sample_additional_camera_position(self):
        """Samples a random pose of a camera on the upper hemisphere."""

        radius_limits = self.ADDITIONAL_CAMERA_SAMPLING_CONFIG["radius_limits"]
        radius = np.random.uniform(*radius_limits)

        # Adjust the camera position horizontally.
        phi_limits = self.ADDITIONAL_CAMERA_SAMPLING_CONFIG["phi_limits"]
        phi = np.random.uniform(*phi_limits)
        # Adjust the camera elevation.
        theta_limits = self.ADDITIONAL_CAMERA_SAMPLING_CONFIG["theta_limits"]
        cos_theta = np.random.uniform(*theta_limits)
        theta = np.arccos(cos_theta)

        # Spherical to Cartesian conversion.
        x = radius * np.sin(theta) * np.cos(phi) + self._center[0]
        y = radius * np.sin(theta) * np.sin(phi) + self._center[1]
        z = radius * np.cos(theta) + self._center[2]
        pos = [x, y, z]
        return pos

    @property
    def _default_sensor_configs(self):
        # Calculate the intrinsic matrix
        FOV = np.pi / 2
        cam_list = [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._camera_resolutions[0],
                width=self._camera_resolutions[1],
                intrinsic=None,
                fov=FOV,
                near=0.01,
                far=100,
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
        ]

        for i in range(self._num_additional_cams):
            cam_pose = sapien_utils.look_at(
                eye=self._sample_additional_camera_position(), target=[0, 0, 0.1]
            )
            cam_list.append(
                CameraConfig(
                    f"cam_additional_{i}",
                    cam_pose,
                    height=self._camera_resolutions[0],
                    width=self._camera_resolutions[1],
                    intrinsic=None,
                    fov=FOV,
                    near=self._near_far[0],
                    far=self._near_far[1],
                    mount=None,
                )
            )
        return cam_list

    @property
    def _default_human_render_camera_configs(self):
        """Add default cameras for rendering when using render_mode='rgb_array'. These can be overriden by the user at env creation time"""
        FOV = np.pi / 2
        pose = sapien_utils.look_at(
            eye=self._sample_additional_camera_position(), target=[0, 0, 0.1]
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            height=self._camera_resolutions[0],
            width=self._camera_resolutions[1],
            intrinsic=None,
            fov=FOV,
            near=self._near_far[0],
            far=self._near_far[1],
        )


MANISKILL_MULTIVIEW_TASKS = {
    "ms3-push-cube-multiview": dict(
        env="MS3-PushCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-pick-cube-multiview": dict(
        env="MS3-PickCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-stack-cube-multiview": dict(
        env="MS3-StackCube-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-pick-ycb-multiview": dict(
        env="MS3-PickSingleYCB-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
    "ms3-peg-insertion-side-multiview": dict(
        env="MS3-PegInsertionSide-v1-multiview",
        control_mode="pd_ee_delta_pose",
    ),
}


@register_env("MS3-PushCube-v1-multiview")
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
            **kwargs,
        )


@register_env("MS3-PickCube-v1-multiview")
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
            **kwargs,
        )


@register_env("MS3-StackCube-v1-multiview")
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
            **kwargs,
        )


@register_env("MS3-PickSingleYCB-v1-multiview")
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
            **kwargs,
        )


@register_env("MS3-PegInsertionSide-v1-multiview")
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
            **kwargs,
        )


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


def make_env(cfg):
    """
    Make ManiSkill3 multiview environment.
    """
    if cfg.task not in MANISKILL_MULTIVIEW_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "rgb_maniskill3", "This task only supports rgb observations."
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
