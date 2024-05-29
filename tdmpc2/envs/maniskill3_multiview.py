import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
import gym
from collections import deque
import torch
from tensordict import TensorDict


class MultiViewEnv:

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

    def _sample_additional_camera_position(self):
        """Samples a random pose of a camera on the upper hemisphere."""

        radius_limits = [0.4, 0.5]
        radius = np.random.uniform(*radius_limits)

        # Adjust the camera position horizontally.
        phi = np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
        # Adjust the camera elevation.
        cos_theta = np.random.uniform(0.0, 1.0)
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


class MultiviewPixelWrapper(gym.Wrapper):
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

    # def _get_obs(self):
    #     frame = self.env.render(
    #         mode="rgb_array", width=self._render_size, height=self._render_size
    #     ).transpose(2, 0, 1)
    #     self._frames.append(frame)
    #     return torch.from_numpy(np.concatenate(self._frames))

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
