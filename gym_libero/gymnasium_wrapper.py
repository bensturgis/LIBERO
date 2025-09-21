import gymnasium as gym
import numpy as np
import random

from collections import OrderedDict
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from typing import Any, Dict, Optional


def make_libero_gymnasium_env(**kwargs) -> gym.Env:
    """
    Create a LIBERO OffScreenRenderEnv wrapped to match Gymnasium + LeRobot conventions.
    """
    # Construct the underlying environment
    env = OffScreenRenderEnv(**kwargs)
    env.unwrapped = env

    env.metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": kwargs.get("control_freq", 20),
    }

    # Define action space
    low, high = env.env.action_spec
    env.action_space = spaces.Box(
        low=np.asarray(low, dtype=np.float32),
        high=np.asarray(high, dtype=np.float32),
        dtype=np.float32,
    )

    # Set up rendering
    default_camera = kwargs.get("render_camera", "frontview")
    env.camera_names = kwargs.get("camera_names", None)
    cam_h = kwargs.get("camera_heights", 256)
    cam_w = kwargs.get("camera_widths", 256)

    def render(mode: str = "rgb_array", camera_name: Optional[str] = None,):
        if mode not in ("rgb_array", "rgb"):
            raise NotImplementedError(f"Render mode {mode} not supported.")
        
        camera_name = default_camera if camera_name is None else camera_name

        if camera_name not in env.camera_names:
            raise ValueError(f"Unknown camera {camera_name!r}")
        
        frame = env.sim.render(camera_name=camera_name, height=cam_h, width=cam_w)
        return frame[::-1]

    env.render = render

    # Observation post-processing helper
    def _convert_obs(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Map robosuite keys to LeRobot keys and build nested 'pixels' dict."""
        obs: Dict[str, Any] = {}

        # Prepare state vector
        if (
            "robot0_eef_pos"  in raw_obs and
            "robot0_eef_quat" in raw_obs and
            "robot0_gripper_qpos" in raw_obs
        ):
            eef_xyz = raw_obs["robot0_eef_pos"]
            eef_rpy = quat2axisangle(raw_obs["robot0_eef_quat"])
            gripper = raw_obs["robot0_gripper_qpos"]
            obs["agent_pos"] = np.concatenate([eef_xyz, eef_rpy, gripper]).astype(np.float32)

        # if "robot0_joint_pos" in raw_obs:
        #     obs["joint_state"] = raw_obs["robot0_joint_pos"].astype(np.float32)

        # Prepare image observations
        pixels: Dict[str, np.ndarray] = {}
        if "agentview_image" in raw_obs:
            pixels["image"] = raw_obs["agentview_image"][::-1].copy()
        if "robot0_eye_in_hand_image" in raw_obs:
            pixels["wrist_image"] = raw_obs["robot0_eye_in_hand_image"][::-1].copy()

        if pixels:
            obs["pixels"] = pixels

        # Preserve ordering
        ordered_output = OrderedDict()
        for key in ("agent_pos", "pixels"):
            if key in obs:
                ordered_output[key] = obs[key]

        return ordered_output

    # Make reset method accept a seed argument for compatability with LeRobot
    original_reset = env.reset
    def reset_with_seed(*args, **kw):
        seed = kw.pop("seed", None)
        if seed is not None:
            env.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        kw.pop("options", None)
        raw_obs = original_reset(*args, **kw)
        return _convert_obs(raw_obs), {"is_success": False}
    env.reset = reset_with_seed

    original_step = env.step
    # Wrap env.step to match Gymnasium's API
    def step_mapped(action):
        # OffScreenRenderEnv step function returns (obs, reward, done, info) but
        # Gymnasium's API expects (obs, reward, terminated, truncated, info)
        raw_obs, reward, done, info = original_step(action)
        terminated = done
        truncated  = False
        info = dict(info or {})
        success_fn = getattr(env.env, "_check_success", None)  # lives on env.env
        info["is_success"] = bool(success_fn()) if success_fn else False

        return _convert_obs(raw_obs), reward, terminated, truncated, info
    env.step = step_mapped

    # Build observation_space from a sample
    sample_raw = env.env._get_observations()
    sample_converted = _convert_obs(sample_raw)

    obs_spaces = OrderedDict()
    obs_spaces["agent_pos"] = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)

    if "pixels" in sample_converted:
        img_h, img_w, _ = sample_converted["pixels"]["image"].shape
        pixel_spaces = {
            "image": spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
            "wrist_image": spaces.Box(0, 255, (img_h, img_w, 3), dtype=np.uint8),
        }
        obs_spaces["pixels"] = spaces.Dict(pixel_spaces)

    env.observation_space = spaces.Dict(obs_spaces) 

    def get_obs():
        raw_obs = env.env._get_observations()
        return _convert_obs(raw_obs)

    env.get_obs = get_obs

    env.post_process = env._post_process

    env.update_observables = env._update_observables

    env.check_success = env.env._check_success

    return env

