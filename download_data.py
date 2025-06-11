"""
OpenX Embodiment Dataset Converter

This module converts datasets from the OpenX Embodiment collection (tensorflow_datasets format)
into a standardized format for training world models. It processes robotics datasets by:
- Resizing and standardizing image observations
- Normalizing and rescaling action spaces
- Converting episodes into video files and action sequences
- Saving in a format suitable for training

The converter supports multiple robotics datasets including RT-1, Bridge, LIBERO, and others.
"""

from typing import Any, Callable, Dict, Optional, Union, Tuple, List
from pathlib import Path
import os
import logging
from tqdm import tqdm
import json

import numpy as np
import tensorflow_datasets as tfds


import tensorflow as tf
import functools

import torch
from torchvision.io import write_video
import argparse

# Configuration constants
DEFAULT_LOCAL_DATASET_HOME = "/nlp/scr/quevedo/open_x_embodiment/data"
DEFAULT_REMOTE_DATASET_HOME = "gs://gresearch/robotics"
DEFAULT_OUTPUT_DIR = "converted_datasets"
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_VIDEO_FPS = 20

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActionStats:
    """Utility class to track action statistics across datasets."""
    
    def __init__(self):
        self.actions: List[np.ndarray] = []
    
    def add_action(self, action: np.ndarray) -> None:
        """Add an action vector to the statistics tracker."""
        self.actions.append(action)
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """Calculate and return action statistics."""
        if not self.actions:
            return {}
        
        all_actions = np.stack(self.actions, axis=0)
        return {
            'min': all_actions.min(axis=0),
            'max': all_actions.max(axis=0),
            'mean': all_actions.mean(axis=0),
            'std': all_actions.std(axis=0),
            'count': len(self.actions)
        }
    
    def print_stats(self) -> None:
        """Print action statistics in a formatted way."""
        stats = self.get_stats()
        if stats:
            logger.info("Action vector statistics:")
            logger.info(f"Count: {stats['count']}")
            logger.info(f"Min:   {stats['min']}")
            logger.info(f"Max:   {stats['max']}")
            logger.info(f"Mean:  {stats['mean']}")
            logger.info(f"Std:   {stats['std']}")


def resize_to_resolution(
    image: Union[tf.Tensor, np.ndarray],
    target_width: int = DEFAULT_IMAGE_WIDTH,
    target_height: int = DEFAULT_IMAGE_HEIGHT,
    to_numpy: bool = True,
) -> Union[tf.Tensor, np.ndarray]:
    """
    Resize image to target resolution with padding and cast to uint8.
    
    Args:
        image: Input image tensor
        target_width: Target width for resizing
        target_height: Target height for resizing
        to_numpy: Whether to convert result to numpy array
        
    Returns:
        Resized image as tensor or numpy array
    """
    image = tf.image.resize_with_pad(
        image,
        target_width=target_width,
        target_height=target_height,
    )
    image = tf.cast(image, tf.uint8)
    if to_numpy:
        image = image.numpy()
    return image


def map_observation(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
    from_image_feature_names: Tuple[str, ...] = ("image",),
    to_image_feature_names: Tuple[str, ...] = ("image",),
) -> None:
    """
    Map observation features from source dataset format to target format.
    
    Args:
        to_step: Target step dictionary to populate
        from_step: Source step dictionary
        from_image_feature_names: Source image feature names
        to_image_feature_names: Target image feature names
    """
    # Copy natural language embedding if present
    if "natural_language_embedding" in from_step["observation"]:
        to_step["observation"]["natural_language_embedding"] = from_step["observation"]["natural_language_embedding"]

    # Map image features
    for from_feature_name, to_feature_name in zip(
        from_image_feature_names, to_image_feature_names
    ):
        to_step["observation"][to_feature_name] = from_step["observation"][from_feature_name]


def terminate_bool_to_act(terminate_episode: np.ndarray) -> np.ndarray:
    """
    Convert boolean termination signal to action encoding.
    
    Args:
        terminate_episode: Boolean termination signal
        
    Returns:
        One-hot encoded termination action [terminate, continue, padding]
    """
    if terminate_episode == 1.0:
        return np.array([1, 0, 0], dtype=np.int32)
    else:
        return np.array([0, 1, 0], dtype=np.int32)


def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
    """
    Rescale actions from original bounds to target bounds with safety margins.
    
    Args:
        actions: Action tensor to rescale
        low: Original lower bound
        high: Original upper bound
        safety_margin: Safety margin to apply
        post_scaling_max: Target upper bound
        post_scaling_min: Target lower bound
        
    Returns:
        Rescaled action tensor
    """
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return tf.clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _rescale_action(
    action: Dict[str, tf.Tensor], 
    action_stats: ActionStats,
    wv_lo: float = -0.05, 
    wv_hi: float = 0.05, 
    rd_lo: float = -0.25, 
    rd_hi: float = 0.25
) -> Dict[str, tf.Tensor]:
    """
    Rescale action components and track statistics.
    
    Args:
        action: Action dictionary with world_vector, rotation_delta components
        action_stats: ActionStats object to track statistics
        wv_lo: World vector lower bound
        wv_hi: World vector upper bound
        rd_lo: Rotation delta lower bound
        rd_hi: Rotation delta upper bound
        
    Returns:
        Rescaled action dictionary
    """
    # Track action statistics
    action_vector = np.concatenate([action["world_vector"], action["rotation_delta"]], axis=0)
    action_stats.add_action(action_vector)
    
    # Rescale world vector
    action["world_vector"] = rescale_action_with_bound(
        action["world_vector"],
        low=wv_lo,
        high=wv_hi,
        safety_margin=0.01,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )
    
    # Rescale rotation delta
    action["rotation_delta"] = rescale_action_with_bound(
        action["rotation_delta"],
        low=rd_lo,
        high=rd_hi,
        safety_margin=0.01,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )

    return action


# Action mapping functions for different datasets
def rt_1_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    """Map RT-1 dataset actions."""
    to_step["action"] = from_step["action"]


def bridge_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any], 
    action_stats: ActionStats
) -> None:
    """Map Bridge dataset actions to standardized format."""
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]

    # Handle gripper action
    open_gripper = from_step["action"]["open_gripper"]
    possible_values = tf.constant([True, False], dtype=tf.bool)
    eq = tf.equal(possible_values, open_gripper)
    assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

    with tf.control_dependencies([assert_op]):
        to_step["action"]["gripper_closedness_action"] = tf.cond(
            open_gripper,
            lambda: tf.constant([-1.0], dtype=tf.float32),  # Open gripper
            lambda: tf.constant([1.0], dtype=tf.float32),   # Close gripper
        )

    to_step["action"] = _rescale_action(to_step["action"], action_stats)


def libero_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any], 
    action_stats: ActionStats
) -> None:
    """Map LIBERO dataset actions to standardized format."""
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(from_step["is_terminal"])
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]
    to_step["action"]["gripper_closedness_action"] = from_step["action"][6:7]

    to_step["action"] = _rescale_action(
        to_step["action"],
        action_stats,
        wv_lo=-1.0,
        wv_hi=+1.0,
        rd_lo=-0.4,
        rd_hi=+0.4
    )


def bridge_v2_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any], 
    action_stats: ActionStats
) -> None:
    """Map Bridge V2 dataset actions to standardized format."""
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(from_step["is_terminal"])
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]

    # Process gripper action
    open_gripper = from_step["action"][6:7]
    open_gripper = tf.round(open_gripper)
    open_gripper = -(open_gripper * 2 - 1)
    to_step['action']['gripper_closedness_action'] = open_gripper

    to_step["action"] = _rescale_action(to_step["action"], action_stats)


# Create partial functions for observation mapping
bridge_v2_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("image_0",),
    to_image_feature_names=("image",),
)


def taco_play_rescale_actions_by_bounds(
    actions: tf.Tensor, 
    lows: tf.Tensor, 
    highs: tf.Tensor, 
    safety_margin: float = 0.01
) -> tf.Tensor:
    """Rescale TacoPlay actions by dimension-specific bounds."""
    resc_actions = (actions - lows) / (highs - lows) * 2 - 1
    return tf.clip_by_value(resc_actions, -1 + safety_margin, 1 - safety_margin)


def taco_play_rescale_action(action: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Rescale TacoPlay actions based on measured per-dimension ranges."""
    # Rotation Delta bounds
    rd_lows = tf.constant([-3.2, -0.8, -1.8])
    rd_highs = tf.constant([3.2, 0.2, 2.5])
    action["rotation_delta"] = taco_play_rescale_actions_by_bounds(
        action["rotation_delta"], lows=rd_lows, highs=rd_highs
    )

    # World Vector bounds
    wv_lows = tf.constant([0.0, -0.5, 0.0])
    wv_highs = tf.constant([0.8, 0.7, 0.6])
    action["world_vector"] = taco_play_rescale_actions_by_bounds(
        action["world_vector"], lows=wv_lows, highs=wv_highs
    )

    return action


def taco_play_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any], 
    action_stats: ActionStats
) -> None:
    """Map TacoPlay Panda actions to standardized format."""
    actions = from_step["action"]["actions"]
    to_step["action"]["world_vector"] = actions[:3]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )
    to_step["action"]["rotation_delta"] = actions[3:6]
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(
        actions[6], axis=-1
    )

    to_step["action"] = _rescale_action(to_step["action"], action_stats)


taco_play_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("rgb_static",),
    to_image_feature_names=("image",),
)


def _normalize(value: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    """Normalize tensor by mean and standard deviation."""
    return (value - mean) / std


def jaco_play_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map JacoPlay actions to standardized format."""
    to_step["action"]["world_vector"] = _normalize(
        from_step["action"]["world_vector"],
        mean=tf.constant([0.00096585, -0.00580069, -0.00395066], dtype=tf.float32),
        std=tf.constant([0.12234575, 0.09676983, 0.11155209], dtype=tf.float32),
    )
    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["terminate_episode"] = from_step["action"]["terminate_episode"]


def berkeley_cable_routing_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map Berkeley Cable Routing actions to standardized format."""
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


def roboturk_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map RoboTurk actions to standardized format."""
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"]
    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


roboturk_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("front_rgb",),
    to_image_feature_names=("image",),
)


def nyu_door_opening_surprising_effectiveness_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map NYU Door Opening dataset actions to standardized format."""
    # Scale world vector from [-0.07, 0.07] to span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 20.0
    
    # Scale rotation delta from [-0.07, 0.07] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"] * 15.0
    
    to_step["action"]["gripper_closedness_action"] = from_step["action"][
        "gripper_closedness_action"
    ]
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )


def viola_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map VIOLA dataset actions to standardized format."""
    # Scale world vector from [-1.0, 1.0] to better span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 1.75
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    # Scale rotation delta from [-0.4, 0.4] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = (
        from_step["action"]["rotation_delta"] * 3.0
    )

    gripper_closedness_action = from_step["action"]["gripper_closedness_action"]

    # Validate gripper action values
    possible_values = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
    eq = possible_values == gripper_closedness_action
    assert eq.any(), f"Invalid gripper action: {gripper_closedness_action}"

    gripper_closedness_action = np.expand_dims(gripper_closedness_action, axis=-1)
    to_step["action"]["gripper_closedness_action"] = gripper_closedness_action


viola_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("agentview_rgb",),
    to_image_feature_names=("image",),
)


def berkeley_autolab_ur5_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map Berkeley Autolab UR5 actions to standardized format."""
    # Scale world vector from [-0.02, 0.02] to span [-2.0, 2.0]
    to_step["action"]["world_vector"] = (
        from_step["action"]["world_vector"] * 100.0
    )
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    # Scale rotation delta from [-0.07, 0.07] to span [-pi/2, pi/2]
    to_step["action"]["rotation_delta"] = (
        from_step["action"]["rotation_delta"] * 15.0
    )
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(
        from_step["action"]["gripper_closedness_action"], axis=0
    )


def toto_map_action(
    to_step: Dict[str, Any], 
    from_step: Dict[str, Any]
) -> None:
    """Map TOTO dataset actions to standardized format."""
    # Scale world vector from [-0.7, 0.7] to better span [-2.0, 2.0]
    to_step["action"]["world_vector"] = from_step["action"]["world_vector"] * 2.0
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["action"]["terminate_episode"]
    )

    to_step["action"]["rotation_delta"] = from_step["action"]["rotation_delta"]
    to_step["action"]["gripper_closedness_action"] = tf.expand_dims(
        from_step["action"]["open_gripper"], axis=0
    )
    to_step["action"]["gripper_closedness_action"] = tf.cast(
        to_step["action"]["gripper_closedness_action"], tf.float32
    )


def episode_map_fn(
    episode: Dict[str, Any], 
    map_step: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Map an episode from source format to target format.
    
    Args:
        episode: Source episode dictionary
        map_step: Function to map individual steps
        
    Returns:
        Mapped episode with video frames and actions
    """
    steps = list(map(map_step, episode["steps"]))
    frames = np.concatenate([s["observation"]["image"] for s in steps], axis=0)
    episode = {
        "video": frames,
        "action": np.stack([s["action"] for s in steps]),
    }
    return episode


def step_map_fn(
    step: Dict[str, Any], 
    map_observation: Callable, 
    map_action: Callable
) -> Dict[str, Any]:
    """
    Map a single step from source format to target format.
    
    Args:
        step: Source step dictionary
        map_observation: Function to map observations
        map_action: Function to map actions
        
    Returns:
        Mapped step dictionary
    """
    transformed_step = {}

    # Initialize observation and action structures
    transformed_step["observation"] = {}
    transformed_step["action"] = {
        "gripper_closedness_action": np.zeros(1, dtype=np.float32),
        "rotation_delta": np.zeros(3, dtype=np.float32),
        "terminate_episode": np.zeros(3, dtype=np.int32),
        "world_vector": np.zeros(3, dtype=np.float32),
        "base_displacement_vertical_rotation": np.zeros(1, dtype=np.float32),
        "base_displacement_vector": np.zeros(2, dtype=np.float32),
    }

    # Apply dataset-specific mappings
    map_observation(transformed_step, step)
    map_action(transformed_step, step)

    # Concatenate action components into single vector
    action = np.concatenate([
        transformed_step["action"]["world_vector"],
        transformed_step["action"]["rotation_delta"],
        transformed_step["action"]["gripper_closedness_action"],
        transformed_step["action"]["base_displacement_vector"],
        transformed_step["action"]["base_displacement_vertical_rotation"],
    ], axis=0)
    transformed_step["action"] = action

    return transformed_step


def get_dataset_configs(
    local_dataset_home: str = DEFAULT_LOCAL_DATASET_HOME,
    action_stats: Optional[ActionStats] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get dataset configurations for all supported datasets.
    
    Args:
        local_dataset_home: Path to local dataset storage
        action_stats: ActionStats object to track statistics
        
    Returns:
        Dictionary mapping dataset names to their configurations
    """
    if action_stats is None:
        action_stats = ActionStats()
    
    return {
        # RT-1
        "rt_1": {
            "builder_dir": f"{local_dataset_home}/fractal20220817_data/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn, map_observation=map_observation, map_action=rt_1_map_action
            ),
        },
        # Bridge
        "bridge": {
            "builder_dir": f"{local_dataset_home}/bridge/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=functools.partial(bridge_map_action, action_stats=action_stats)
            ),
        },
        "bridge_v2": {
            "builder_dir": "/matx/u/quevedo/tfds/bridge_dataset/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=bridge_v2_map_observation, 
                map_action=functools.partial(bridge_v2_map_action, action_stats=action_stats)
            ),
        },
        # LIBERO
        "libero_10": {
            "builder_dir": f"{local_dataset_home}/LIBERO/libero/modified_libero_rlds/libero_10_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=functools.partial(libero_map_action, action_stats=action_stats)
            ),
        },
        "libero_object": {
            "builder_dir": f"{local_dataset_home}/LIBERO/libero/modified_libero_rlds/libero_object_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=functools.partial(libero_map_action, action_stats=action_stats)
            ),
        },
        "libero_goal": {
            "builder_dir": f"{local_dataset_home}/LIBERO/libero/modified_libero_rlds/libero_goal_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=functools.partial(libero_map_action, action_stats=action_stats)
            ),
        },
        "libero_spatial": {
            "builder_dir": f"{local_dataset_home}/LIBERO/libero/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=functools.partial(libero_map_action, action_stats=action_stats)
            ),
        },
        # Task Agnostic Robot Play
        "taco_play": {
            "builder_dir": f"{local_dataset_home}/taco_play/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=taco_play_map_observation,
                map_action=functools.partial(taco_play_map_action, action_stats=action_stats),
            ),
        },
        # Jaco Play
        "jaco_play": {
            "builder_dir": f"{local_dataset_home}/jaco_play/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=jaco_play_map_action,
            ),
        },
        # Cable Routing
        "berkeley_cable_routing": {
            "builder_dir": f"{local_dataset_home}/berkeley_cable_routing/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=berkeley_cable_routing_map_action,
            ),
        },
        # Roboturk
        "roboturk": {
            "builder_dir": f"{local_dataset_home}/roboturk/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=roboturk_map_observation,
                map_action=roboturk_map_action,
            ),
        },
        # NYU VINN
        "nyu_door_opening_surprising_effectiveness": {
            "builder_dir": f"{local_dataset_home}/nyu_door_opening_surprising_effectiveness/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=nyu_door_opening_surprising_effectiveness_map_action,
            ),
        },
        # Austin VIOLA
        "viola": {
            "builder_dir": f"{local_dataset_home}/viola/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=viola_map_observation,
                map_action=viola_map_action,
            ),
        },
        # Berkeley Autolab UR5
        "berkeley_autolab_ur5": {
            "builder_dir": f"{local_dataset_home}/berkeley_autolab_ur5/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=berkeley_autolab_ur5_map_action,
            ),
        },
        # TOTO
        "toto": {
            "builder_dir": f"{local_dataset_home}/toto/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn, 
                map_observation=map_observation, 
                map_action=toto_map_action
            ),
        },
    }


def convert_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    video_fps: int = DEFAULT_VIDEO_FPS
) -> None:
    """
    Convert a single dataset from TFDS format to training format.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration dictionary for the dataset
        output_dir: Output directory for converted data
        video_fps: FPS for output videos
    """
    logger.info(f"Converting {dataset_name}...")
    
    # Create output directories
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    logger.info("Building dataset...")
    try:
        dataset_builder = tfds.builder_from_directory(
            builder_dir=dataset_config["builder_dir"]
        )
    except Exception as e:
        logger.error(f"Failed to build dataset {dataset_name}: {e}")
        return
    
    logger.info("Dataset built successfully.")
    
    # Process each split
    for split in ["train", "test"]:
        split_output_dir = dataset_output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        try:
            dataset = dataset_builder.as_dataset(split=split, shuffle_files=True)
        except Exception as e:
            logger.warning(f"Split {split} not available for {dataset_name}: {e}")
            continue
            
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)

        logger.info(f"Processing {split} split...")
        for i, episode in tqdm(enumerate(dataset), desc=f"{dataset_name}-{split}"):
            try:
                # Map episode to target format
                episode = episode_map_fn(episode, map_step=dataset_config["step_map_fn"])
                
                # Extract video and actions
                video = torch.from_numpy(episode["video"])
                action = episode["action"]
                
                # Save files
                base_path = split_output_dir / f"{i:09d}"
                write_video(str(base_path) + ".mp4", video, fps=video_fps)
                np.savez(str(base_path) + ".npz", action=action)
                
            except Exception as e:
                logger.error(f"Failed to process episode {i} in {dataset_name}-{split}: {e}")
                continue


def main(args: argparse.Namespace) -> None:
    """
    Main function to convert OpenX Embodiment datasets.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting dataset conversion...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Initialize action statistics tracker
    action_stats = ActionStats()
    
    # Get dataset configurations
    dataset_configs = get_dataset_configs(
        local_dataset_home=args.local_dataset_home,
        action_stats=action_stats
    )
    
    # Filter configs based on requested dataset name
    if args.dataset_name == "all":
        selected_configs = dataset_configs
    else:
        selected_configs = {
            k: v for k, v in dataset_configs.items() 
            if args.dataset_name in k
        }
    
    if not selected_configs:
        logger.error(f"No datasets found matching '{args.dataset_name}'")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert each selected dataset
    for dataset_name, dataset_config in selected_configs.items():
        convert_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            output_dir=args.output_dir,
            video_fps=args.video_fps
        )
    
    # Print action statistics
    action_stats.print_stats()
    
    # Save statistics to file
    stats_file = Path(args.output_dir) / "action_statistics.json"
    with open(stats_file, 'w') as f:
        stats = action_stats.get_stats()
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)
    
    logger.info(f"Conversion complete! Statistics saved to {stats_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert OpenX Embodiment datasets to training format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=True,
        help="Name of dataset to convert (or 'all' for all datasets)"
    )
    parser.add_argument(
        "--local-dataset-home",
        type=str,
        default=DEFAULT_LOCAL_DATASET_HOME,
        help="Path to local dataset storage"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for converted datasets"
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help="FPS for output videos"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

