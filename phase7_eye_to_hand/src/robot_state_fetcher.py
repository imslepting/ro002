"""Real-time robot state fetching from remote server.

Based on run_robot_webcam.py robot state management.
Handles three data formats:
  1. New format with embedded gripper: {'6axis_angle': [x, y, z, rx, ry, rz, gripper]}
  2. New format with separate grip_state: {'6axis_angle': [x, y, z, rx, ry, rz], 'grip_state': 0|1}
  3. Old format: {'state': [x, y, z, rx, ry, rz, gripper]}

IMPORTANT: Robot server returns positions in millimeters (mm).
This module automatically converts to meters (m) for calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import requests
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Current robot state."""
    t_gripper2base: np.ndarray  # (3,) in meters
    R_gripper2base: np.ndarray  # (3,3)
    euler_deg: np.ndarray  # (3,) raw [rx, ry, rz] in degrees from robot
    gripper_state: int  # 0 or 1
    raw_joints: list[float]  # Raw joint values


def fetch_robot_state(
    server_url: str = "http://140.118.117.61:5000/get_status",
    timeout: float = 1.0,
    euler_order: str = "xyz",
) -> RobotState | None:
    """
    Fetch robot state from remote server.
    
    Supports three formats:
    1. New format with embedded gripper: {'6axis_angle': [x, y, z, rx, ry, rz, gripper]}
    2. New format with separate grip_state: {'6axis_angle': [x, y, z, rx, ry, rz], 'grip_state': 0|1}
    3. Old format: {'state': [x, y, z, rx, ry, rz, gripper]}
    
    Note: All position values from robot server are in millimeters (mm).
    This function automatically converts them to meters (m).
    
    Args:
        server_url: URL of robot state endpoint
        timeout: Request timeout in seconds
        euler_order: Euler axis order used to decode [rx, ry, rz]
    
    Returns:
        RobotState object (with positions in meters) if successful, None otherwise
    """
    valid_orders = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
    if euler_order not in valid_orders:
        raise ValueError(f"invalid euler_order='{euler_order}', expected one of {sorted(valid_orders)}")

    try:
        response = requests.get(server_url, timeout=timeout)
        data = response.json()
        logger.debug(f"Received robot state: {data}")
        
        # Handle new format: 6axis_angle (with or without embedded gripper)
        if '6axis_angle' in data:
            axis_data = data.get('6axis_angle', [])
            grip_state = data.get('grip_state', 0)
            
            if isinstance(axis_data, list):
                # Format 1: 6axis_angle has 7 elements [x, y, z, rx, ry, rz, gripper]
                if len(axis_data) == 7:
                    # Convert from mm to m (robot server returns mm)
                    t_gripper2base = np.array(axis_data[:3], dtype=np.float64) / 1000.0
                    grip_state = int(axis_data[6])
                    euler_angles = axis_data[3:6]
                    raw_joints = list(axis_data)
                
                # Format 2: 6axis_angle has 6 elements, gripper in separate field
                elif len(axis_data) == 6:
                    # Convert from mm to m (robot server returns mm)
                    t_gripper2base = np.array(axis_data[:3], dtype=np.float64) / 1000.0
                    grip_state = int(grip_state) if grip_state is not None else 0
                    euler_angles = axis_data[3:6]
                    raw_joints = list(axis_data) + [grip_state]
                else:
                    logger.warning(f"6axis_angle has unexpected length {len(axis_data)}: {data}")
                    return None
                
                # Create rotation from Euler angles (in degrees)
                euler_deg = np.asarray(euler_angles, dtype=np.float64).reshape(3)
                R = Rotation.from_euler(euler_order, euler_deg, degrees=True).as_matrix()
                
                return RobotState(
                    t_gripper2base=t_gripper2base,
                    R_gripper2base=R.astype(np.float64),
                    euler_deg=euler_deg,
                    gripper_state=grip_state,
                    raw_joints=raw_joints,
                )
        
        # Handle old format: state (7 DOF)
        else:
            state = data.get('state', [])
            
            if isinstance(state, list) and len(state) >= 7:
                # Old format: assume state is [x, y, z, rx, ry, rz, gripper]
                # Convert from mm to m (robot server returns mm)
                t_gripper2base = np.array(state[:3], dtype=np.float64) / 1000.0
                
                # Create rotation from Euler angles (assuming degrees)
                euler_deg = np.asarray(state[3:6], dtype=np.float64).reshape(3)
                R = Rotation.from_euler(euler_order, euler_deg, degrees=True).as_matrix()
                
                gripper_state = int(state[6])
                
                return RobotState(
                    t_gripper2base=t_gripper2base,
                    R_gripper2base=R.astype(np.float64),
                    euler_deg=euler_deg,
                    gripper_state=gripper_state,
                    raw_joints=list(state),
                )
        
        logger.warning(f"Unknown robot state format: {data}")
        return None
                
    except requests.exceptions.Timeout:
        logger.warning(f"Robot state fetch timed out (timeout={timeout}s)")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Failed to connect to robot server at {server_url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch robot state: {e}")
        return None
