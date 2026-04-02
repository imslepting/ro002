#!/usr/bin/env python3
"""Integration test for Phase7 eye-to-hand calibration with real-time robot state.

Tests:
  1. Import all modified modules
  2. Verify RobotState dataclass
  3. Verify SamplePair supports robot state
  4. Verify sample_pairs_to_robot_samples converter
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase7_eye_to_hand.src.robot_state_fetcher import RobotState
from phase7_eye_to_hand.src.io_utils import SamplePair
from phase7_eye_to_hand.src.robot_pose_parser import RobotPoseSample
from phase7_eye_to_hand.main_eye_to_hand import _sample_pairs_to_robot_samples


def test_robot_state():
    """Test RobotState creation."""
    print("[Test 1] RobotState dataclass...")
    
    R = np.eye(3)
    t = np.array([0.1, 0.2, 0.3])
    state = RobotState(
        t_gripper2base=t,
        R_gripper2base=R,
        gripper_state=1,
        raw_joints=[0, 125, -35, 0, -45, 0],
    )
    
    assert state.t_gripper2base.shape == (3,)
    assert state.R_gripper2base.shape == (3, 3)
    assert state.gripper_state == 1
    print("  ✓ RobotState works correctly")


def test_sample_pair_with_robot_state():
    """Test SamplePair with embedded robot state."""
    print("[Test 2] SamplePair with robot state...")
    
    R_target = np.eye(3)
    t_target = np.array([[0.0], [0.0], [0.1]])
    R_gripper = np.eye(3)
    t_gripper = np.array([0.5, 0.3, 0.8])
    
    pair = SamplePair(
        sample_index=0,
        robot_row_index=0,
        image_path="/tmp/sample_0000.jpg",
        R_target2cam=R_target,
        t_target2cam=t_target,
        num_corners=36,
        R_gripper2base=R_gripper,
        t_gripper2base=t_gripper,
    )
    
    assert pair.R_gripper2base is not None
    assert pair.t_gripper2base is not None
    print("  ✓ SamplePair with robot state works correctly")


def test_conversion():
    """Test sample_pairs to robot_samples conversion."""
    print("[Test 3] Sample pairs to robot samples conversion...")
    
    pairs = []
    for i in range(3):
        R_target = np.eye(3)
        t_target = np.array([[0.0], [0.0], [0.1]])
        R_gripper = np.eye(3)
        t_gripper = np.array([0.5 + i*0.1, 0.3, 0.8])
        
        pair = SamplePair(
            sample_index=i,
            robot_row_index=i,
            image_path=f"/tmp/sample_{i:04d}.jpg",
            R_target2cam=R_target,
            t_target2cam=t_target,
            num_corners=36,
            R_gripper2base=R_gripper,
            t_gripper2base=t_gripper,
        )
        pairs.append(pair)
    
    robot_samples = _sample_pairs_to_robot_samples(pairs)
    
    assert len(robot_samples) == 3
    for i, rs in enumerate(robot_samples):
        assert isinstance(rs, RobotPoseSample)
        assert rs.index == i
        assert rs.t_gripper2base.shape == (3,)
        assert rs.R_gripper2base.shape == (3, 3)
    
    print("  ✓ Conversion to robot samples works correctly")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase7 Eye-to-Hand Integration Tests")
    print("="*60 + "\n")
    
    try:
        test_robot_state()
        test_sample_pair_with_robot_state()
        test_conversion()
        
        print("\n" + "="*60)
        print("✓ All integration tests passed!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
