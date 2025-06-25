#!/usr/bin/env python3
"""
Create a demonstration video of the current genesis_humanoid_rl implementation.
Shows G1 robot loading, grounding system, and basic physics simulation.
"""

import genesis as gs
import numpy as np
import torch
import os
import sys
import time

# Add project root to Python path  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from robot_grounding import RobotGroundingCalculator


def main():
    print("=== Genesis Humanoid RL Demo Video Creation ===")
    
    # Initialize Genesis
    gs.init()
    
    # Create scene in headless mode for recording
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
        show_viewer=False,  # Headless mode for server environment
    )
    
    # Add ground plane
    scene.add_entity(gs.morphs.Plane())
    
    # Add camera for recording
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(3.5, -2.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        fov=45,
        GUI=False
    )
    
    # Load G1 robot with automatic grounding
    urdf_path = os.path.join(project_root, "assets/robots/g1/g1_29dof.urdf")
    
    print("Loading Unitree G1 robot...")
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0, 0, 1.0),  # Initial position, will be adjusted
            euler=(0, 0, 0),
        ),
    )
    
    # Build scene
    scene.build()
    
    # Apply automatic grounding
    print("Applying robot grounding system...")
    calculator = RobotGroundingCalculator(robot, verbose=True)
    grounding_height = calculator.get_grounding_height(safety_margin=0.03)
    robot.set_pos(torch.tensor([0, 0, grounding_height]))
    
    # Let the robot settle
    print("Stabilizing robot...")
    for _ in range(10):
        scene.step()
    
    print(f"✓ G1 robot loaded with {robot.n_dofs} DOFs")
    print(f"✓ Robot positioned at height: {grounding_height:.3f}m")
    print(f"✓ Robot has {robot.n_links} links")
    
    # Create demo videos directory
    os.makedirs("demo_videos", exist_ok=True)
    
    # Start recording
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = f"demo_videos/genesis_humanoid_rl_demo_{timestamp}.mp4"
    
    camera.start_recording()
    print(f"Recording demo video: {video_path}")
    
    # Get initial joint positions
    initial_pos = robot.get_dofs_position()
    
    step_count = 0
    max_steps = 1500  # 15 seconds at 100 FPS
    
    try:
        while step_count < max_steps:
            sim_time = step_count * 0.01
            
            # Define phases
            if step_count < 300:
                phase_name = "Phase 1: Robot Loading & Grounding"
                target_pos = initial_pos.clone()
            elif step_count < 700:
                phase_name = "Phase 2: Joint Movement Demo"
                t = (step_count - 300) * 0.01
                target_pos = initial_pos.clone()
                
                # Simple joint oscillation to show the robot is responsive
                if robot.n_dofs > 6:
                    amplitude = 0.2
                    frequency = 0.5
                    target_pos[0] += amplitude * np.sin(2 * np.pi * frequency * t)
                    target_pos[1] += amplitude * np.cos(2 * np.pi * frequency * t)
                    if robot.n_dofs > 8:
                        target_pos[6] += amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
                        target_pos[7] += amplitude * np.cos(2 * np.pi * frequency * t + np.pi)
            elif step_count < 1200:
                phase_name = "Phase 3: Observation System Demo"
                # Show different joint positions to demonstrate observation extraction
                t = (step_count - 700) * 0.01
                target_pos = initial_pos.clone()
                
                # More dynamic movement to show observation capabilities
                if robot.n_dofs > 12:
                    freq = 0.8
                    amp = 0.3
                    target_pos[0] += amp * np.sin(2 * np.pi * freq * t)
                    target_pos[2] += amp * np.sin(2 * np.pi * freq * t + np.pi/3)
                    target_pos[6] += amp * np.sin(2 * np.pi * freq * t + np.pi)
                    target_pos[8] += amp * np.sin(2 * np.pi * freq * t + 4*np.pi/3)
            else:
                phase_name = "Phase 4: Return to Rest"
                target_pos = initial_pos.clone()
            
            # Print progress every 2 seconds
            if step_count % 200 == 0:
                print(f"{phase_name} - Time: {sim_time:.1f}s/{max_steps*0.01:.1f}s")
                
                # Show current observations
                pos = robot.get_pos().cpu().numpy()
                quat = robot.get_quat().cpu().numpy()
                joint_pos = robot.get_dofs_position().cpu().numpy()
                joint_vel = robot.get_dofs_velocity().cpu().numpy()
                
                print(f"  Robot position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  Joint positions: {joint_pos[:4]}...")
                print(f"  Joint velocities: {joint_vel[:4]}...")
            
            # Apply control
            robot.control_dofs_position(target_pos)
            
            # Step simulation
            scene.step()
            
            # Render frame for recording
            camera.render(rgb=True)
            
            step_count += 1
        
        # Stop recording and save
        camera.stop_recording(save_to_filename=video_path, fps=60)
        print(f"\n✓ Demo video saved: {video_path}")
        print(f"✓ Demo completed successfully in {step_count * 0.01:.1f} seconds!")
        
        # Show final statistics
        final_pos = robot.get_pos().cpu().numpy()
        final_joint_pos = robot.get_dofs_position().cpu().numpy()
        
        print(f"\n=== Final State ===")
        print(f"Robot final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Total DOFs controlled: {robot.n_dofs}")
        print(f"Observation space size: {3 + 4 + robot.n_dofs + robot.n_dofs + robot.n_dofs + 1}")
        print(f"Action space size: {robot.n_dofs}")
        
    except KeyboardInterrupt:
        camera.stop_recording(save_to_filename=video_path, fps=60)
        print(f"\n✓ Video saved: {video_path}")
        print("Demo stopped by user")
    except Exception as e:
        camera.stop_recording(save_to_filename=video_path, fps=60)
        print(f"\n✓ Video saved: {video_path}")
        print(f"Demo stopped due to error: {e}")


if __name__ == "__main__":
    main()