# Robot Grounding Library

A Python library for automatically calculating the appropriate grounding height for robots in Genesis physics simulation. This library analyzes robot URDF structure to detect foot links and compute the height needed to place robots with their feet touching the ground.

## Features

- **Automatic foot detection**: Identifies foot/ankle links from robot structure
- **Accurate height calculation**: Computes exact height for ground contact
- **Multi-robot support**: Works with humanoid, quadruped, and other robot types
- **Safety margin**: Configurable clearance above ground (default 5mm)
- **Real-time updates**: Track foot positions during simulation

## Installation

The library is included in the `genesis_humanoid_learning` project. No separate installation needed.

## Quick Start

```python
import genesis as gs
from robot_grounding import RobotGroundingCalculator

# Initialize Genesis and create scene
gs.init()
scene = gs.Scene()

# Load robot at any height
robot = scene.add_entity(
    gs.morphs.URDF(file="robot.urdf", pos=(0, 0, 5.0))
)

# Build scene (required for position calculations)
scene.build()

# Calculate grounding height
calculator = RobotGroundingCalculator(robot)
ground_height = calculator.get_grounding_height()

print(f"Place robot at height: {ground_height:.3f}m")
```

## API Reference

### RobotGroundingCalculator

Main class for computing robot grounding height.

#### Constructor
```python
RobotGroundingCalculator(robot, verbose=True)
```
- `robot`: Genesis robot entity
- `verbose`: Print debug information (default: True)

#### Methods

**get_grounding_height(safety_margin=0.005)**
- Calculate height to place robot base for ground contact
- `safety_margin`: Distance above ground in meters (default: 0.005)
- Returns: Height in meters

**get_current_foot_positions()**
- Get current world positions of detected foot links
- Returns: Tensor of shape (n_feet, 3) or None

## How It Works

1. **Link Analysis**: Examines robot structure to find end effectors
2. **Pattern Matching**: Identifies foot links by name (ankle, foot, etc.)
3. **Position Calculation**: Computes lowest point of foot links
4. **Height Adjustment**: Calculates base height for ground contact

## Examples

See the `examples/` directory for complete examples:
- `test_robot_grounding.py`: Basic functionality test
- `use_robot_grounding.py`: Practical usage example

## Supported Robots

Tested with:
- Unitree G1 (humanoid)
- Other URDF robots with standard naming conventions

The library uses pattern matching to identify foot links, looking for keywords like:
- ankle, foot, toe, sole, heel
- Excludes: hand, finger, wrist, gripper

## Limitations

- Requires `scene.build()` before height calculation
- Assumes flat ground at Z=0
- Detection relies on naming conventions
- Does not account for complex foot geometries

## Future Enhancements

- Collision shape analysis for exact contact points
- Terrain adaptation for non-flat surfaces
- Custom detection rules per robot type
- Orientation adjustment for stable landing