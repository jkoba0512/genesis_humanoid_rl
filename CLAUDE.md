# CLAUDE.md

This file provides essential guidance to Claude Code when working with the genesis_humanoid_rl project.

## Project Overview
**Genesis Humanoid RL** - Production-ready framework for training humanoid robots to walk using reinforcement learning.

- **Repository**: https://github.com/jkoba0512/genesis_humanoid_rl
- **Status**: ✅ Production Ready - fully implemented, tested, and deployed
- **Purpose**: Train Unitree G1 humanoid robot (35 DOF) to walk using PPO with curriculum learning

## Technology Stack
- **Python**: 3.10 (required for compatibility)
- **Physics**: Genesis v0.2.1 (`pip install genesis-world`)
- **RL**: Stable-Baselines3 (`pip install stable-baselines3[extra]`)
- **Robot**: Unitree G1 with automatic grounding system
- **Package Manager**: uv (modern Python package manager)
- **GPU**: NVIDIA with CUDA support recommended

## Quick Start Commands
```bash
# Setup (one-time)
git clone https://github.com/jkoba0512/genesis_humanoid_rl.git
cd genesis_humanoid_rl
uv run python tools/setup_environment.py

# Training
uv run python scripts/train_sb3.py --config configs/test.json          # Quick test (5 min)
uv run python scripts/train_curriculum.py --config configs/curriculum_medium.json  # Recommended (1-2 hours)

# Monitoring
uv run tensorboard --logdir ./logs/curriculum_medium --port 6006

# Evaluation & Video
uv run python scripts/evaluate_sb3.py ./models/sb3/best_model --render --episodes 5
uv run python scripts/genesis_video_record.py --steps 200
```

## Project Structure
```
genesis_humanoid_rl/
├── scripts/                   # Training, evaluation, video recording scripts
├── src/genesis_humanoid_rl/   
│   ├── environments/         # RL environments (humanoid_env.py, curriculum_env.py)
│   ├── curriculum/           # Curriculum learning system
│   ├── rewards/              # Reward functions
│   ├── domain/               # DDD architecture (value objects, entities, services)
│   ├── api/                  # REST API (46 endpoints)
│   └── infrastructure/       # Genesis adapter, monitoring, error handling
├── configs/                   # JSON training configurations
├── assets/robots/g1/         # Unitree G1 URDF and meshes
└── tools/                    # Setup and verification scripts
```

## Key Features
1. **Curriculum Learning**: 7-stage progressive training (Balance → Walking → Advanced)
2. **Parallel Training**: Multi-environment support with automatic resource management
3. **Video Recording**: Genesis camera integration for training visualization
4. **REST API**: 46 endpoints for training management and monitoring
5. **DDD Architecture**: Enterprise-grade domain-driven design
6. **Automatic Grounding**: Robot positioning system preventing ground penetration

## Critical Genesis v0.2.1 Compatibility Notes

### API Changes (Must Know!)
```python
# Scene Creation - REMOVED solver parameter
gs.init()  # Required before scene creation!
scene = gs.Scene(
    sim_options=gs.SimOptions(
        dt=0.01,
        substeps=2  # Use substeps, not solver
    )
)

# Entity Addition - NO material parameter
scene.add_entity(gs.morphs.Plane())  # Direct addition

# Robot DOF Access
robot.n_dofs  # Not len(robot.dofs)
robot.get_dofs_position()  # Not robot.dofs_position
robot.get_dofs_velocity()  # Not robot.dofs_velocity

# No scene.close() - Genesis handles cleanup
```

### Common Issues & Fixes
1. **"Genesis hasn't been initialized"**: Call `gs.init()` before scene creation
2. **SB3 activation_fn error**: Remove `"activation_fn": "tanh"` from policy_kwargs
3. **TensorBoard remote access**: Use `--host 0.0.0.0` for VPN access
4. **Missing trimesh**: Run `uv add trimesh` (required for mesh loading)

## Training Configurations
- `test.json` - Quick 5k steps test
- `default.json` - Production 1M steps  
- `curriculum_test.json` - Quick curriculum (10k steps)
- `curriculum_medium.json` - Recommended (100k steps, 1-2 hours)
- `curriculum.json` - Full training (2M steps)
- `progressive_phase[1-3].json` - Multi-phase progressive training

## Reward System
The robot learns to walk through these reward components:
- **Forward Velocity** (1.0x): Movement toward target speed
- **Stability** (0.5x): Upright posture maintenance  
- **Height** (0.3x): Proper walking height (~0.8m)
- **Energy Efficiency** (-0.1x): Smooth, efficient motion
- **Action Smoothness** (-0.1x): Continuous movements
- **Height Safety** (-0.5x): Prevents unrealistic jumping

## Development Workflow

### Adding New Features
1. Check existing patterns in similar files
2. Follow DDD architecture (domain → application → infrastructure)
3. Update tests in corresponding test files
4. Use existing reward components as templates

### Training a Model
```bash
# 1. Choose configuration
# 2. Start training with monitoring
uv run python scripts/train_curriculum.py --config configs/curriculum_medium.json
uv run tensorboard --logdir ./logs/curriculum_medium

# 3. Generate videos at checkpoints
uv run python scripts/checkpoint_video_generator.py

# 4. Evaluate final model
uv run python scripts/evaluate_sb3.py ./models/curriculum_medium/best_model --render
```

### API Usage
```bash
# Start API server
uv run python -m genesis_humanoid_rl.api.cli --dev --port 8001

# Access endpoints
curl http://localhost:8001/health/
curl http://localhost:8001/training/sessions
curl http://localhost:8001/robots/
```

## Performance Expectations
- **Simulation**: 200+ FPS with Genesis v0.2.1
- **Training**: ~100k steps/hour (varies by hardware)
- **Convergence**: 3-5x faster with curriculum learning
- **Memory**: ~20GB for multi-environment training
- **GPU**: NVIDIA RTX 3060 Ti or better recommended

## Architecture Patterns
- **Domain-Driven Design**: Rich domain model with business logic
- **Repository Pattern**: Abstract data access interfaces
- **Unit of Work**: Transaction management
- **Anti-Corruption Layer**: Genesis adapter for clean integration
- **Command Pattern**: Application layer commands
- **Event-Driven**: Domain events for loose coupling

## Testing
```bash
# Run all tests
uv run python -m pytest tests/ -v

# Specific categories
uv run python -m pytest tests/domain/ -v        # Domain logic
uv run python -m pytest tests/infrastructure/ -v # Genesis integration
uv run python -m pytest tests/api/ -v           # REST API
```

## Additional Resources
- **Detailed History**: See `CLAUDE.md.backup` for complete development phases
- **API Documentation**: Auto-generated at http://localhost:8001/docs
- **TensorBoard Guide**: Monitor training at http://localhost:6006
- **GitHub Issues**: Report problems at https://github.com/jkoba0512/genesis_humanoid_rl/issues

## Current Capabilities Summary
✅ Complete humanoid robot training pipeline  
✅ Curriculum learning with automatic progression  
✅ High-performance Genesis v0.2.1 integration  
✅ REST API with 46 endpoints  
✅ Enterprise DDD architecture  
✅ Comprehensive testing (300+ tests)  
✅ Production deployment ready