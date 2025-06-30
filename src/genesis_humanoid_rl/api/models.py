"""
API models and schemas for Genesis Humanoid RL REST API.

Provides Pydantic models for request/response serialization
and validation for the REST API endpoints.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict


# Status enums
class TrainingStatus(str, Enum):
    """Training session status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationStatus(str, Enum):
    """Evaluation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RobotTypeAPI(str, Enum):
    """Robot types for API."""

    UNITREE_G1 = "unitree_g1"
    GENERIC_HUMANOID = "generic_humanoid"
    CUSTOM = "custom"


class SkillTypeAPI(str, Enum):
    """Skill types for API."""

    POSTURAL_CONTROL = "postural_control"
    STATIC_BALANCE = "static_balance"
    DYNAMIC_BALANCE = "dynamic_balance"
    FORWARD_WALKING = "forward_walking"
    BACKWARD_WALKING = "backward_walking"
    TURNING = "turning"
    SPEED_CONTROL = "speed_control"
    TERRAIN_ADAPTATION = "terrain_adaptation"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"


# Base models
class BaseResponse(BaseModel):
    """Base response model."""

    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """Error response model."""

    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# Configuration models
class TrainingConfig(BaseModel):
    """Training configuration model."""

    algorithm: str = "PPO"
    total_timesteps: int = Field(default=1000000, ge=1000)
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, ge=1)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0)
    ent_coef: float = Field(default=0.01, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    policy_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"net_arch": [256, 256, 128], "activation_fn": "tanh"}
    )

    @validator("algorithm")
    def validate_algorithm(cls, v):
        allowed = ["PPO", "SAC", "TD3", "A2C"]
        if v not in allowed:
            raise ValueError(f"Algorithm must be one of {allowed}")
        return v


class EnvironmentConfig(BaseModel):
    """Environment configuration model."""

    episode_length: int = Field(default=1000, ge=100)
    simulation_fps: int = Field(default=100, ge=10, le=1000)
    control_freq: int = Field(default=20, ge=1, le=100)
    target_velocity: float = Field(default=1.0, ge=0, le=10)
    n_envs: int = Field(default=4, ge=1, le=64)
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "forward_velocity": 1.0,
            "stability": 0.5,
            "height_maintenance": 0.3,
            "energy_efficiency": -0.1,
            "action_smoothness": -0.1,
        }
    )


class CurriculumConfig(BaseModel):
    """Curriculum learning configuration."""

    enable_curriculum: bool = True
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    auto_progression: bool = True
    progression_threshold: float = Field(default=0.8, ge=0, le=1)
    min_episodes_per_stage: int = Field(default=50, ge=10)


class RobotConfig(BaseModel):
    """Robot configuration model."""

    robot_type: RobotTypeAPI = RobotTypeAPI.UNITREE_G1
    name: str = Field(..., min_length=1, max_length=100)
    urdf_path: Optional[str] = None
    joint_count: int = Field(default=35, ge=1)
    height: float = Field(default=1.2, gt=0)
    weight: float = Field(default=35.0, gt=0)
    max_joint_velocity: float = Field(default=10.0, gt=0)
    max_joint_torque: float = Field(default=100.0, gt=0)
    control_frequency: int = Field(default=20, ge=1)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


# Request models
class CreateTrainingRequest(BaseModel):
    """Request to create a new training session."""

    session_name: str = Field(..., min_length=1, max_length=200)
    robot_config: RobotConfig
    training_config: TrainingConfig
    environment_config: EnvironmentConfig
    curriculum_config: Optional[CurriculumConfig] = None
    description: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list)


class UpdateTrainingRequest(BaseModel):
    """Request to update training session parameters."""

    session_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    tags: Optional[List[str]] = None
    training_config: Optional[TrainingConfig] = None
    environment_config: Optional[EnvironmentConfig] = None


class TrainingControlRequest(BaseModel):
    """Request to control training session (start/pause/stop)."""

    action: str = Field(..., pattern="^(start|pause|resume|stop|cancel)$")
    reason: Optional[str] = Field(None, max_length=500)


class EvaluationRequest(BaseModel):
    """Request to evaluate a trained model."""

    model_path: str = Field(..., min_length=1)
    num_episodes: int = Field(default=10, ge=1, le=100)
    render: bool = False
    save_video: bool = False
    video_path: Optional[str] = None
    environment_config: Optional[EnvironmentConfig] = None
    evaluation_name: Optional[str] = Field(None, max_length=200)


class SkillAssessmentRequest(BaseModel):
    """Request to assess robot skills."""

    robot_id: str = Field(..., min_length=1)
    skill_types: List[SkillTypeAPI] = Field(..., min_items=1)
    assessment_episodes: int = Field(default=20, ge=5, le=100)
    environment_config: Optional[EnvironmentConfig] = None


# Response models
class TrainingSessionResponse(BaseModel):
    """Training session response model."""

    session_id: str
    session_name: str
    status: TrainingStatus
    robot_config: RobotConfig
    training_config: TrainingConfig
    environment_config: EnvironmentConfig
    curriculum_config: Optional[CurriculumConfig]
    description: Optional[str]
    tags: List[str]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    model_path: Optional[str] = None
    log_path: Optional[str] = None
    error_message: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Training metrics model."""

    episode: int
    timestep: int
    episode_reward: float
    episode_length: int
    success_rate: Optional[float] = None
    fps: Optional[float] = None
    learning_rate: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy_loss: Optional[float] = None
    explained_variance: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationResult(BaseModel):
    """Evaluation result model."""

    evaluation_id: str
    evaluation_name: Optional[str]
    model_path: str
    status: EvaluationStatus
    num_episodes: int
    episodes_completed: int = 0
    average_reward: Optional[float] = None
    success_rate: Optional[float] = None
    average_episode_length: Optional[float] = None
    video_path: Optional[str] = None
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class SkillAssessmentResult(BaseModel):
    """Skill assessment result model."""

    assessment_id: str
    robot_id: str
    skill_type: SkillTypeAPI
    proficiency_score: float = Field(..., ge=0, le=1)
    confidence_level: float = Field(..., ge=0, le=1)
    evidence_quality: float = Field(..., ge=0, le=1)
    assessment_episodes: int
    success_rate: float = Field(..., ge=0, le=1)
    average_reward: float
    improvement_trend: Optional[float] = None
    recommendations: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.now)


class RobotStatus(BaseModel):
    """Robot status model."""

    robot_id: str
    robot_name: str
    robot_type: RobotTypeAPI
    status: str = "idle"  # idle, training, evaluating, error
    current_session_id: Optional[str] = None
    learned_skills: Dict[SkillTypeAPI, float] = Field(default_factory=dict)
    total_training_time: float = 0.0  # hours
    total_episodes: int = 0
    last_activity: Optional[datetime] = None
    performance_summary: Dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    """System status model."""

    system_id: str = "genesis_humanoid_rl"
    version: str = "1.0.0"
    status: str = "healthy"  # healthy, degraded, error
    uptime: float  # seconds
    active_training_sessions: int = 0
    active_evaluations: int = 0
    total_robots: int = 0
    genesis_status: Dict[str, Any] = Field(default_factory=dict)
    system_resources: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


class GenesisCompatibilityResponse(BaseModel):
    """Genesis compatibility response model."""

    genesis_version: str
    compatibility_level: str
    compatibility_score: float = Field(..., ge=0, le=1)
    working_features: List[str] = Field(default_factory=list)
    broken_features: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    test_duration: float
    tested_at: datetime


# List response models
class TrainingSessionListResponse(BaseResponse):
    """List of training sessions response."""

    sessions: List[TrainingSessionResponse] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0


class EvaluationListResponse(BaseResponse):
    """List of evaluations response."""

    evaluations: List[EvaluationResult] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0


class RobotListResponse(BaseResponse):
    """List of robots response."""

    robots: List[RobotStatus] = Field(default_factory=list)
    total_count: int = 0


class MetricsResponse(BaseResponse):
    """Training metrics response."""

    metrics: List[TrainingMetrics] = Field(default_factory=list)
    session_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class FilterParams(BaseModel):
    """Filtering parameters."""

    status: Optional[TrainingStatus] = None
    robot_type: Optional[RobotTypeAPI] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    search: Optional[str] = Field(None, max_length=200)


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str = Field(..., min_length=1)
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class TrainingProgressMessage(WebSocketMessage):
    """Real-time training progress message."""

    type: str = "training_progress"
    session_id: str
    metrics: TrainingMetrics


class SystemStatusMessage(WebSocketMessage):
    """System status update message."""

    type: str = "system_status"
    status: SystemStatus


class ErrorMessage(WebSocketMessage):
    """Error message for WebSocket."""

    type: str = "error"
    error_code: str
    error_message: str


# Health check models
class HealthCheckResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    dependencies: Dict[str, str] = Field(default_factory=dict)
    uptime: float = 0.0


class DetailedHealthCheck(HealthCheckResponse):
    """Detailed health check with component status."""

    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    system_resources: Dict[str, Any] = Field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = Field(default_factory=list)


# Monitoring models
class MonitoringMetrics(BaseModel):
    """System monitoring metrics."""

    timestamp: datetime = Field(default_factory=datetime.now)
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    training_metrics: Dict[str, Any] = Field(default_factory=dict)
    genesis_metrics: Dict[str, Any] = Field(default_factory=dict)
    api_metrics: Dict[str, Any] = Field(default_factory=dict)


class SystemAlert(BaseModel):
    """System alert model."""

    alert_id: str
    alert_type: str  # system, resource, training, evaluation
    severity: str  # low, medium, high, critical
    title: str
    description: str
    status: str = "active"  # active, acknowledged, resolved
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertListResponse(BaseResponse):
    """List of system alerts response."""

    alerts: List[SystemAlert] = Field(default_factory=list)
    total_count: int = 0
    active_count: int = 0


class PerformanceReport(BaseModel):
    """Performance analysis report."""

    report_id: str
    period: str
    generated_at: datetime = Field(default_factory=datetime.now)
    summary: Dict[str, Any] = Field(default_factory=dict)
    training_performance: Dict[str, Any] = Field(default_factory=dict)
    resource_utilization: Dict[str, Any] = Field(default_factory=dict)
    issues_identified: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
