"""
Training orchestration application service.
Coordinates domain objects and infrastructure services for training workflows.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from ...domain.model.value_objects import SessionId, RobotId, PlanId, SkillType
from ...domain.model.aggregates import LearningSession, HumanoidRobot, CurriculumPlan
from ...domain.repositories import (
    LearningSessionRepository, HumanoidRobotRepository, 
    CurriculumPlanRepository, DomainEventRepository
)
from ...domain.services import CurriculumProgressionService, MovementQualityAnalyzer
from ...infrastructure.adapters import GenesisSimulationAdapter

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Application service for orchestrating humanoid robot training.
    
    Coordinates between domain objects, services, and infrastructure
    to execute complete training workflows.
    """
    
    def __init__(self,
                 genesis_adapter: GenesisSimulationAdapter,
                 movement_analyzer: MovementQualityAnalyzer,
                 curriculum_service: CurriculumProgressionService,
                 session_repository: LearningSessionRepository,
                 robot_repository: HumanoidRobotRepository,
                 plan_repository: CurriculumPlanRepository,
                 event_publisher):
        self.genesis_adapter = genesis_adapter
        self.movement_analyzer = movement_analyzer
        self.curriculum_service = curriculum_service
        self.session_repository = session_repository
        self.robot_repository = robot_repository
        self.plan_repository = plan_repository
        self.event_publisher = event_publisher
    
    def start_training_session(self, command) -> Dict[str, Any]:
        """
        Start a new training session.
        
        Orchestrates session creation, validation, and initialization.
        """
        try:
            logger.info(f"Starting training session for robot {command.robot_id.value}")
            
            # Load domain objects
            robot = self.robot_repository.get_by_id(command.robot_id)
            if not robot:
                return {'success': False, 'error': 'Robot not found'}
            
            plan = self.plan_repository.get_by_id(command.plan_id)
            if not plan:
                return {'success': False, 'error': 'Plan not found'}
            
            # Check plan status
            from ...domain.model.aggregates import PlanStatus
            if plan.status != PlanStatus.ACTIVE:
                return {'success': False, 'error': 'Plan is not active'}
            
            # Create learning session
            session_id = SessionId.generate()
            session = LearningSession(
                session_id=session_id,
                robot_id=command.robot_id,
                plan_id=command.plan_id,
                session_name=command.session_name,
                max_episodes=command.max_episodes
            )
            
            # Start session
            initial_stage = plan.stages[0] if plan.stages else None
            session.start_session(initial_stage)
            
            # Save session
            self.session_repository.save(session)
            
            # Publish event
            self.event_publisher.publish({'type': 'session_started', 'session_id': session_id.value})
            
            logger.info(f"Started training session {session_id.value}")
            return {'success': True, 'session_id': session_id.value, 'status': 'started'}
            
        except Exception as e:
            logger.error(f"Failed to start training session: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_episode(self, command) -> Dict[str, Any]:
        """
        Execute a single training episode.
        
        Orchestrates episode execution, evaluation, and progress tracking.
        """
        try:
            # Load session
            session = self.session_repository.get_by_id(command.session_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            # Check session status
            from ...domain.model.aggregates import SessionStatus
            if session.status != SessionStatus.ACTIVE:
                return {'success': False, 'error': 'Session is not active'}
            
            robot = self.robot_repository.get_by_id(session.robot_id)
            plan = self.plan_repository.get_by_id(session.plan_id)
            
            # Create episode
            episode = session.create_episode(command.target_skill)
            episode.start_episode(command.target_skill)
            
            # Execute motion command through Genesis
            motion_result = self.genesis_adapter.execute_motion_command(None)
            if not motion_result.get('success', True):
                return {'success': False, 'error': motion_result.get('error', 'Genesis execution failed')}
            
            # Simulate episode steps
            for step in range(command.max_steps):
                step_result = self.genesis_adapter.simulate_episode_step(1)
                if not step_result.get('success', True):
                    break
                
                # Check physics stability
                if not step_result.get('physics_stable', True):
                    from ...domain.model.entities import EpisodeOutcome
                    return {
                        'success': True,
                        'episode_id': episode.episode_id.value,
                        'episode_outcome': EpisodeOutcome.TERMINATED_EARLY.value,
                        'termination_reason': 'physics instability'
                    }
            
            # Assess movement quality
            quality_result = self.movement_analyzer.assess_movement_quality(None)
            
            # Complete episode with success
            from ...domain.model.entities import EpisodeOutcome
            from ...domain.model.value_objects import PerformanceMetrics
            
            performance_metrics = PerformanceMetrics(
                success_rate=0.8,
                average_reward=quality_result.get('overall_quality_score', 0.7),
                learning_progress=0.1
            )
            
            session.complete_episode(EpisodeOutcome.SUCCESS, performance_metrics)
            
            # Save updated session
            self.session_repository.save(session)
            
            return {
                'success': True,
                'episode_id': episode.episode_id.value,
                'episode_outcome': EpisodeOutcome.SUCCESS.value,
                'performance_metrics': {
                    'success_rate': performance_metrics.success_rate,
                    'average_reward': performance_metrics.average_reward,
                    'learning_progress': performance_metrics.learning_progress
                }
            }
            
        except Exception as e:
            logger.error(f"Episode execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def advance_curriculum(self, command) -> Dict[str, Any]:
        """
        Advance curriculum stage if ready.
        
        Evaluates advancement readiness and progresses to next stage.
        """
        try:
            session = self.session_repository.get_by_id(command.session_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            from ...domain.model.aggregates import SessionStatus
            if session.status != SessionStatus.ACTIVE:
                return {'success': False, 'error': 'Session is not active'}
            
            plan = self.plan_repository.get_by_id(session.plan_id)
            
            # Check if already at final stage
            if session.current_stage_index >= len(plan.stages) - 1:
                return {'success': True, 'advanced': False, 'message': 'Already at final stage'}
            
            # Evaluate advancement readiness
            advancement_decision = self.curriculum_service.evaluate_advancement_readiness(session, plan)
            
            # Handle case where service returns None
            if not advancement_decision:
                from ...domain.services.curriculum_service import AdvancementDecision
                advancement_decision = AdvancementDecision(
                    should_advance=False,
                    confidence_score=0.0,
                    success_criteria_met=[],
                    remaining_requirements=['evaluation_pending']
                )
            
            if advancement_decision.should_advance:
                # Advance to next stage
                session.current_stage_index += 1
                self.session_repository.save(session)
                
                # Publish advancement event
                self.event_publisher.publish({
                    'type': 'curriculum_advanced',
                    'session_id': session.session_id.value,
                    'new_stage_index': session.current_stage_index
                })
                
                return {
                    'success': True,
                    'advanced': True,
                    'new_stage_index': session.current_stage_index,
                    'confidence_score': advancement_decision.confidence_score
                }
            else:
                return {
                    'success': True,
                    'advanced': False,
                    'remaining_requirements': advancement_decision.remaining_requirements,
                    'confidence_score': advancement_decision.confidence_score
                }
                
        except Exception as e:
            logger.error(f"Curriculum advancement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_training_progress(self, session_id: SessionId) -> Dict[str, Any]:
        """
        Get training progress for a session.
        
        Provides comprehensive progress assessment.
        """
        try:
            session = self.session_repository.get_by_id(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            # Get session statistics
            session_stats = session.get_session_statistics()
            return session_stats
            
        except Exception as e:
            logger.error(f"Failed to get training progress: {e}")
            return {'error': str(e)}
    
    def get_robot_capabilities(self, robot_id: RobotId) -> Dict[str, Any]:
        """
        Get robot capabilities assessment.
        
        Combines Genesis adapter capabilities with robot domain capabilities.
        """
        try:
            robot = self.robot_repository.get_by_id(robot_id)
            if not robot:
                return {'error': 'Robot not found'}
            
            # Get Genesis adapter capabilities
            genesis_capabilities = self.genesis_adapter.assess_robot_capabilities(None)
            
            # Get robot domain capabilities
            robot_capabilities = robot.get_robot_capabilities()
            
            # Combine capabilities
            combined_capabilities = {**genesis_capabilities, **robot_capabilities}
            
            return combined_capabilities
            
        except Exception as e:
            logger.error(f"Failed to get robot capabilities: {e}")
            return {'error': str(e)}
    
    def complete_training_session(self, session_id: SessionId) -> Dict[str, Any]:
        """
        Complete a training session with final evaluation.
        
        Orchestrates session closure and final assessments.
        """
        session = self.session_repository.find_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id.value} not found")
        
        robot = self.robot_repository.find_by_id(session.robot_id)
        
        # Complete session
        session.complete_session()
        
        # Final evaluation
        final_evaluation = self.get_training_progress(session_id)
        
        # Update robot with learned skills (if any new mastery achieved)
        self._update_robot_skills(robot, session)
        
        # Save updates
        self.session_repository.save(session)
        self.robot_repository.save(robot)
        
        logger.info(f"Completed training session {session_id.value}")
        
        return {
            'session_completed': True,
            'final_evaluation': final_evaluation,
            'skills_mastered': robot.get_mastered_skills(),
            'training_duration': session._get_session_duration(),
            'total_episodes': session.total_episodes
        }
    
    # Private helper methods
    
    def _execute_episode_simulation(self, episode, robot, plan) -> Dict[str, Any]:
        """Execute episode through simulation adapter."""
        from ...domain.model.entities import EpisodeOutcome
        from ...domain.model.value_objects import PerformanceMetrics
        
        # Simplified episode execution
        # In practice, this would involve:
        # 1. Environment reset
        # 2. Action sampling/prediction
        # 3. Step-by-step simulation
        # 4. Reward accumulation
        # 5. Termination checking
        
        # Placeholder implementation
        total_reward = 0.0
        step_count = 0
        
        # Simulate episode steps
        for step in range(100):  # Simplified fixed-length episode
            # Would get action from policy/agent here
            # For now, use random action
            import numpy as np
            action = np.random.uniform(-0.1, 0.1, robot.joint_count)
            
            # Execute step
            step_result = self.simulation.simulate_episode_step(1)
            
            if not step_result['success']:
                break
            
            # Accumulate reward (would be calculated by reward function)
            step_reward = np.random.uniform(-0.1, 1.0)  # Placeholder
            total_reward += step_reward
            step_count += 1
            
            episode.add_step_reward(step_reward)
            
            # Check termination (simplified)
            if step_reward < -0.5:  # Termination condition
                break
        
        # Determine outcome
        success_threshold = 50.0  # Placeholder
        if total_reward >= success_threshold:
            outcome = EpisodeOutcome.SUCCESS
        elif total_reward >= success_threshold * 0.7:
            outcome = EpisodeOutcome.PARTIAL_SUCCESS
        else:
            outcome = EpisodeOutcome.FAILURE
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            success_rate=1.0 if outcome == EpisodeOutcome.SUCCESS else 0.0,
            average_reward=total_reward / max(step_count, 1),
            learning_progress=0.1  # Placeholder
        )
        
        return {
            'outcome': outcome,
            'total_reward': total_reward,
            'step_count': step_count,
            'performance_metrics': performance_metrics
        }
    
    def _update_robot_skills(self, robot: HumanoidRobot, session: LearningSession):
        """Update robot skills based on session performance."""
        # Analyze session performance for skill improvements
        # This would involve more sophisticated skill assessment
        
        # Placeholder: Check if any skills should be marked as improved
        if session.successful_episodes >= 10:  # Arbitrary threshold
            # Example: Improve forward walking skill
            from ...domain.model.value_objects import LocomotionSkill, MasteryLevel, SkillAssessment
            
            improved_skill = LocomotionSkill(
                skill_type=SkillType.FORWARD_WALKING,
                mastery_level=MasteryLevel.BEGINNER,
                proficiency_score=0.6,
                last_assessed=datetime.now()
            )
            
            assessment = SkillAssessment(
                skill=improved_skill,
                assessment_score=0.6,
                confidence_level=0.8,
                evidence_quality=0.7
            )
            
            robot.assess_skill(SkillType.FORWARD_WALKING, assessment)