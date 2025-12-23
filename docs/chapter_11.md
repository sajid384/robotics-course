---
sidebar_position: 11
title: "Chapter 11: Control Systems"
---

# Chapter 11: Control Systems

## Learning Outcomes
By the end of this chapter, students will be able to:
- Design and implement PID controllers for robotic systems
- Apply advanced control techniques for humanoid locomotion
- Implement feedback control for balance and stability
- Develop trajectory planning and execution systems
- Evaluate control system performance and stability
- Integrate perception and control for closed-loop systems

## Overview

Control systems form the nervous system of humanoid robots, translating high-level goals into precise actuator commands. These systems must handle the complex dynamics of multi-degree-of-freedom systems while maintaining stability, accuracy, and safety. Humanoid control presents unique challenges due to the robot's high degrees of freedom, underactuation, and need for dynamic balance.

Modern humanoid control systems integrate multiple control layers, from low-level motor control to high-level task planning. This chapter covers fundamental control principles and their application to humanoid robotics, including balance control, motion planning, and adaptive control techniques.

## Control System Fundamentals

### PID Control

PID (Proportional-Integral-Derivative) control is the foundation of most robotic control systems. It provides a simple yet effective approach to controlling robotic joints and systems.

```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(None, None)):
        """
        Initialize PID controller

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) output values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        # Internal state
        self.setpoint = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update(self, measurement, dt=None):
        """
        Update PID controller with new measurement

        Args:
            measurement: Current measured value
            dt: Time step (if None, calculated internally)

        Returns:
            Control output
        """
        current_time = time.time()

        if dt is None:
            if self.last_time is None:
                dt = 0.0
            else:
                dt = current_time - self.last_time

        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(output, self.output_limits[0])
        if self.output_limits[1] is not None:
            output = min(output, self.output_limits[1])

        # Update internal state
        self.last_error = error
        self.last_time = current_time

        return output

    def set_setpoint(self, setpoint):
        """Set new setpoint for the controller"""
        self.setpoint = setpoint
        # Reset integral term to prevent windup
        self.integral = 0.0
```

### Multi-Joint PID Control for Humanoid Robots

```python
import numpy as np
import time

class MultiJointPIDController:
    def __init__(self, joint_names, kp, ki, kd, output_limits=(-100, 100)):
        """
        Multi-joint PID controller for humanoid robot

        Args:
            joint_names: List of joint names
            kp: Proportional gain (can be scalar or array)
            ki: Integral gain (can be scalar or array)
            kd: Derivative gain (can be scalar or array)
            output_limits: Output limits for all joints
        """
        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # Initialize PID controllers for each joint
        if np.isscalar(kp):
            kp = np.full(self.num_joints, kp)
        if np.isscalar(ki):
            ki = np.full(self.num_joints, ki)
        if np.isscalar(kd):
            kd = np.full(self.num_joints, kd)

        self.controllers = []
        for i in range(self.num_joints):
            controller = PIDController(
                kp=kp[i], ki=ki[i], kd[kd[i]],
                output_limits=output_limits
            )
            self.controllers.append(controller)

        # State variables
        self.current_positions = np.zeros(self.num_joints)
        self.target_positions = np.zeros(self.num_joints)
        self.velocities = np.zeros(self.num_joints)
        self.last_positions = np.zeros(self.num_joints)
        self.last_time = time.time()

    def update(self, current_positions, dt=None):
        """
        Update all joint controllers

        Args:
            current_positions: Current joint positions
            dt: Time step

        Returns:
            Control outputs for all joints
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

        # Calculate velocities if dt > 0
        if dt > 0:
            self.velocities = (current_positions - self.last_positions) / dt
            self.last_positions = current_positions.copy()

        # Update each joint controller
        outputs = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            self.controllers[i].setpoint = self.target_positions[i]
            outputs[i] = self.controllers[i].update(current_positions[i], dt)

        return outputs

    def set_target_positions(self, target_positions):
        """Set target positions for all joints"""
        if isinstance(target_positions, dict):
            # Convert dict to array in joint order
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in target_positions:
                    self.target_positions[i] = target_positions[joint_name]
        else:
            # Assume array/list in correct order
            self.target_positions = np.array(target_positions)

    def get_error(self):
        """Get current error for all joints"""
        return self.target_positions - self.current_positions
```

## Advanced Control Techniques

### Inverse Dynamics Control

Inverse dynamics control uses the robot's dynamic model to compute the required joint torques for a desired motion.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class InverseDynamicsController:
    def __init__(self, robot_model):
        """
        Inverse dynamics controller using robot model

        Args:
            robot_model: Robot model with kinematic and dynamic parameters
        """
        self.robot_model = robot_model
        self.gravity = np.array([0, 0, -9.81])  # Gravity vector

    def compute_inverse_dynamics(self, q, qd, qdd, external_forces=None):
        """
        Compute required joint torques using inverse dynamics

        Args:
            q: Joint positions
            qd: Joint velocities
            qdd: Joint accelerations
            external_forces: External forces/torques

        Returns:
            Required joint torques
        """
        # Compute Coriolis and centrifugal forces
        C = self.compute_coriolis_matrix(q, qd)

        # Compute gravitational forces
        G = self.compute_gravity_vector(q)

        # Compute mass matrix
        M = self.compute_mass_matrix(q)

        # Total required torques
        tau = M @ qdd + C @ qd + G

        # Add external forces if provided
        if external_forces is not None:
            tau += external_forces

        return tau

    def compute_mass_matrix(self, q):
        """Compute mass matrix using recursive Newton-Euler algorithm"""
        # Simplified implementation - in practice, use robot dynamics library
        # like Pinocchio, KDL, or custom implementation based on robot model
        n = len(q)
        M = np.zeros((n, n))

        # Fill mass matrix based on robot structure
        # This is a placeholder - actual implementation depends on robot model
        for i in range(n):
            M[i, i] = 1.0  # Diagonal terms as joint inertia approximations

        return M

    def compute_coriolis_matrix(self, q, qd):
        """Compute Coriolis and centrifugal forces matrix"""
        n = len(q)
        C = np.zeros((n, n))

        # Simplified model - actual implementation requires full dynamics
        # calculation based on robot kinematics
        for i in range(n):
            C[i, i] = 0.1 * abs(qd[i])  # Damping approximation

        return C

    def compute_gravity_vector(self, q):
        """Compute gravitational force vector"""
        n = len(q)
        G = np.zeros(n)

        # Calculate gravity effects on each joint
        # This depends on the specific robot model
        for i in range(n):
            G[i] = 0.5  # Simplified gravity compensation

        return G
```

### Model Predictive Control (MPC)

MPC is particularly useful for humanoid balance control, as it can handle constraints and optimize over a prediction horizon.

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class ModelPredictiveController:
    def __init__(self, state_dim, control_dim, prediction_horizon=10):
        """
        Model Predictive Controller for humanoid balance

        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            prediction_horizon: Number of steps to predict
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.N = prediction_horizon

        # Control and state weights
        self.Q = np.eye(state_dim) * 0.1  # State cost matrix
        self.R = np.eye(control_dim) * 1.0  # Control cost matrix
        self.Qf = np.eye(state_dim) * 10.0  # Terminal cost matrix

        # Constraints
        self.control_limits = np.array([10.0] * control_dim)  # Max control effort
        self.state_limits = np.array([np.inf] * state_dim)  # Max state values

    def setup_mpc_problem(self, x0, reference_trajectory):
        """
        Set up MPC optimization problem

        Args:
            x0: Initial state
            reference_trajectory: Desired state trajectory

        Returns:
            Optimized control sequence
        """
        # Define optimization variables
        X = cp.Variable((self.state_dim, self.N + 1))  # State sequence
        U = cp.Variable((self.control_dim, self.N))    # Control sequence

        # Objective function
        cost = 0

        # Running costs
        for k in range(self.N):
            state_error = X[:, k] - reference_trajectory[k]
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(U[:, k], self.R)

        # Terminal cost
        terminal_error = X[:, self.N] - reference_trajectory[self.N]
        cost += cp.quad_form(terminal_error, self.Qf)

        # Constraints
        constraints = []

        # Initial state constraint
        constraints.append(X[:, 0] == x0)

        # System dynamics constraints (simplified linear model)
        A = np.eye(self.state_dim)  # State transition matrix
        B = np.random.rand(self.state_dim, self.control_dim) * 0.1  # Control matrix

        for k in range(self.N):
            # Linearized dynamics: x_{k+1} = A*x_k + B*u_k
            constraints.append(X[:, k+1] == A @ X[:, k] + B @ U[:, k])

            # Control limits
            constraints.append(cp.norm_inf(U[:, k]) <= self.control_limits[0])

        # State limits
        for k in range(self.N + 1):
            constraints.append(cp.norm_inf(X[:, k]) <= self.state_limits[0])

        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status not in ["infeasible", "unbounded"]:
                # Return first control in sequence
                return U[:, 0].value
            else:
                # Return zero control if problem is infeasible
                return np.zeros(self.control_dim)
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            return np.zeros(self.control_dim)

    def update_reference(self, current_state, target_state, steps_ahead=5):
        """
        Generate reference trajectory

        Args:
            current_state: Current robot state
            target_state: Target state
            steps_ahead: Number of steps to plan ahead

        Returns:
            Reference trajectory
        """
        # Create smooth transition from current to target state
        trajectory = []

        for i in range(self.N + 1):
            # Interpolate between current and target states
            alpha = min(i / steps_ahead, 1.0)
            state = (1 - alpha) * current_state + alpha * target_state
            trajectory.append(state)

        return np.array(trajectory)
```

## Balance Control

### Zero Moment Point (ZMP) Control

ZMP control is fundamental for humanoid balance, ensuring the robot's center of pressure remains within its support polygon.

```python
import numpy as np

class ZMPController:
    def __init__(self, robot_mass=70.0, gravity=9.81, com_height=0.8):
        """
        Zero Moment Point controller for humanoid balance

        Args:
            robot_mass: Total robot mass (kg)
            gravity: Gravitational acceleration (m/s^2)
            com_height: Center of mass height (m)
        """
        self.mass = robot_mass
        self.g = gravity
        self.h = com_height

        # ZMP calculation parameters
        self.com_position = np.array([0.0, 0.0, com_height])
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)

        # Support polygon (simplified as rectangle)
        self.support_polygon = self.define_support_polygon()

        # PID controllers for ZMP tracking
        self.zmp_pid_x = PIDController(kp=100, ki=10, kd=5, output_limits=(-100, 100))
        self.zmp_pid_y = PIDController(kp=100, ki=10, kd=5, output_limits=(-100, 100))

    def define_support_polygon(self, foot_length=0.25, foot_width=0.15):
        """
        Define support polygon based on foot dimensions

        Returns:
            Vertices of support polygon [[x1,y1], [x2,y2], ...]
        """
        # Simplified rectangular support polygon
        half_length = foot_length / 2
        half_width = foot_width / 2

        return np.array([
            [-half_length, -half_width],  # rear left
            [half_length, -half_width],   # rear right
            [half_length, half_width],    # front right
            [-half_length, half_width]    # front left
        ])

    def calculate_zmp(self, com_pos, com_vel, com_acc):
        """
        Calculate Zero Moment Point based on COM dynamics

        Args:
            com_pos: Center of mass position [x, y, z]
            com_vel: Center of mass velocity [vx, vy, vz]
            com_acc: Center of mass acceleration [ax, ay, az]

        Returns:
            ZMP position [x, y]
        """
        x_com, y_com, z_com = com_pos
        vx_com, vy_com, vz_com = com_vel
        ax_com, ay_com, az_com = com_acc

        # ZMP calculation (simplified)
        zmp_x = x_com - (self.h * ax_com) / self.g
        zmp_y = y_com - (self.h * ay_com) / self.g

        return np.array([zmp_x, zmp_y])

    def is_zmp_stable(self, zmp_position, tolerance=0.05):
        """
        Check if ZMP is within support polygon with tolerance

        Args:
            zmp_position: ZMP position [x, y]
            tolerance: Safety margin around support polygon

        Returns:
            True if ZMP is stable, False otherwise
        """
        x, y = zmp_position

        # Get support polygon bounds with tolerance
        x_min = np.min(self.support_polygon[:, 0]) + tolerance
        x_max = np.max(self.support_polygon[:, 0]) - tolerance
        y_min = np.min(self.support_polygon[:, 1]) + tolerance
        y_max = np.max(self.support_polygon[:, 1]) - tolerance

        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def balance_control(self, com_state, target_zmp=np.array([0.0, 0.0])):
        """
        Generate balance control commands based on COM state

        Args:
            com_state: Dictionary with 'position', 'velocity', 'acceleration'
            target_zmp: Target ZMP position [x, y]

        Returns:
            Balance control commands
        """
        # Calculate current ZMP
        current_zmp = self.calculate_zmp(
            com_state['position'],
            com_state['velocity'],
            com_state['acceleration']
        )

        # Calculate ZMP errors
        zmp_error_x = target_zmp[0] - current_zmp[0]
        zmp_error_y = target_zmp[1] - current_zmp[1]

        # Generate control corrections using PID
        correction_x = self.zmp_pid_x.update(zmp_error_x)
        correction_y = self.zmp_pid_y.update(zmp_error_y)

        # Check stability
        is_stable = self.is_zmp_stable(current_zmp)

        return {
            'zmp_position': current_zmp,
            'zmp_error': np.array([zmp_error_x, zmp_error_y]),
            'correction': np.array([correction_x, correction_y]),
            'is_stable': is_stable,
            'target_zmp': target_zmp
        }

    def adjust_com_trajectory(self, current_com, target_com, dt=0.01):
        """
        Adjust COM trajectory to maintain balance

        Args:
            current_com: Current COM position
            target_com: Target COM position
            dt: Time step

        Returns:
            Adjusted COM trajectory parameters
        """
        # Simple trajectory adjustment based on ZMP error
        com_state = {
            'position': current_com,
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3)
        }

        balance_result = self.balance_control(com_state)

        if not balance_result['is_stable']:
            # Adjust target COM to improve stability
            adjustment = balance_result['correction'] * dt
            adjusted_target = target_com + np.append(adjustment, [0])  # Add z adjustment if needed

            return adjusted_target

        return target_com
```

### Capture Point Control

Capture point is an extension of ZMP that considers the robot's ability to come to a complete stop.

```python
import numpy as np

class CapturePointController:
    def __init__(self, com_height=0.8, gravity=9.81):
        """
        Capture Point controller for humanoid walking stability

        Args:
            com_height: Center of mass height
            gravity: Gravitational acceleration
        """
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_capture_point(self, com_position, com_velocity):
        """
        Calculate capture point based on COM state

        Args:
            com_position: COM position [x, y, z]
            com_velocity: COM velocity [vx, vy, vz]

        Returns:
            Capture point position [x, y]
        """
        x_com, y_com, _ = com_position
        vx_com, vy_com, _ = com_velocity

        # Capture point calculation
        capture_point_x = x_com + vx_com / self.omega
        capture_point_y = y_com + vy_com / self.omega

        return np.array([capture_point_x, capture_point_y])

    def is_capturable(self, capture_point, support_polygon, margin=0.05):
        """
        Check if robot can capture itself (come to stop within support)

        Args:
            capture_point: Capture point position
            support_polygon: Current support polygon vertices
            margin: Safety margin

        Returns:
            True if capturable, False otherwise
        """
        cp_x, cp_y = capture_point

        # Get support polygon bounds
        x_min = np.min(support_polygon[:, 0]) + margin
        x_max = np.max(support_polygon[:, 0]) - margin
        y_min = np.min(support_polygon[:, 1]) + margin
        y_max = np.max(support_polygon[:, 1]) - margin

        return (x_min <= cp_x <= x_max) and (y_min <= cp_y <= y_max)

    def generate_footstep_plan(self, current_capture_point, goal_position,
                              step_length=0.3, step_width=0.2):
        """
        Generate footstep plan based on capture point

        Args:
            current_capture_point: Current capture point
            goal_position: Goal position to reach
            step_length: Maximum step length
            step_width: Step width (lateral step distance)

        Returns:
            Planned footsteps
        """
        footsteps = []

        # Calculate direction to goal
        direction_to_goal = goal_position - current_capture_point
        distance_to_goal = np.linalg.norm(direction_to_goal)

        if distance_to_goal > 0:
            step_direction = direction_to_goal / distance_to_goal

            # Plan footsteps toward goal
            current_pos = np.array([0.0, 0.0])  # Starting position
            steps_needed = int(np.ceil(distance_to_goal / step_length))

            for i in range(steps_needed):
                # Alternate between left and right foot
                foot_offset = step_width / 2 if i % 2 == 0 else -step_width / 2
                step_pos = current_pos + step_direction * min(step_length, distance_to_goal)
                step_pos[1] += foot_offset  # Add lateral offset

                footsteps.append({
                    'position': step_pos,
                    'step_number': i + 1,
                    'support_polygon': self.generate_support_polygon(step_pos, foot_offset)
                })

                current_pos = step_pos
                distance_to_goal = np.linalg.norm(goal_position - current_capture_point)

        return footsteps

    def generate_support_polygon(self, foot_position, lateral_offset):
        """
        Generate support polygon for a foot position

        Args:
            foot_position: Foot position [x, y]
            lateral_offset: Lateral offset for foot orientation

        Returns:
            Support polygon vertices
        """
        # Simple rectangular support polygon around foot position
        foot_length = 0.25
        foot_width = 0.15

        half_length = foot_length / 2
        half_width = foot_width / 2

        polygon = np.array([
            [foot_position[0] - half_length, foot_position[1] - half_width + lateral_offset],
            [foot_position[0] + half_length, foot_position[1] - half_width + lateral_offset],
            [foot_position[0] + half_length, foot_position[1] + half_width + lateral_offset],
            [foot_position[0] - half_length, foot_position[1] + half_width + lateral_offset]
        ])

        return polygon
```

## Walking Pattern Generation

### Inverse Kinematics for Leg Motion

```python
import numpy as np

class LegIKSolver:
    def __init__(self, leg_lengths={'thigh': 0.4, 'shin': 0.4}):
        """
        Inverse kinematics solver for humanoid leg

        Args:
            leg_lengths: Dictionary with leg segment lengths
        """
        self.thigh_length = leg_lengths['thigh']
        self.shin_length = leg_lengths['shin']

    def solve_2d_reach(self, target_x, target_y, leg_side='left'):
        """
        Solve 2D inverse kinematics for leg (simplified planar model)

        Args:
            target_x, target_y: Target foot position relative to hip
            leg_side: 'left' or 'right' leg (affects sign conventions)

        Returns:
            Hip and knee angles [hip_angle, knee_angle]
        """
        # Calculate distance to target
        distance = np.sqrt(target_x**2 + target_y**2)

        # Check if target is reachable
        max_reach = self.thigh_length + self.shin_length
        min_reach = abs(self.thigh_length - self.shin_length)

        if distance > max_reach:
            # Target is too far, extend leg fully
            hip_angle = np.arctan2(target_y, target_x)
            knee_angle = 0  # Fully extended
        elif distance < min_reach:
            # Target is too close, return None or handle specially
            return None, None
        else:
            # Calculate knee angle using law of cosines
            cos_knee = (self.thigh_length**2 + self.shin_length**2 - distance**2) / \
                      (2 * self.thigh_length * self.shin_length)
            knee_angle = np.pi - np.arccos(np.clip(cos_knee, -1, 1))

            # Calculate hip angle
            cos_hip_intermediate = (self.thigh_length**2 + distance**2 - self.shin_length**2) / \
                                  (2 * self.thigh_length * distance)
            alpha = np.arccos(np.clip(cos_hip_intermediate, -1, 1))
            beta = np.arctan2(target_y, target_x)
            hip_angle = beta + alpha

        # Adjust for leg side (right leg may need sign flip)
        if leg_side == 'right':
            hip_angle = -hip_angle
            knee_angle = -knee_angle

        return hip_angle, knee_angle

    def solve_3d_reach(self, target_position, current_hip_orientation):
        """
        Solve 3D inverse kinematics for leg

        Args:
            target_position: Target foot position [x, y, z] in hip frame
            current_hip_orientation: Current hip orientation

        Returns:
            Joint angles [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]
        """
        x, y, z = target_position

        # Decompose into planar problems
        # Yaw (rotation around z-axis)
        hip_yaw = np.arctan2(y, x)

        # Calculate hip pitch and knee angle using 2D solver
        r_xy = np.sqrt(x**2 + y**2)  # Horizontal distance
        hip_pitch, knee_pitch = self.solve_2d_reach(r_xy, z, 'left')

        # For simplicity, set ankle angles to maintain foot orientation
        ankle_pitch = -hip_pitch - knee_pitch  # Maintain foot level
        ankle_roll = 0  # Keep foot level laterally

        # Hip roll is typically set based on balance requirements
        hip_roll = 0  # Adjust based on balance controller

        return {
            'hip_yaw': hip_yaw,
            'hip_roll': hip_roll,
            'hip_pitch': hip_pitch,
            'knee_pitch': knee_pitch,
            'ankle_pitch': ankle_pitch,
            'ankle_roll': ankle_roll
        }
```

### Walking Pattern Generator

```python
import numpy as np

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.8):
        """
        Generate walking patterns for humanoid robot

        Args:
            step_length: Forward step length (m)
            step_height: Maximum foot lift height (m)
            step_time: Time for one step (s)
        """
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time
        self.foot_separation = 0.2  # Lateral distance between feet

    def generate_step_trajectory(self, start_pos, end_pos, step_height=None):
        """
        Generate trajectory for single step

        Args:
            start_pos: Starting foot position [x, y, z]
            end_pos: Ending foot position [x, y, z]
            step_height: Maximum step height (optional)

        Returns:
            Array of positions over time
        """
        if step_height is None:
            step_height = self.step_height

        # Generate time points
        dt = 0.01  # 10ms time steps
        t = np.arange(0, self.step_time, dt)
        num_points = len(t)

        # Generate trajectories for each axis
        x_traj = np.linspace(start_pos[0], end_pos[0], num_points)
        y_traj = np.linspace(start_pos[1], end_pos[1], num_points)

        # Z trajectory with parabolic lift
        z_lift = np.zeros(num_points)
        lift_phase = int(num_points * 0.4)  # 40% of step for lifting
        lower_phase = int(num_points * 0.7)  # 70% of step for lowering

        # Parabolic lift
        for i in range(lift_phase):
            progress = i / lift_phase
            z_lift[i] = start_pos[2] + step_height * (1 - np.cos(progress * np.pi)) / 2

        # Parabolic lower
        for i in range(lift_phase, lower_phase):
            progress = (i - lift_phase) / (lower_phase - lift_phase)
            z_lift[i] = start_pos[2] + step_height * (1 + np.cos(progress * np.pi)) / 2

        # Keep at ground level for rest
        for i in range(lower_phase, num_points):
            z_lift[i] = start_pos[2]

        # Combine trajectories
        trajectory = np.column_stack([x_traj, y_traj, z_lift])

        return trajectory, t

    def generate_walk_cycle(self, num_steps=4, walking_speed=0.5):
        """
        Generate complete walking cycle

        Args:
            num_steps: Number of steps to generate
            walking_speed: Desired walking speed (m/s)

        Returns:
            Dictionary containing left and right foot trajectories
        """
        # Adjust step time based on desired speed
        adjusted_step_time = self.step_length / walking_speed if walking_speed > 0 else self.step_time

        # Initialize foot positions
        left_foot_pos = np.array([0, self.foot_separation/2, 0])
        right_foot_pos = np.array([0, -self.foot_separation/2, 0])

        # Store trajectories
        left_trajectories = []
        right_trajectories = []
        left_times = []
        right_times = []

        for step in range(num_steps):
            if step % 2 == 0:
                # Left foot moves forward
                next_left_pos = left_foot_pos + np.array([self.step_length, 0, 0])
                left_traj, left_t = self.generate_step_trajectory(
                    left_foot_pos, next_left_pos, self.step_height
                )

                # Add time offset for this step
                if step > 0:
                    time_offset = right_times[-1][-1] if right_times else 0
                    left_t = left_t + time_offset
                    right_t = np.linspace(time_offset,
                                        time_offset + adjusted_step_time,
                                        len(right_trajectories[-1])) if right_trajectories else np.array([])

                left_trajectories.append(left_traj)
                left_times.append(left_t)
                left_foot_pos = next_left_pos

            else:
                # Right foot moves forward
                next_right_pos = right_foot_pos + np.array([self.step_length, 0, 0])
                right_traj, right_t = self.generate_step_trajectory(
                    right_foot_pos, next_right_pos, self.step_height
                )

                # Add time offset for this step
                time_offset = left_times[-1][-1] if left_times else 0
                right_t = right_t + time_offset
                if left_trajectories:
                    left_t = np.linspace(time_offset,
                                       time_offset + adjusted_step_time,
                                       len(left_trajectories[-1]))

                right_trajectories.append(right_traj)
                right_times.append(right_t)
                right_foot_pos = next_right_pos

        # Combine all trajectories
        all_left = np.vstack(left_trajectories) if left_trajectories else np.array([])
        all_right = np.vstack(right_trajectories) if right_trajectories else np.array([])

        # Combine time arrays
        all_left_times = np.hstack(left_times) if left_times else np.array([])
        all_right_times = np.hstack(right_times) if right_times else np.array([])

        return {
            'left_foot_trajectory': all_left,
            'right_foot_trajectory': all_right,
            'left_foot_times': all_left_times,
            'right_foot_times': all_right_times,
            'step_timing': adjusted_step_time
        }

    def generate_com_trajectory(self, walk_data):
        """
        Generate Center of Mass trajectory synchronized with footsteps
        """
        # Generate CoM trajectory that moves smoothly between footsteps
        left_times = walk_data['left_foot_times']
        right_times = walk_data['right_foot_times']

        if len(left_times) == 0 and len(right_times) == 0:
            return np.array([]), np.array([])

        # Determine common time base
        max_time = max(
            left_times[-1] if len(left_times) > 0 else 0,
            right_times[-1] if len(right_times) > 0 else 0
        )

        dt = 0.01
        com_times = np.arange(0, max_time, dt)

        # Generate CoM trajectory (simplified: moves in straight line between step phases)
        com_trajectory = []
        current_com = np.array([0, 0, 0.8])  # Start at neutral position

        for t in com_times:
            # Simple CoM pattern that moves forward with walking
            progress = t / max_time if max_time > 0 else 0
            forward_progress = progress * self.step_length * len(walk_data.get('left_foot_trajectory', [])) / 100

            # Lateral oscillation for balance
            lateral_oscillation = 0.02 * np.sin(2 * np.pi * t / self.step_time) if self.step_time > 0 else 0

            com_pos = np.array([
                forward_progress,
                lateral_oscillation,
                0.8  # Maintain constant height
            ])

            com_trajectory.append(com_pos)

        return np.array(com_trajectory), com_times
```

## Trajectory Planning and Execution

### Joint Trajectory Controller

```python
import numpy as np
import time

class JointTrajectoryController:
    def __init__(self, joint_names, max_velocity=1.0, max_acceleration=2.0):
        """
        Controller for executing joint trajectories

        Args:
            joint_names: List of joint names
            max_velocity: Maximum joint velocity (rad/s)
            max_acceleration: Maximum joint acceleration (rad/s^2)
        """
        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # Current state
        self.current_positions = np.zeros(self.n_joints)
        self.current_velocities = np.zeros(self.n_joints)
        self.current_accelerations = np.zeros(self.n_joints)

        # Trajectory tracking
        self.active_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0

        # PID controllers for trajectory following
        self.position_pids = [
            PIDController(kp=10.0, ki=0.1, kd=0.5, output_limits=(-50, 50))
            for _ in range(self.n_joints)
        ]

    def execute_trajectory(self, trajectory, sample_time=0.01):
        """
        Execute joint trajectory

        Args:
            trajectory: Dictionary with 'positions', 'velocities', 'time_from_start'
            sample_time: Controller sample time
        """
        self.active_trajectory = trajectory
        self.trajectory_start_time = time.time()
        self.trajectory_index = 0

        # Execute trajectory point by point
        for i, (pos, vel, t) in enumerate(zip(
            trajectory['positions'],
            trajectory['velocities'],
            trajectory['time_from_start']
        )):
            self.trajectory_index = i

            # Wait until time for this point
            elapsed_time = time.time() - self.trajectory_start_time
            sleep_time = t.to_sec() - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)

            # Update current state (simulated)
            self.current_positions = np.array(pos)
            self.current_velocities = np.array(vel)

            # Generate control commands
            control_outputs = self.compute_control(pos, vel)

    def compute_control(self, target_positions, target_velocities):
        """
        Compute control commands for trajectory following

        Args:
            target_positions: Desired joint positions
            target_velocities: Desired joint velocities

        Returns:
            Control outputs for each joint
        """
        outputs = []

        for i in range(self.n_joints):
            # Set target for PID controller
            self.position_pids[i].setpoint = target_positions[i]

            # Update controller with current measurement
            output = self.position_pids[i].update(self.current_positions[i])
            outputs.append(output)

        return np.array(outputs)

    def interpolate_trajectory(self, waypoints, times, sample_time=0.01):
        """
        Interpolate trajectory between waypoints

        Args:
            waypoints: List of joint position arrays
            times: Time for each waypoint
            sample_time: Sample time for interpolation

        Returns:
            Interpolated trajectory
        """
        # Create time vector
        total_time = times[-1]
        time_vector = np.arange(0, total_time, sample_time)

        # Interpolate each joint separately
        interpolated_positions = []
        interpolated_velocities = []

        for joint_idx in range(self.n_joints):
            joint_positions = [waypoint[joint_idx] for waypoint in waypoints]

            # Linear interpolation for positions
            joint_interp_pos = np.interp(time_vector, times, joint_positions)

            # Compute velocities using finite differences
            joint_interp_vel = np.gradient(joint_interp_pos, sample_time)

            interpolated_positions.append(joint_interp_pos)
            interpolated_velocities.append(joint_interp_vel)

        # Transpose to get time-major format
        positions = np.array(interpolated_positions).T
        velocities = np.array(interpolated_velocities).T

        return {
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'time_from_start': time_vector.tolist()
        }

    def check_trajectory_feasibility(self, trajectory):
        """
        Check if trajectory is kinematically and dynamically feasible

        Args:
            trajectory: Trajectory to check

        Returns:
            True if feasible, False otherwise
        """
        positions = np.array(trajectory['positions'])
        velocities = np.array(trajectory['velocities'])

        # Check velocity limits
        max_vel = np.max(np.abs(velocities))
        if max_vel > self.max_velocity:
            print(f"Trajectory violates velocity limits: {max_vel} > {self.max_velocity}")
            return False

        # Check acceleration limits (approximate)
        accelerations = np.gradient(velocities, axis=0) / 0.01  # Assuming 10ms dt
        max_acc = np.max(np.abs(accelerations))
        if max_acc > self.max_acceleration:
            print(f"Trajectory violates acceleration limits: {max_acc} > {self.max_acceleration}")
            return False

        return True
```

## Control System Integration

### Complete Humanoid Controller

```python
import numpy as np
import threading
import time
from collections import deque

class HumanoidController:
    def __init__(self, robot_config):
        """
        Complete humanoid robot controller integrating all control systems

        Args:
            robot_config: Configuration dictionary with robot parameters
        """
        # Robot configuration
        self.joint_names = robot_config.get('joint_names', [])
        self.n_joints = len(self.joint_names)

        # Initialize control modules
        self.pid_controller = MultiJointPIDController(
            self.joint_names,
            kp=robot_config.get('kp', 10.0),
            ki=robot_config.get('ki', 0.1),
            kd=robot_config.get('kd', 0.5)
        )

        self.zmp_controller = ZMPController(
            robot_mass=robot_config.get('mass', 70.0),
            com_height=robot_config.get('com_height', 0.8)
        )

        self.walk_generator = WalkingPatternGenerator(
            step_length=robot_config.get('step_length', 0.3),
            step_height=robot_config.get('step_height', 0.05),
            step_time=robot_config.get('step_time', 0.8)
        )

        self.trajectory_controller = JointTrajectoryController(self.joint_names)

        # State variables
        self.current_joint_positions = np.zeros(self.n_joints)
        self.current_joint_velocities = np.zeros(self.n_joints)
        self.com_state = {
            'position': np.array([0.0, 0.0, 0.8]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3)
        }

        # Control flags
        self.is_running = False
        self.control_mode = 'idle'  # 'idle', 'balance', 'walk', 'trajectory'
        self.target_positions = np.zeros(self.n_joints)

        # Threading
        self.control_thread = None
        self.dt = 0.01  # 10ms control loop

    def start_controller(self):
        """Start the control loop in a separate thread"""
        self.is_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_controller(self):
        """Stop the control loop"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()

    def control_loop(self):
        """Main control loop running at specified frequency"""
        rate = self.dt
        last_time = time.time()

        while self.is_running:
            current_time = time.time()
            if current_time - last_time >= rate:
                # Execute control cycle
                self.execute_control_cycle()
                last_time = current_time

            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def execute_control_cycle(self):
        """Execute one control cycle"""
        if self.control_mode == 'balance':
            self.execute_balance_control()
        elif self.control_mode == 'walk':
            self.execute_walk_control()
        elif self.control_mode == 'trajectory':
            self.execute_trajectory_control()
        else:
            # Idle mode - maintain current position
            control_outputs = self.pid_controller.update(
                self.current_joint_positions
            )
            # Apply control outputs to actuators (implementation depends on robot interface)
            self.apply_control_outputs(control_outputs)

    def execute_balance_control(self):
        """Execute balance control"""
        # Calculate ZMP-based balance corrections
        balance_result = self.zmp_controller.balance_control(self.com_state)

        if not balance_result['is_stable']:
            # Adjust target positions to improve balance
            self.adjust_balance_targets(balance_result['correction'])

        # Update PID targets and compute control
        self.pid_controller.set_target_positions(self.target_positions)
        control_outputs = self.pid_controller.update(self.current_joint_positions)

        # Apply control outputs
        self.apply_control_outputs(control_outputs)

    def execute_walk_control(self):
        """Execute walking control"""
        # Generate walking pattern
        walk_data = self.walk_generator.generate_walk_cycle(num_steps=1)

        # Generate CoM trajectory for stability
        com_trajectory, com_times = self.walk_generator.generate_com_trajectory(walk_data)

        # Update balance controller with CoM trajectory
        if len(com_trajectory) > 0:
            self.com_state['position'] = com_trajectory[min(len(com_trajectory)-1,
                                                         int(time.time()*100) % len(com_trajectory))]

        # Execute balance control with walking adjustments
        self.execute_balance_control()

    def execute_trajectory_control(self):
        """Execute trajectory following control"""
        # Use trajectory controller to follow planned trajectory
        # Implementation would call trajectory controller methods
        pass

    def adjust_balance_targets(self, correction):
        """Adjust target positions based on balance correction"""
        # Simplified balance adjustment - in practice, this would involve
        # inverse kinematics to determine how to move joints to improve balance
        adjustment_magnitude = np.linalg.norm(correction)
        if adjustment_magnitude > 0.01:  # Only adjust if significant
            # Distribute correction to relevant joints (hips, ankles)
            hip_joints = [j for j in self.joint_names if 'hip' in j]
            ankle_joints = [j for j in self.joint_names if 'ankle' in j]

            # Apply small adjustments to hip and ankle joints
            for joint_idx, joint_name in enumerate(self.joint_names):
                if 'hip' in joint_name or 'ankle' in joint_name:
                    # Apply proportional adjustment based on correction
                    if 'x' in joint_name or 'pitch' in joint_name:
                        self.target_positions[joint_idx] += correction[0] * 0.1
                    elif 'y' in joint_name or 'roll' in joint_name:
                        self.target_positions[joint_idx] += correction[1] * 0.1

    def apply_control_outputs(self, outputs):
        """Apply control outputs to robot actuators"""
        # This would interface with actual robot hardware
        # For simulation, update internal state
        # In practice, send commands to actuators via ROS or other interface
        pass

    def set_control_mode(self, mode):
        """Set controller mode"""
        if mode in ['idle', 'balance', 'walk', 'trajectory']:
            self.control_mode = mode
            print(f"Control mode set to: {mode}")
        else:
            print(f"Invalid control mode: {mode}")

    def set_target_positions(self, target_positions):
        """Set target joint positions"""
        if isinstance(target_positions, dict):
            # Convert dict to array in joint order
            targets = np.zeros(self.n_joints)
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in target_positions:
                    targets[i] = target_positions[joint_name]
            self.target_positions = targets
        else:
            self.target_positions = np.array(target_positions)

    def update_sensor_data(self, joint_positions, joint_velocities, com_state):
        """Update controller with current sensor data"""
        self.current_joint_positions = np.array(joint_positions)
        self.current_joint_velocities = np.array(joint_velocities)
        self.com_state = com_state

# Example usage
def main():
    # Robot configuration
    robot_config = {
        'joint_names': [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll'
        ],
        'kp': 10.0,
        'ki': 0.1,
        'kd': 0.5,
        'mass': 70.0,
        'com_height': 0.8,
        'step_length': 0.3,
        'step_height': 0.05,
        'step_time': 0.8
    }

    # Initialize controller
    controller = HumanoidController(robot_config)

    # Start controller
    controller.start_controller()

    # Set to balance mode
    controller.set_control_mode('balance')

    try:
        # Simulate for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Simulate updating sensor data
            joint_positions = np.random.randn(controller.n_joints) * 0.1
            joint_velocities = np.random.randn(controller.n_joints) * 0.01
            com_state = {
                'position': np.array([0.0, 0.0, 0.8]),
                'velocity': np.random.randn(3) * 0.01,
                'acceleration': np.random.randn(3) * 0.1
            }

            controller.update_sensor_data(joint_positions, joint_velocities, com_state)

            time.sleep(0.1)  # Update at 10Hz

    finally:
        # Stop controller
        controller.stop_controller()

if __name__ == "__main__":
    main()
```

## Weekly Breakdown for Chapter 11
- **Week 11.1**: PID control and fundamentals
- **Week 11.2**: Advanced control techniques (MPC, inverse dynamics)
- **Week 11.3**: Balance control (ZMP, capture point)
- **Week 11.4**: Walking pattern generation and trajectory execution

## Assessment
- **Quiz 11.1**: Control system fundamentals and PID control (Multiple choice and short answer)
- **Assignment 11.2**: Implement a balance controller for humanoid robot
- **Lab Exercise 11.1**: Design and test walking pattern generator

## Diagram Placeholders
- ![Control System Architecture](./images/control_system_architecture.png)
- ![ZMP Control Diagram](./images/zmp_control_diagram.png)
- ![Walking Pattern Generation](./images/walking_pattern_generation.png)

## Code Snippet: Advanced Control System
```python
#!/usr/bin/env python3

import numpy as np
import rospy
import time
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3
import threading
import queue

class AdvancedHumanoidController:
    """
    Advanced humanoid controller integrating multiple control strategies
    """
    def __init__(self):
        # ROS initialization
        rospy.init_node('advanced_humanoid_controller')

        # Publishers and subscribers
        self.joint_cmd_pub = rospy.Publisher('/joint_commands', JointState, queue_size=10)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.com_sub = rospy.Subscriber('/center_of_mass', Vector3, self.com_callback)

        # Control parameters
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        self.n_joints = len(self.joint_names)

        # State variables
        self.current_positions = np.zeros(self.n_joints)
        self.current_velocities = np.zeros(self.n_joints)
        self.current_efforts = np.zeros(self.n_joints)
        self.com_position = np.array([0.0, 0.0, 0.8])
        self.com_velocity = np.zeros(3)

        # Control modules
        self.impedance_controllers = self.initialize_impedance_controllers()
        self.mpc_controller = ModelPredictiveController(state_dim=6, control_dim=6, prediction_horizon=10)
        self.balance_controller = ZMPController()

        # Threading
        self.control_thread = None
        self.is_running = False
        self.control_queue = queue.Queue(maxsize=10)

        # Performance tracking
        self.control_times = []
        self.stability_history = []

        rospy.loginfo("Advanced Humanoid Controller initialized")

    def initialize_impedance_controllers(self):
        """Initialize impedance controllers for each joint"""
        controllers = []
        for i in range(self.n_joints):
            # Each joint has stiffness and damping properties
            controller = {
                'stiffness': 1000.0,  # N*m/rad
                'damping': 50.0,      # N*m*s/rad
                'target_position': 0.0
            }
            controllers.append(controller)
        return controllers

    def joint_state_callback(self, msg):
        """Handle joint state messages"""
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.name.index(name)
                self.current_positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.current_velocities[i] = msg.velocity[idx]
                if idx < len(msg.effort):
                    self.current_efforts[i] = msg.effort[idx]
            except ValueError:
                # Joint not found in message
                pass

    def com_callback(self, msg):
        """Handle center of mass messages"""
        self.com_position = np.array([msg.x, msg.y, msg.z])

    def start_control_loop(self):
        """Start the control loop"""
        self.is_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()

    def control_loop(self):
        """Main control loop"""
        rate = rospy.Rate(100)  # 100 Hz control frequency

        while self.is_running and not rospy.is_shutdown():
            start_time = time.time()

            try:
                # Execute multi-level control
                control_commands = self.multi_level_control()

                # Publish control commands
                self.publish_joint_commands(control_commands)

                # Track performance
                control_time = time.time() - start_time
                self.control_times.append(control_time)

                # Check stability
                stability = self.check_stability()
                self.stability_history.append(stability)

                # Log performance if needed
                if len(self.control_times) % 100 == 0:
                    avg_time = np.mean(self.control_times[-100:])
                    rospy.loginfo(f"Control loop avg time: {avg_time*1000:.1f}ms, "
                                f"Stable: {stability}")

            except Exception as e:
                rospy.logerr(f"Control loop error: {e}")

            rate.sleep()

    def multi_level_control(self):
        """Execute multi-level control strategy"""
        # Level 1: High-level task planning (walking, manipulation, etc.)
        task_commands = self.high_level_planning()

        # Level 2: Balance control using ZMP
        balance_corrections = self.balance_control()

        # Level 3: Impedance control for compliance
        impedance_commands = self.impedance_control()

        # Level 4: MPC for trajectory optimization
        mpc_commands = self.mpc_control()

        # Combine all control levels
        final_commands = self.combine_control_levels(
            task_commands, balance_corrections,
            impedance_commands, mpc_commands
        )

        return final_commands

    def high_level_planning(self):
        """High-level task planning"""
        # For now, return neutral positions
        # In practice, this would interface with high-level planners
        neutral_positions = np.zeros(self.n_joints)
        return neutral_positions

    def balance_control(self):
        """Balance control using ZMP"""
        com_state = {
            'position': self.com_position,
            'velocity': self.com_velocity,
            'acceleration': np.zeros(3)  # Approximate or estimate
        }

        balance_result = self.balance_controller.balance_control(com_state)

        # Convert balance corrections to joint space
        # This is a simplified mapping - real implementation would use
        # inverse kinematics to determine how to move joints to improve balance
        balance_joints = np.zeros(self.n_joints)

        if not balance_result['is_stable']:
            # Apply corrections to hip and ankle joints
            correction = balance_result['correction']

            # Map X correction to hip pitch joints
            balance_joints[2] += correction[0] * 0.05  # left_hip_pitch
            balance_joints[8] += correction[0] * 0.05  # right_hip_pitch

            # Map Y correction to hip roll joints
            balance_joints[1] += correction[1] * 0.03  # left_hip_roll
            balance_joints[7] += correction[1] * 0.03  # right_hip_roll

        return balance_joints

    def impedance_control(self):
        """Impedance control for compliant behavior"""
        commands = np.zeros(self.n_joints)

        for i in range(self.n_joints):
            controller = self.impedance_controllers[i]

            # Calculate impedance force: F = K(x_target - x_current) + D(v_target - v_current)
            position_error = controller['target_position'] - self.current_positions[i]
            velocity_error = 0.0 - self.current_velocities[i]  # Assuming target velocity = 0

            force = (controller['stiffness'] * position_error +
                    controller['damping'] * velocity_error)

            commands[i] = force

        return commands

    def mpc_control(self):
        """Model Predictive Control"""
        # Define current state for MPC
        # [com_x, com_y, com_z, com_dx, com_dy, com_dz]
        current_state = np.hstack([self.com_position, self.com_velocity])

        # Define reference trajectory (for now, maintain current CoM)
        reference_trajectory = self.mpc_controller.update_reference(
            current_state, current_state, steps_ahead=5
        )

        # Solve MPC problem
        mpc_control = self.mpc_controller.setup_mpc_problem(
            current_state, reference_trajectory
        )

        # Convert MPC control to joint commands
        # This is a simplified conversion
        joint_commands = np.zeros(self.n_joints)
        if mpc_control is not None:
            # Map control to relevant joints (simplified)
            for i in range(min(len(mpc_control), self.n_joints)):
                joint_commands[i] = mpc_control[i] * 0.1  # Scale appropriately

        return joint_commands

    def combine_control_levels(self, task_cmd, balance_cmd, impedance_cmd, mpc_cmd):
        """Combine commands from different control levels"""
        # Weighted combination of different control strategies
        weight_task = 0.3
        weight_balance = 0.4
        weight_impedance = 0.2
        weight_mpc = 0.1

        combined_commands = (
            weight_task * task_cmd +
            weight_balance * balance_cmd +
            weight_impedance * impedance_cmd +
            weight_mpc * mpc_cmd
        )

        return combined_commands

    def check_stability(self):
        """Check overall system stability"""
        # Check ZMP stability
        com_state = {
            'position': self.com_position,
            'velocity': self.com_velocity,
            'acceleration': np.zeros(3)
        }

        balance_result = self.balance_controller.balance_control(com_state)
        zmp_stable = balance_result['is_stable']

        # Check joint limits
        joint_limits_ok = np.all(np.abs(self.current_positions) < 2.0)  # Example limit

        # Check velocity limits
        velocity_ok = np.all(np.abs(self.current_velocities) < 5.0)  # Example limit

        return zmp_stable and joint_limits_ok and velocity_ok

    def publish_joint_commands(self, commands):
        """Publish joint commands to robot"""
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [0.0] * self.n_joints  # Position commands
        msg.velocity = commands.tolist()      # Velocity commands (for impedance control)
        msg.effort = [0.0] * self.n_joints   # Effort commands

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        self.joint_cmd_pub.publish(msg)

    def set_target_pose(self, pose_type='standing'):
        """Set target pose for the robot"""
        if pose_type == 'standing':
            # Set joints to standing position
            targets = np.array([
                0.0, 0.0, 0.0,  # left hip: yaw, roll, pitch
                0.0, 0.0, 0.0,  # left ankle: pitch, roll
                0.0, 0.0, 0.0,  # right hip: yaw, roll, pitch
                0.0, 0.0, 0.0   # right ankle: pitch, roll
            ])
        elif pose_type == 'ready':
            # Set joints to ready position (slightly bent knees)
            targets = np.array([
                0.0, 0.0, 0.1,  # left hip: yaw, roll, pitch (slight forward)
                0.2, 0.0, 0.0,  # left ankle: pitch, roll
                0.0, 0.0, 0.1,  # right hip: yaw, roll, pitch
                0.2, 0.0, 0.0   # right ankle: pitch, roll
            ])
        else:
            targets = np.zeros(self.n_joints)

        # Set targets in impedance controllers
        for i in range(self.n_joints):
            self.impedance_controllers[i]['target_position'] = targets[i]

def main():
    controller = AdvancedHumanoidController()

    try:
        # Set to standing position
        controller.set_target_pose('standing')

        # Start control loop
        controller.start_control_loop()

        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30 and not rospy.is_shutdown():
            time.sleep(1)

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down controller")
    finally:
        controller.stop_control_loop()

if __name__ == '__main__':
    main()
```

## Additional Resources
- Modern Robotics: Mechanics, Planning, and Control by Kevin Lynch and Frank Park
- Robot Dynamics and Control by Spong, Hutchinson, and Vidyasagar
- ROS Control Framework Documentation
- Model Predictive Control for Robotics
- Zero Moment Point: An Intuitive Guide for Walking Behaviors