---
sidebar_position: 3
title: "Chapter 3: Gazebo Simulation Environment"
---

# Chapter 3: Gazebo Simulation Environment

## Learning Outcomes
By the end of this chapter, students will be able to:
- Configure and operate Gazebo simulation environments
- Create and import 3D models for humanoid robots
- Implement physics-based simulation for robot dynamics
- Integrate Gazebo with ROS2 for realistic testing
- Design custom worlds and scenarios for humanoid testing
- Evaluate robot performance in simulated environments

## Overview

Gazebo is a powerful 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. For humanoid robotics, Gazebo serves as a crucial testing platform that allows developers to validate algorithms, test robot behaviors, and conduct experiments without the risks and costs associated with physical robots.

The simulation environment supports complex physics interactions, sensor simulation, and multi-robot scenarios. It integrates seamlessly with ROS2 through the Gazebo ROS packages, enabling comprehensive testing of humanoid robot systems before deployment on physical hardware.

## Gazebo Architecture and Components

### Core Components
- **Physics Engine**: Supports ODE, Bullet, Simbody, and DART for realistic physics simulation
- **Rendering Engine**: Provides high-quality 3D visualization
- **Sensor Simulation**: Implements cameras, LIDAR, IMU, force/torque sensors, and more
- **Plugin System**: Extensible architecture for custom functionality
- **World Editor**: Tool for creating and modifying simulation environments

### Physics Simulation Capabilities
- **Rigid Body Dynamics**: Accurate simulation of joint constraints and collisions
- **Contact Simulation**: Realistic friction, restitution, and contact forces
- **Fluid Simulation**: Approximation of fluid dynamics for underwater robotics
- **Terrain Simulation**: Support for complex outdoor environments
- **Multi-Body Dynamics**: Simulation of complex articulated systems

## Installing and Setting Up Gazebo

### Installation Requirements
```bash
# Install Gazebo Garden (latest stable version)
sudo apt install gazebo
# Or for the latest features, install Gazebo Harmonic
sudo apt install gz-harmonic
```

### ROS2 Integration Setup
```bash
# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-dev
```

### Environment Configuration
```bash
# Set Gazebo model paths
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models:/path/to/custom/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/path/to/custom/worlds
```

## Creating Humanoid Robot Models

### URDF (Unified Robot Description Format)
URDF is the standard format for describing robot models in ROS:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.2 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.2 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Neck Joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.25 0.1 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="5.0" velocity="1.0"/>
  </joint>

  <!-- Additional joints and links for complete humanoid model -->
  <!-- ... (legs, additional arms, etc.) ... -->

  <!-- Gazebo-specific tags -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Transmission for ROS control -->
  <transmission name="left_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

## Gazebo Plugins for Humanoid Robotics

### Sensor Plugins
Gazebo provides various sensor plugins for humanoid robots:

```xml
<!-- RGB-D Camera for perception -->
<gazebo reference="head">
  <sensor name="rgbd_camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU for balance control -->
<gazebo reference="base_link">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topic>__model__/imu</topic>
      <body_name>base_link</body_name>
      <frame_name>base_link</frame_name>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>

<!-- Force/Torque sensors for foot contact -->
<gazebo reference="left_foot">
  <sensor name="left_foot_ft" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="left_foot_ft_plugin" filename="libgazebo_ros_ft_sensor.so">
      <frame_name>left_foot_frame</frame_name>
      <topic>left_foot/ft_sensor</topic>
    </plugin>
  </sensor>
</gazebo>
```

### Controller Plugins
```xml
<!-- ROS2 Control interface -->
<gazebo>
  <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
    <parameters>$(find my_robot_description)/config/humanoid_control.yaml</parameters>
  </plugin>
</gazebo>
```

## World Creation and Environment Design

### Basic World File
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom objects -->
    <model name="table">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Humanoid robot spawn -->
    <include>
      <name>humanoid_robot</name>
      <pose>0 0 1 0 0 0</pose>
      <uri>model://humanoid_robot</uri>
    </include>
  </world>
</sdf>
```

### Advanced World Features
- **Terrain Generation**: Create realistic outdoor environments
- **Dynamic Objects**: Moving obstacles and interactive elements
- **Lighting Effects**: Simulate different times of day
- **Weather Simulation**: Add fog, rain, or other atmospheric effects

## ROS2 Integration

### Launch File for Gazebo Simulation
```python
# launch/humanoid_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'humanoid_lab.world'
            ])
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{
            'use_sim_time': True,
            'robot_description': Command(['xacro ', PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'humanoid_robot.urdf.xacro'
            ])])
        }]
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
        robot_state_publisher
    ])
```

### Controller Configuration
```yaml
# config/humanoid_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    humanoid_controller:
      type: position_controllers/JointGroupPositionController

humanoid_controller:
  ros__parameters:
    joints:
      - neck_joint
      - left_shoulder
      - left_elbow
      - right_shoulder
      - right_elbow
      - left_hip
      - left_knee
      - left_ankle
      - right_hip
      - right_knee
      - right_ankle
```

## Simulation Scenarios for Humanoid Testing

### Balance and Locomotion Testing
- **Static Balance**: Test ability to maintain upright position
- **Dynamic Walking**: Evaluate gait patterns and stability
- **Push Recovery**: Assess response to external disturbances
- **Stair Navigation**: Test complex terrain traversal

### Manipulation Tasks
- **Object Grasping**: Test precision and force control
- **Tool Use**: Evaluate complex manipulation skills
- **Assembly Tasks**: Assess multi-step operations
- **Human-Robot Interaction**: Test collaborative tasks

### Perception Challenges
- **Object Recognition**: Test vision systems in various lighting
- **SLAM Performance**: Evaluate mapping and localization
- **Human Detection**: Assess social interaction capabilities
- **Scene Understanding**: Test higher-level perception

## Performance Evaluation in Simulation

### Metrics for Humanoid Performance
- **Stability Metrics**: Center of Mass tracking, Zero Moment Point
- **Efficiency Metrics**: Energy consumption, computational load
- **Accuracy Metrics**: Task completion success rate
- **Robustness Metrics**: Recovery from disturbances

### Data Collection and Analysis
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray

class SimulationEvaluator(Node):
    def __init__(self):
        super().__init__('simulation_evaluator')

        # Subscriptions
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.com_sub = self.create_subscription(
            PointStamped, '/center_of_mass', self.com_callback, 10)

        # Publishers for metrics
        self.stability_pub = self.create_publisher(
            Float64MultiArray, '/stability_metrics', 10)

        # Data collection
        self.joint_data = []
        self.com_data = []
        self.time_start = self.get_clock().now()

    def joint_callback(self, msg):
        self.joint_data.append((self.get_clock().now(), msg))

    def com_callback(self, msg):
        self.com_data.append((self.get_clock().now(), msg))

    def calculate_stability_metrics(self):
        # Calculate ZMP, CoM deviation, etc.
        stability_data = Float64MultiArray()
        # ... implementation ...
        return stability_data
```

## Best Practices for Gazebo Simulation

### Model Optimization
- Use simplified collision geometries for better performance
- Implement level-of-detail models for distant objects
- Optimize mesh complexity while maintaining accuracy
- Use appropriate physics parameters for realistic behavior

### Simulation Accuracy
- Validate simulation results against physical robot behavior
- Tune physics parameters to match real-world dynamics
- Include sensor noise models for realistic perception
- Account for actuator limitations and delays

### Performance Optimization
- Use appropriate update rates for different components
- Limit simulation complexity for real-time performance
- Implement efficient collision checking
- Use multi-threading where appropriate

## Weekly Breakdown for Chapter 3
- **Week 3.1**: Gazebo installation and basic usage
- **Week 3.2**: Robot model creation and URDF
- **Week 3.3**: Sensor integration and plugin development
- **Week 3.4**: World creation and simulation scenarios

## Assessment
- **Quiz 3.1**: Gazebo architecture and components (Multiple choice and short answer)
- **Assignment 3.2**: Create a humanoid robot model in Gazebo
- **Lab Exercise 3.1**: Implement a walking controller in simulation

## Diagram Placeholders
- ![Gazebo Architecture and Components](./images/gazebo_architecture.png)
- ![Humanoid Robot Model in Gazebo](./images/humanoid_gazebo_model.png)
- ![Simulation Workflow for Humanoid Robotics](./images/gazebo_simulation_workflow.png)

## Code Snippet: Gazebo World Spawn Script
```python
#!/usr/bin/env python3

import subprocess
import time
import rospy
from geometry_msgs.msg import Pose

def spawn_humanoid_robot(x=0, y=0, z=1, model_name="humanoid_robot"):
    """
    Spawn a humanoid robot model in Gazebo at specified coordinates
    """
    spawn_cmd = [
        "ros2", "run", "gazebo_ros", "spawn_entity.py",
        "-entity", model_name,
        "-x", str(x), "-y", str(y), "-z", str(z),
        "-database", "humanoid_robot"
    ]

    try:
        result = subprocess.run(spawn_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully spawned {model_name} at ({x}, {y}, {z})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to spawn {model_name}: {e}")
        return False

if __name__ == "__main__":
    # Initialize ROS node
    import rclpy
    rclpy.init()

    # Spawn robot in simulation
    success = spawn_humanoid_robot()

    if success:
        print("Robot spawned successfully in Gazebo simulation")
    else:
        print("Failed to spawn robot in simulation")

    rclpy.shutdown()
```

## Additional Resources
- Gazebo Documentation: http://gazebosim.org/
- ROS2 Gazebo Tutorials: https://classic.gazebosim.org/tutorials?cat=connect_ros
- Gazebo Community Forum
- Example humanoid robot models on GitHub