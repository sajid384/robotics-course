---
sidebar_position: 8
title: "Chapter 8: Laboratory Exercises Part 1"
---

# Chapter 8: Laboratory Exercises Part 1

## Learning Outcomes
By the end of this chapter, students will be able to:
- Set up and configure robotic simulation environments
- Implement basic robot control algorithms
- Integrate sensors and actuators in a robotic system
- Debug and troubleshoot common robotics issues
- Evaluate robot performance using quantitative metrics
- Document and report experimental results

## Overview

Laboratory exercises form a critical component of the Physical AI and Humanoid Robotics course, providing hands-on experience with the theoretical concepts covered in previous chapters. This first part of the laboratory exercises focuses on foundational skills including simulation setup, basic control algorithms, and system integration. These exercises build progressively from simple tasks to more complex robotic behaviors.

The exercises are designed to be performed in both simulation and with physical robots when available. Simulation provides a safe, repeatable environment for initial development and testing, while physical robots offer real-world validation of concepts.

## Lab Exercise 8.1: Setting Up the Simulation Environment

### Objective
Configure and validate a complete simulation environment for humanoid robotics using Gazebo and ROS2.

### Equipment Required
- Computer with Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill installed
- Gazebo Garden or later
- Basic understanding of Linux terminal

### Procedure

#### Step 1: Verify Installation
```bash
# Check ROS2 installation
source /opt/ros/humble/setup.bash
echo $ROS_DISTRO

# Check Gazebo installation
gazebo --version

# Check required packages
dpkg -l | grep ros-humble-gazebo
```

#### Step 2: Create Workspace
```bash
# Create workspace directory
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws

# Build workspace
colcon build --packages-select gazebo_ros_pkgs
source install/setup.bash
```

#### Step 3: Launch Basic Simulation
```bash
# Launch empty world
ros2 launch gazebo_ros empty_world.launch.py

# In another terminal, verify connection
ros2 topic list
```

#### Step 4: Create Custom Robot Model
Create a simple robot URDF file:

```xml
<!-- ~/robotics_ws/src/my_robot_description/urdf/simple_humanoid.urdf -->
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>
</robot>
```

#### Step 5: Launch Robot in Simulation
```bash
# Create launch file
mkdir -p ~/robotics_ws/src/my_robot_description/launch
```

Create launch file:

```python
# ~/robotics_ws/src/my_robot_description/launch/simple_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import TextSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')
    my_robot_description_dir = get_package_share_directory('my_robot_description')

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_package_dir, 'launch', 'gazebo.launch.py')
        )
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(
                os.path.join(my_robot_description_dir, 'urdf', 'simple_humanoid.urdf')
            ).read()
        }]
    )

    # Spawn robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

### Expected Results
- Gazebo simulation window opens
- Simple humanoid robot model appears in the world
- Robot model is stable and properly positioned
- All joints are visible and connected correctly

### Troubleshooting
- If robot doesn't appear: Check URDF syntax and joint connections
- If Gazebo doesn't start: Verify installation and permissions
- If joints are unstable: Check inertial properties and limits

## Lab Exercise 8.2: Basic Joint Control

### Objective
Implement and test basic joint control for the simulated humanoid robot.

### Equipment Required
- Completed Lab 8.1 setup
- ROS2 workspace with robot model

### Procedure

#### Step 1: Create Joint Controller Node
```python
# ~/robotics_ws/src/my_robot_controller/my_robot_controller/joint_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize joint positions
        self.joint_positions = {
            'neck_joint': 0.0
        }

        self.get_logger().info('Joint Controller initialized')

    def control_loop(self):
        """Main control loop"""
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set joint names
        msg.name = list(self.joint_positions.keys())

        # Calculate oscillating motion for neck joint
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.joint_positions['neck_joint'] = 0.4 * math.sin(current_time)

        # Set joint positions
        msg.position = list(self.joint_positions.values())

        # Publish joint commands
        self.joint_pub.publish(msg)

        self.get_logger().info(f'Published joint positions: {self.joint_positions}')

def main(args=None):
    rclpy.init(args=args)
    controller = JointController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down joint controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Configure Joint State Controller
Create controller configuration file:

```yaml
# ~/robotics_ws/src/my_robot_controller/config/joint_state_controller.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_controller:
      type: joint_state_controller/JointStateController

joint_state_controller:
  ros__parameters:
    interface_name: position
```

#### Step 3: Create Controller Launch File
```python
# ~/robotics_ws/src/my_robot_controller/launch/controller.launch.py
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch joint controller
    joint_controller = Node(
        package='my_robot_controller',
        executable='joint_controller',
        name='joint_controller',
        output='screen'
    )

    # Load and start joint state controller
    joint_state_controller = Node(
        package='controller_manager',
        executable='spawner.py',
        arguments=['joint_state_controller'],
        output='screen'
    )

    # Load and start robot controller
    robot_controller = Node(
        package='controller_manager',
        executable='spawner.py',
        arguments=['robot_controller'],
        output='screen'
    )

    # Set up event handler to start controllers after robot is spawned
    delay_joint_state_after_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_controller,
            on_exit=[joint_state_controller],
        )
    )

    return LaunchDescription([
        joint_controller,
        delay_joint_state_after_spawner,
        robot_controller
    ])
```

#### Step 4: Test Joint Control
```bash
# Terminal 1: Launch simulation
cd ~/robotics_ws
source install/setup.bash
ros2 launch my_robot_description simple_humanoid.launch.py

# Terminal 2: Run controller
cd ~/robotics_ws
source install/setup.bash
ros2 run my_robot_controller joint_controller
```

### Expected Results
- Robot's neck joint moves in an oscillating pattern
- Joint positions published at 10Hz
- Smooth, continuous motion without jerks
- No errors in the console output

### Troubleshooting
- If joint doesn't move: Check controller configuration and joint names
- If motion is jerky: Increase control frequency or adjust gains
- If errors occur: Verify package dependencies and permissions

## Lab Exercise 8.3: Sensor Integration and Data Processing

### Objective
Integrate basic sensors (IMU, camera) with the robot model and process the sensor data.

### Equipment Required
- Completed Lab 8.1 and 8.2
- Working robot simulation

### Procedure

#### Step 1: Add Sensors to Robot Model
Update the URDF file to include sensors:

```xml
<!-- Add to ~/robotics_ws/src/my_robot_description/urdf/simple_humanoid.urdf -->
<!-- Add IMU sensor -->
<gazebo reference="head">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/simple_humanoid</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>

<!-- Add camera sensor -->
<gazebo reference="head">
  <sensor name="camera_sensor" type="camera">
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
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/simple_humanoid</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

#### Step 2: Create Sensor Processing Node
```python
# ~/robotics_ws/src/my_robot_controller/my_robot_controller/sensor_processor.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/simple_humanoid/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/simple_humanoid/camera/image_raw', self.camera_callback, 10)

        # Create publishers
        self.roll_pitch_pub = self.create_publisher(
            Float32, '/robot_orientation/roll', 10)
        self.pitch_pub = self.create_publisher(
            Float32, '/robot_orientation/pitch', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize variables
        self.imu_data = None
        self.camera_data = None

        self.get_logger().info('Sensor Processor initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation from quaternion
        orientation_q = msg.orientation
        r = R.from_quat([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

        # Convert to roll, pitch, yaw
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        # Publish roll and pitch
        roll_msg = Float32()
        roll_msg.data = float(roll)
        self.roll_pitch_pub.publish(roll_msg)

        pitch_msg = Float32()
        pitch_msg.data = float(pitch)
        self.pitch_pub.publish(pitch_msg)

        self.get_logger().info(f'Roll: {roll:.2f}°, Pitch: {pitch:.2f}°')

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform basic image processing (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Display image (optional for debugging)
            cv2.imshow('Camera Feed', cv_image)
            cv2.imshow('Edges', edges)
            cv2.waitKey(1)

            # Process image data for perception
            self.process_image_features(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def process_image_features(self, image):
        """Process image for feature extraction"""
        # Example: Find contours in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count objects in the image
        object_count = len(contours)

        if object_count > 0:
            self.get_logger().info(f'Detected {object_count} objects in the image')

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down sensor processor')
    finally:
        cv2.destroyAllWindows()
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Launch Sensor Integration Test
```bash
# Terminal 1: Launch simulation with sensors
cd ~/robotics_ws
source install/setup.bash
ros2 launch my_robot_description simple_humanoid.launch.py

# Terminal 2: Run sensor processor
cd ~/robotics_ws
source install/setup.bash
ros2 run my_robot_controller sensor_processor
```

### Expected Results
- IMU data processed and orientation calculated
- Camera feed displayed with edge detection
- Roll and pitch values published at 100Hz
- Object detection working in camera feed

### Troubleshooting
- If IMU data is not received: Check Gazebo plugin configuration
- If camera feed is black: Verify camera parameters in URDF
- If processing is slow: Reduce image resolution or processing complexity

## Lab Exercise 8.4: Basic Locomotion Control

### Objective
Implement a basic walking controller for a simplified bipedal robot model.

### Equipment Required
- Completed previous labs
- Working simulation environment
- Basic understanding of control systems

### Procedure

#### Step 1: Create Bipedal Robot Model
```xml
<!-- ~/robotics_ws/src/my_robot_description/urdf/bipedal_robot.urdf -->
<?xml version="1.0"?>
<robot name="bipedal_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left leg -->
  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right leg (similar to left) -->
  <link name="right_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <link name="right_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_hip" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.0 0.1 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1"/>
  </joint>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.78" upper="0.78" effort="50" velocity="1"/>
  </joint>

  <joint name="right_hip" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0.0 -0.1 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1"/>
  </joint>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.78" upper="0.78" effort="50" velocity="1"/>
  </joint>

  <!-- Gazebo plugins for control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/bipedal_robot</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

#### Step 2: Create Walking Controller
```python
# ~/robotics_ws/src/my_robot_controller/my_robot_controller/walking_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Header
import math
import numpy as np

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/bipedal_robot/joint_commands', 10)

        # Subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu, '/bipedal_robot/imu/data', self.imu_callback, 10)

        # Timer for control loop
        self.timer = self.create_timer(0.02, self.control_loop)  # 50Hz

        # Initialize joint positions
        self.joint_positions = {
            'left_hip': 0.0,
            'left_knee': 0.0,
            'left_ankle': 0.0,
            'right_hip': 0.0,
            'right_knee': 0.0,
            'right_ankle': 0.0
        }

        # Walking parameters
        self.walking_frequency = 0.5  # Hz
        self.walking_amplitude = 0.3   # radians
        self.walking_phase_offset = math.pi  # phase difference between legs

        # IMU data
        self.imu_roll = 0.0
        self.imu_pitch = 0.0

        # Walking state
        self.walk_enabled = True
        self.walk_speed = 0.0  # 0.0 to 1.0

        self.get_logger().info('Walking Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Extract roll and pitch from orientation
        # Convert quaternion to Euler angles
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.imu_roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.imu_pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.imu_pitch = math.asin(sinp)

    def control_loop(self):
        """Main walking control loop"""
        if self.walk_enabled:
            # Calculate walking pattern
            current_time = self.get_clock().now().nanoseconds / 1e9
            phase = 2 * math.pi * self.walking_frequency * current_time

            # Generate walking pattern with balance correction
            self.generate_walking_pattern(phase)

            # Apply balance corrections
            self.apply_balance_control()

            # Publish joint commands
            self.publish_joint_commands()

        self.get_logger().info(f'Walking - Roll: {math.degrees(self.imu_roll):.2f}°, Pitch: {math.degrees(self.imu_pitch):.2f}°')

    def generate_walking_pattern(self, phase):
        """Generate walking pattern for both legs"""
        # Left leg pattern
        self.joint_positions['left_hip'] = self.walking_amplitude * math.sin(phase) * self.walk_speed
        self.joint_positions['left_knee'] = self.walking_amplitude * 0.5 * math.sin(phase + math.pi/2) * self.walk_speed
        self.joint_positions['left_ankle'] = -self.walking_amplitude * 0.3 * math.sin(phase) * self.walk_speed

        # Right leg pattern (opposite phase)
        self.joint_positions['right_hip'] = self.walking_amplitude * math.sin(phase + self.walking_phase_offset) * self.walk_speed
        self.joint_positions['right_knee'] = self.walking_amplitude * 0.5 * math.sin(phase + math.pi/2 + self.walking_phase_offset) * self.walk_speed
        self.joint_positions['right_ankle'] = -self.walking_amplitude * 0.3 * math.sin(phase + self.walking_phase_offset) * self.walk_speed

    def apply_balance_control(self):
        """Apply balance corrections based on IMU data"""
        # Simple PD controller for balance
        roll_correction = -self.imu_roll * 2.0  # PD gain for roll
        pitch_correction = -self.imu_pitch * 1.5  # PD gain for pitch

        # Apply corrections to hip joints
        self.joint_positions['left_hip'] += roll_correction
        self.joint_positions['right_hip'] -= roll_correction
        self.joint_positions['left_hip'] += pitch_correction
        self.joint_positions['right_hip'] += pitch_correction

        # Apply corrections to ankle joints
        self.joint_positions['left_ankle'] += roll_correction * 0.5
        self.joint_positions['right_ankle'] -= roll_correction * 0.5

    def publish_joint_commands(self):
        """Publish joint state commands"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())

        # Set velocities and efforts to zero (or calculate based on position)
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        self.joint_pub.publish(msg)

    def enable_walking(self, enable=True):
        """Enable or disable walking"""
        self.walk_enabled = enable
        if not enable:
            # Return to neutral position
            for joint in self.joint_positions:
                self.joint_positions[joint] = 0.0

def main(args=None):
    rclpy.init(args=args)
    controller = WalkingController()

    try:
        # Start walking after a short delay
        def start_walking():
            controller.walk_speed = 0.3
            controller.get_logger().info('Starting walking controller')

        # Schedule start after 2 seconds
        controller.create_timer(2.0, start_walking)

        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down walking controller')
        controller.enable_walking(False)
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Launch Walking Controller Test
```bash
# Terminal 1: Launch bipedal robot simulation
cd ~/robotics_ws
source install/setup.bash
# Create launch file for bipedal robot and launch it

# Terminal 2: Run walking controller
cd ~/robotics_ws
source install/setup.bash
ros2 run my_robot_controller walking_controller
```

### Expected Results
- Robot maintains balance using IMU feedback
- Legs move in coordinated walking pattern
- Robot remains upright during walking motion
- Smooth transitions between walking phases

### Troubleshooting
- If robot falls: Adjust balance control gains or initial positions
- If walking is unstable: Reduce walking frequency or amplitude
- If joints move erratically: Check joint limits and control parameters

## Weekly Breakdown for Chapter 8
- **Week 8.1**: Lab Exercise 8.1 - Simulation Environment Setup
- **Week 8.2**: Lab Exercise 8.2 - Basic Joint Control
- **Week 8.3**: Lab Exercise 8.3 - Sensor Integration
- **Week 8.4**: Lab Exercise 8.4 - Basic Locomotion Control

## Assessment
- **Lab Report 8.1**: Simulation environment setup and validation
- **Lab Report 8.2**: Joint control implementation and testing
- **Lab Report 8.3**: Sensor integration and data processing
- **Lab Report 8.4**: Walking controller development and performance

## Diagram Placeholders
- ![Simulation Environment Setup](./images/lab_simulation_setup.png)
- ![Robot Joint Control Architecture](./images/lab_joint_control.png)
- ![Sensor Integration Diagram](./images/lab_sensor_integration.png)
- ![Walking Pattern Visualization](./images/lab_walking_pattern.png)

## Code Snippet: Complete Laboratory Control System
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Header, Bool, Float32
from geometry_msgs.msg import Twist
import math
import numpy as np
import time

class LaboratoryControlSystem(Node):
    """
    Complete laboratory control system integrating all components
    """
    def __init__(self):
        super().__init__('lab_control_system')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/robot/joint_commands', 10)
        self.status_pub = self.create_publisher(Bool, '/robot/operational', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/robot/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/robot/imu/data', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Timer for main control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz

        # Initialize state variables
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.desired_velocity = {'linear': 0.0, 'angular': 0.0}
        self.robot_state = {'operational': True, 'balance': 0.0}

        # Control parameters
        self.control_mode = 'idle'  # idle, walking, manipulation
        self.balance_threshold = 0.3  # radians
        self.safety_enabled = True

        self.get_logger().info('Laboratory Control System initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions and velocities"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Convert quaternion to Euler angles
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.imu_data = {'roll': roll, 'pitch': pitch, 'yaw': yaw}

        # Calculate balance metric
        self.robot_state['balance'] = math.sqrt(roll**2 + pitch**2)

    def cmd_vel_callback(self, msg):
        """Receive velocity commands"""
        self.desired_velocity['linear'] = msg.linear.x
        self.desired_velocity['angular'] = msg.angular.z

    def control_loop(self):
        """Main control loop"""
        # Safety check
        if self.safety_enabled and self.robot_state['balance'] > self.balance_threshold:
            self.emergency_stop()
            return

        # Execute control based on mode
        if self.control_mode == 'walking':
            commands = self.generate_walking_commands()
        elif self.control_mode == 'manipulation':
            commands = self.generate_manipulation_commands()
        else:  # idle
            commands = self.generate_idle_commands()

        # Publish joint commands
        self.publish_joint_commands(commands)

        # Publish operational status
        status_msg = Bool()
        status_msg.data = self.robot_state['operational']
        self.status_pub.publish(status_msg)

        # Log status
        self.get_logger().info(
            f'Control Mode: {self.control_mode}, '
            f'Balance: {self.robot_state["balance"]:.3f}, '
            f'Operational: {self.robot_state["operational"]}'
        )

    def generate_walking_commands(self):
        """Generate walking pattern commands"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Walking pattern parameters
        frequency = 0.5 + abs(self.desired_velocity['linear']) * 0.5  # Adjust frequency based on desired speed
        amplitude = 0.3 * min(abs(self.desired_velocity['linear']), 1.0)  # Adjust amplitude based on desired speed
        phase = 2 * math.pi * frequency * current_time

        # Balance correction
        roll_correction = -self.imu_data['roll'] * 2.0
        pitch_correction = -self.imu_data['pitch'] * 1.5

        commands = {}

        # Generate walking pattern with balance correction
        commands['left_hip'] = amplitude * math.sin(phase) + roll_correction + pitch_correction
        commands['left_knee'] = amplitude * 0.5 * math.sin(phase + math.pi/2)
        commands['left_ankle'] = -amplitude * 0.3 * math.sin(phase) + roll_correction * 0.5

        commands['right_hip'] = amplitude * math.sin(phase + math.pi) - roll_correction + pitch_correction
        commands['right_knee'] = amplitude * 0.5 * math.sin(phase + math.pi/2 + math.pi)
        commands['right_ankle'] = -amplitude * 0.3 * math.sin(phase + math.pi) - roll_correction * 0.5

        return commands

    def generate_manipulation_commands(self):
        """Generate manipulation commands"""
        # Placeholder for manipulation control
        # This would implement arm movement patterns
        return {
            'left_shoulder': 0.1,
            'left_elbow': 0.2,
            'right_shoulder': 0.1,
            'right_elbow': 0.2
        }

    def generate_idle_commands(self):
        """Generate idle position commands"""
        return {
            'left_hip': 0.0,
            'left_knee': 0.0,
            'left_ankle': 0.0,
            'right_hip': 0.0,
            'right_knee': 0.0,
            'right_ankle': 0.0
        }

    def publish_joint_commands(self, commands):
        """Publish joint commands to robot"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = list(commands.keys())
        msg.position = list(commands.values())
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        self.joint_cmd_pub.publish(msg)

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.robot_state['operational'] = False
        self.control_mode = 'idle'

        # Publish zero commands to all joints
        zero_commands = {}
        for joint in self.current_joint_positions:
            zero_commands[joint] = 0.0

        self.publish_joint_commands(zero_commands)

        self.get_logger().error('EMERGENCY STOP: Robot balance exceeded threshold')

    def set_control_mode(self, mode):
        """Set control mode (walking, manipulation, idle)"""
        if mode in ['idle', 'walking', 'manipulation']:
            self.control_mode = mode
            self.get_logger().info(f'Switched to control mode: {mode}')
            return True
        else:
            self.get_logger().error(f'Invalid control mode: {mode}')
            return False

def main(args=None):
    rclpy.init(args=args)
    control_system = LaboratoryControlSystem()

    try:
        # Example: Set to walking mode after 5 seconds
        def enable_walking():
            control_system.set_control_mode('walking')
            control_system.get_logger().info('Walking mode enabled')

        # Schedule walking after 5 seconds
        control_system.create_timer(5.0, enable_walking)

        rclpy.spin(control_system)
    except KeyboardInterrupt:
        control_system.get_logger().info('Shutting down laboratory control system')
        control_system.emergency_stop()
    finally:
        control_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Additional Resources
- ROS2 Control Tutorials
- Gazebo Simulation Tutorials
- Robot Operating System Documentation
- Control Systems Engineering Resources
- Safety Guidelines for Humanoid Robots