---
sidebar_position: 2
title: "Chapter 2: ROS2 Fundamentals for Robotics"
---

# Chapter 2: ROS2 Fundamentals for Robotics

## Learning Outcomes
By the end of this chapter, students will be able to:
- Understand the architecture and concepts of ROS2
- Create and manage ROS2 packages and nodes
- Implement publishers and subscribers for communication
- Use services and actions for complex interactions
- Configure and launch ROS2 systems
- Integrate ROS2 with other robotics frameworks

## Overview

Robot Operating System 2 (ROS2) is the next-generation framework for developing robotic applications. Unlike its predecessor, ROS2 addresses critical issues such as real-time performance, multi-robot systems, and industrial deployment. This chapter covers the essential concepts and practical implementation of ROS2 for humanoid robotics applications.

ROS2 provides a middleware-based architecture that enables communication between different robotic components, regardless of programming language or operating system. It offers tools for visualization, simulation, debugging, and building complex robotic systems.

## ROS2 Architecture

ROS2 uses a distributed system architecture based on the Data Distribution Service (DDS) middleware. This provides:

### Key Concepts:
- **Nodes**: Individual processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Messages**: Simple data structures exchanged between nodes
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goal-oriented communication
- **Parameters**: Configuration values accessible to nodes
- **Launch files**: Scripts to start multiple nodes simultaneously

### DDS Middleware
- Provides reliable message delivery
- Supports real-time communication requirements
- Enables multi-robot coordination
- Offers built-in discovery mechanisms

## Setting Up ROS2 Environment

### Installation Requirements
```bash
# Ubuntu 22.04 LTS recommended
# Install ROS2 Humble Hawksbill (LTS)
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Add to ~/.bashrc for automatic sourcing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating ROS2 Packages

### Package Structure
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
├── include/
├── launch/
├── config/
├── scripts/
└── test/
```

### Creating a New Package
```bash
# Create package with dependencies
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs geometry_msgs my_robot_package
```

## Nodes and Communication

### Creating a Publisher Node (C++)
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("humanoid_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("robot_status", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Humanoid status: operational - " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

### Creating a Subscriber Node (Python)
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('humanoid_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received robot status: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Actions

### Services for Synchronous Communication
Services provide request/response communication pattern:

```python
# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('humanoid_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        return response
```

### Actions for Asynchronous Goals
Actions are ideal for long-running tasks:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Launch Files for System Configuration

Launch files allow starting multiple nodes simultaneously:

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='humanoid_controller',
            name='controller',
            parameters=[
                {'kp': 1.0},
                {'ki': 0.1},
                {'kd': 0.05}
            ]
        ),
        Node(
            package='my_robot_package',
            executable='humanoid_sensor_processor',
            name='sensor_processor'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', 'path/to/config.rviz']
        )
    ])
```

## ROS2 for Humanoid Robotics

### Specific Considerations for Humanoid Systems

#### Real-time Requirements
- Use real-time kernel for critical control loops
- Configure DDS for low-latency communication
- Implement proper thread management

#### Multi-Node Architecture
```
Humanoid Robot System:
├── Perception Nodes
│   ├── Vision Processing
│   ├── Audio Processing
│   └── Sensor Fusion
├── Control Nodes
│   ├── Balance Controller
│   ├── Motion Planner
│   └── Trajectory Generator
├── Planning Nodes
│   ├── Path Planning
│   ├── Task Planning
│   └── Grasping Planning
└── Interface Nodes
    ├── Human-Robot Interaction
    ├── Teleoperation
    └── Monitoring
```

#### Safety and Coordination
- Implement emergency stop mechanisms
- Use action servers for critical operations
- Implement proper error handling and recovery

## Integration with Other Frameworks

### ROS2 with Gazebo Simulation
```xml
<!-- package.xml dependencies -->
<depend>gazebo_ros_pkgs</depend>
<depend>gazebo_ros2_control</depend>
```

### ROS2 with NVIDIA Isaac
- Use ROS2 bridge for Isaac ecosystem
- Integrate perception and planning modules
- Leverage GPU acceleration

### ROS2 with Unity
- Use ROS2Sharp for Unity integration
- Implement communication bridges
- Synchronize simulation states

## Weekly Breakdown for Chapter 2
- **Week 2.1**: ROS2 architecture and concepts
- **Week 2.2**: Node creation and communication patterns
- **Week 2.3**: Services, actions, and parameters
- **Week 2.4**: Launch files and system integration

## Assessment
- **Quiz 2.1**: ROS2 concepts and architecture (Multiple choice and short answer)
- **Assignment 2.2**: Create a ROS2 package for humanoid robot control
- **Lab Exercise 2.1**: Implement publisher/subscriber for robot joint states

## Diagram Placeholders
- ![ROS2 Architecture Diagram](./images/ros2_architecture.png)
- ![Node Communication Patterns](./images/ros2_communication.png)
- ![Humanoid ROS2 System Integration](./images/humanoid_ros2_integration.png)

## Code Snippet: Complete Humanoid Controller Node
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.emergency_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_callback, 10)

        # Timers
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.joint_states = JointState()
        self.emergency_stop = False

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def emergency_callback(self, msg):
        self.emergency_stop = msg.data

    def control_loop(self):
        if not self.emergency_stop:
            # Implement humanoid control logic here
            pass

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Additional Resources
- ROS2 Documentation: https://docs.ros.org/en/humble/
- ROS2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- Robotics Stack Exchange for ROS questions
- GitHub repositories with ROS2 examples