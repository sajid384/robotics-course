# Chatbot Features and Functionality

## Overview
The Physical AI and Humanoid Robotics Course includes an intelligent chatbot assistant that helps students ask questions about the course content. The chatbot is integrated throughout the documentation site and provides a dedicated page for focused interaction.

## Features

### 1. Floating Widget
- Available on all pages as a floating chat widget
- Accessible via the "💬 Course Assistant" button
- Can be opened/closed as needed

### 2. Dedicated Chatbot Page
- Accessible at `/chatbot`
- Full-screen chat experience
- Optimized for focused interaction

### 3. Knowledge Base
The chatbot has a comprehensive knowledge base covering all 14 chapters of the course:

- **Introduction**: Fundamental concepts of Physical AI and Humanoid Robotics
- **ROS2**: Robot Operating System 2 fundamentals
- **Gazebo**: Simulation environment
- **Unity**: Unity for robotics simulation
- **NVIDIA Isaac**: Isaac platform for robotics
- **Vision-Language-Action (VLA) Models**: Multimodal learning systems
- **Hardware Components**: Actuators, sensors, computing platforms
- **Laboratory Exercises**: Practical implementation
- **Edge AI**: Computing platforms and optimization
- **Perception Systems**: Environmental understanding
- **Control Systems**: Balance and locomotion control
- **Learning and Adaptation**: Reinforcement and imitation learning
- **Ethics and Safety**: Ethical frameworks and safety standards
- **Future Directions**: Emerging technologies and research

### 4. Chapter-Specific Information
The chatbot can provide detailed information about specific chapters:
- Ask about "Chapter 1", "Chapter 2", etc.
- Ask about "Lesson X" or "ch. X"

### 5. GitHub Integration
- Direct link to the GitHub repository within the chatbot interface
- Easy access to source code and course materials

## How It Works

The chatbot uses a keyword matching system to respond to user queries:

1. User types a question about the course content
2. The system checks for keywords in the knowledge base
3. If keywords are found, it returns the relevant information
4. If chapter-related queries are detected, it returns chapter-specific content
5. If no specific topic is found, it provides a general overview

## Technical Implementation

- **Frontend**: React-based component with Docusaurus integration
- **Knowledge Base**: Comprehensive rule-based system with predefined responses
- **UI/UX**: Responsive design with typing indicators and message history
- **Integration**: Available site-wide through Docusaurus theme components

## Access

- **Floating Widget**: Available on all pages (bottom-right corner)
- **Dedicated Page**: Navigate to `/chatbot`
- **Footer Link**: Access through the "More" section in the footer

## GitHub Repository

The complete source code for this chatbot and the entire course is available at:
[https://github.com/syedsajidhussain/robotics-course](https://github.com/syedsajidhussain/robotics-course)