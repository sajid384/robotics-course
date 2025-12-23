import React, { useState, useEffect } from 'react';
import './BookChatbot.css';

const BookChatbot = ({ isPage = false }) => {
  const [isOpen, setIsOpen] = useState(isPage); // Start open if on a dedicated page
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your Physical AI and Humanoid Robotics course assistant. Ask me anything about the course content!", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Course knowledge base - comprehensive for all chapters
  const knowledgeBase = {
    "introduction": "The Physical AI and Humanoid Robotics course introduces the fundamental concepts of embodied artificial intelligence and humanoid robotics. It covers the integration of AI, mechanics, electronics, and human-centered design. The course explores how robots can understand, reason, and act in the physical world.",

    "ros2": "ROS2 (Robot Operating System 2) is the next-generation framework for developing robotic applications. It provides middleware-based architecture for communication between different robotic components. Key features include improved security, real-time support, and better multi-robot systems support.",

    "gazebo": "Gazebo is a powerful 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces for robotics development. It's widely used for testing and validating robotic algorithms before deployment on real hardware.",

    "unity": "Unity for robotics provides high-fidelity graphics and an extensive ecosystem of tools. It's particularly valuable for training perception systems that need to operate in real-world conditions. Unity integrates well with ROS/ROS2 through packages like Unity Robotics Package.",

    "nvidia isaac": "The NVIDIA Isaac platform is a comprehensive solution for developing, simulating, and deploying AI-powered robots. It includes Isaac Sim for simulation, Isaac ROS for GPU-accelerated perception, and Isaac Apps for reference applications.",

    "vla": "Vision-Language-Action (VLA) models represent a significant advancement in embodied AI, enabling robots to understand natural language commands, perceive their environment visually, and execute appropriate actions. These models allow for more intuitive human-robot interaction.",

    "hardware": "Hardware components include actuators (servos, DC motors, Series Elastic Actuators), sensors (IMU, cameras, force/torque sensors), computing platforms (NVIDIA Jetson, etc.), and structural elements. Proper integration of these components is crucial for humanoid robot functionality.",

    "perception": "Perception systems integrate multiple sensor modalities (cameras, LIDAR, IMU, tactile sensors) to provide comprehensive environmental awareness for humanoid robots. This includes object recognition, scene understanding, and state estimation.",

    "control": "Control systems manage robot behavior using various techniques including PID control, Model Predictive Control (MPC), and advanced methods like Zero Moment Point (ZMP) for balance control in humanoid robots.",

    "learning": "The course covers reinforcement learning, imitation learning, adaptive control, and lifelong learning systems for humanoid robots. These techniques enable robots to improve their performance over time and adapt to new situations.",

    "ethics": "Ethics in robotics addresses safety standards, privacy protection, human-robot interaction ethics, and societal impact considerations. This includes standards like ISO 13482 for personal care robots.",

    "future": "Future directions include neuromorphic computing, soft robotics, quantum computing applications, advanced AI cognitive architectures, and more sophisticated human-robot collaboration systems.",

    "balance": "Balance control in humanoid robots often uses techniques like Zero Moment Point (ZMP) and Capture Point control. These methods ensure the robot maintains stability during locomotion and static poses.",

    "locomotion": "Locomotion for humanoid robots involves complex control strategies for walking, including gait generation, footstep planning, and dynamic balance maintenance.",

    "manipulation": "Manipulation involves controlling the robot's arms and hands to interact with objects. This includes grasp planning, trajectory generation, and force control.",

    "navigation": "Navigation systems enable robots to move through environments safely, using techniques like SLAM, path planning, and obstacle avoidance.",

    "computer vision": "Computer vision enables robots to understand their visual environment through techniques like object detection, recognition, and scene understanding.",

    "machine learning": "Machine learning techniques including deep learning, reinforcement learning, and imitation learning enable robots to adapt and improve their behavior.",

    "human robot interaction": "HRI focuses on how humans and robots can work together effectively, including communication, trust, and collaboration aspects."
  };

  const findRelevantInfo = (userInput) => {
    const input = userInput.toLowerCase();

    // Check for keywords in knowledge base
    for (const [key, value] of Object.entries(knowledgeBase)) {
      if (input.includes(key.toLowerCase())) {
        return value;
      }
    }

    // Check for chapter-related queries
    const chapterMatch = input.match(/chapter (\d+)|ch\.? (\d+)|lesson (\d+)/i);
    if (chapterMatch) {
      const chapterNum = chapterMatch[1] || chapterMatch[2] || chapterMatch[3];
      switch(chapterNum) {
        case '1':
          return "Chapter 1 covers Introduction to Physical AI and Humanoid Robotics. It includes fundamental concepts of embodied AI, humanoid robotics, learning outcomes, and weekly breakdown of the introduction to the field.";
        case '2':
          return "Chapter 2 covers ROS2 Fundamentals for Robotics. This includes ROS2 architecture, nodes, topics, services, actions, and practical implementation for robotic systems.";
        case '3':
          return "Chapter 3 covers Gazebo Simulation Environment. It includes simulation setup, robot models, physics engines, and integration with ROS2 for testing robotic algorithms.";
        case '4':
          return "Chapter 4 covers Unity for Robotics Simulation. This includes Unity robotics packages, perception training, and VR/AR integration for robotics applications.";
        case '5':
          return "Chapter 5 covers NVIDIA Isaac Platform. It includes Isaac Sim, Isaac ROS packages, GPU-accelerated perception, and deployment on Jetson platforms.";
        case '6':
          return "Chapter 6 covers Vision-Language-Action Models. This includes VLA architectures, multimodal learning, and implementation for robotic control.";
        case '7':
          return "Chapter 7 covers Hardware Components and Integration. This includes actuators, sensors, computing platforms, and system integration for humanoid robots.";
        case '8':
          return "Chapter 8 covers Laboratory Exercises Part 1. This includes practical exercises, simulation environments, and hands-on robotics projects.";
        case '9':
          return "Chapter 9 covers Edge AI and Computing Platforms. This includes NVIDIA Jetson, computing optimization, and deployment strategies for robotics.";
        case '10':
          return "Chapter 10 covers Perception Systems. This includes sensor fusion, computer vision, and environmental understanding for robots.";
        case '11':
          return "Chapter 11 covers Control Systems. This includes PID control, balance control, walking patterns, and dynamic control for humanoid robots.";
        case '12':
          return "Chapter 12 covers Learning and Adaptation. This includes reinforcement learning, imitation learning, and adaptive control systems.";
        case '13':
          return "Chapter 13 covers Ethics and Safety in Robotics. This includes safety standards, ethical frameworks, and societal impact of robotics.";
        case '14':
          return "Chapter 14 covers Future Directions and Research. This includes emerging technologies, research challenges, and future trends in humanoid robotics.";
        default:
          return `Chapter ${chapterNum} covers advanced topics in Physical AI and Humanoid Robotics. For specific details, please refer to the course materials.`;
      }
    }

    // Default response if no specific topic found
    return "I'm here to help with the Physical AI and Humanoid Robotics course. The course covers 14 comprehensive chapters including: Introduction, ROS2, Gazebo, Unity, NVIDIA Isaac, VLA models, Hardware, Lab Exercises, Edge AI, Perception, Control Systems, Learning, Ethics, and Future Directions. What specific topic or chapter would you like to know more about?";
  };

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate bot thinking and response
    setTimeout(() => {
      const botResponse = {
        id: Date.now() + 1,
        text: findRelevantInfo(inputValue),
        sender: 'bot'
      };

      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={isPage ? "book-chatbot-page" : "book-chatbot"}>
      {isOpen ? (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>Course Assistant</h3>
            {!isPage && (
              <button className="close-btn" onClick={toggleChat}>×</button>
            )}
          </div>
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.sender}`}
              >
                <div className="message-text">{message.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-text typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>
          <div className="chatbot-input">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about the course content..."
              rows="2"
            />
            <button onClick={handleSendMessage} disabled={isLoading}>
              Send
            </button>
          </div>
          <div className="chatbot-footer">
            <a href="https://github.com/syedsajidhussain/robotics-course" target="_blank" rel="noopener noreferrer" className="github-link">
              📌 GitHub Repository
            </a>
          </div>
        </div>
      ) : (
        !isPage && (
          <button className="chatbot-toggle" onClick={toggleChat}>
            💬 Course Assistant
          </button>
        )
      )}
    </div>
  );
};

export default BookChatbot;