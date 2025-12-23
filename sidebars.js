// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Manual sidebar for the Physical AI and Humanoid Robotics Course
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Course Constitution',
      items: ['Constitution'],
    },
    {
      type: 'category',
      label: 'Introduction to Physical AI and Humanoid Robotics',
      items: ['chapter_1'],
    },
    {
      type: 'category',
      label: 'ROS2 Fundamentals for Robotics',
      items: ['chapter_2'],
    },
    {
      type: 'category',
      label: 'Gazebo Simulation Environment',
      items: ['chapter_3'],
    },
    {
      type: 'category',
      label: 'Unity for Robotics Simulation',
      items: ['chapter_4'],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac Platform',
      items: ['chapter_5'],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Models',
      items: ['chapter_6'],
    },
    {
      type: 'category',
      label: 'Hardware Components and Integration',
      items: ['chapter_7'],
    },
    {
      type: 'category',
      label: 'Laboratory Exercises Part 1',
      items: ['chapter_8'],
    },
    {
      type: 'category',
      label: 'Edge AI and Computing Platforms',
      items: ['chapter_9'],
    },
    {
      type: 'category',
      label: 'Perception Systems',
      items: ['chapter_10'],
    },
    {
      type: 'category',
      label: 'Control Systems',
      items: ['chapter_11'],
    },
    {
      type: 'category',
      label: 'Learning and Adaptation',
      items: ['chapter_12'],
    },
    {
      type: 'category',
      label: 'Ethics and Safety in Robotics',
      items: ['chapter_13'],
    },
    {
      type: 'category',
      label: 'Future Directions and Research',
      items: ['chapter_14'],
    },
  ],
};

export default sidebars;
