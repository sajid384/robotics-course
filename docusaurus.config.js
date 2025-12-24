// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robotics Course',
  tagline: 'Comprehensive Guide to Embodied Artificial Intelligence',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // ✅ Vercel production URL
  url: 'https://robotics-course.vercel.app',

  // ✅ Root base URL for Vercel (IMPORTANT)
  baseUrl: '/',

  trailingSlash: false,

  // (Safe to keep – mainly for GitHub Pages)
  organizationName: 'syedsajidhussain',
  projectName: 'robotics-course',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'warn',
    },
  },

  staticDirectories: ['static', 'public'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Robotics Course',
      logo: {
        alt: 'Physical AI and Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Course Content',
        },
        {
          href: 'https://github.com/syedsajidhussain/robotics-course',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course Content',
          items: [
            { label: 'Course Constitution', to: '/docs/Constitution' },
            { label: 'Introduction', to: '/docs/chapter_1' },
            { label: 'Future Directions', to: '/docs/chapter_14' },
          ],
        },
        {
          title: 'Resources',
          items: [
            { label: 'ROS Documentation', href: 'https://docs.ros.org/' },
            { label: 'Gazebo Simulation', href: 'https://gazebosim.org/' },
            { label: 'NVIDIA Isaac', href: 'https://developer.nvidia.com/isaac' },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/syedsajidhussain/robotics-course',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI and Humanoid Robotics Course.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
