import React from 'react';
import Layout from '@theme/Layout';
import BookChatbot from '@site/src/components/BookChatbot';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function ChatbotPage() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`Chatbot - ${siteConfig.title}`}
      description="Physical AI and Humanoid Robotics Course Assistant">
      <main>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '80vh',
          padding: '2rem'
        }}>
          <h1 style={{ textAlign: 'center', marginBottom: '2rem' }}>Course Assistant</h1>
          <p style={{ textAlign: 'center', marginBottom: '2rem', maxWidth: '600px' }}>
            Welcome to the Physical AI and Humanoid Robotics Course Assistant!
            Ask me anything about the course content, and I'll help you find relevant information from all 14 chapters.
          </p>
          <div style={{ width: '100%', maxWidth: '600px', height: '600px' }}>
            <BookChatbot isPage={true} />
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default ChatbotPage;