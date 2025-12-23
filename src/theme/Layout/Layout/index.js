import React from 'react';
import BookChatbot from '@site/src/components/BookChatbot';

const LayoutWrapper = ({ children }) => {
  return (
    <>
      {children}
      <BookChatbot />
    </>
  );
};

export default LayoutWrapper;