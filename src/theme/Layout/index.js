import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import BookChatbot from '@site/src/components/BookChatbot';

export default function LayoutWrapper(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
      </OriginalLayout>
      <BookChatbot />
    </>
  );
}