'use client';

import React from 'react';
import { useSidebar } from '@/components/Sidebar/SidebarProvider';

interface MainContentProps {
  children: React.ReactNode;
}

const MainContent: React.FC<MainContentProps> = ({ children }) => {
  const { isCollapsed } = useSidebar();

  return (
    <main 
      className={`flex-1 transition-all duration-300 ${
        isCollapsed ? 'md:ml-16' : 'md:ml-64'
      } ml-0 pb-16 md:pb-0`}
    >
      <div className="pl-4 md:pl-8">
        {children}
      </div>
    </main>
  );
};

export default MainContent;
