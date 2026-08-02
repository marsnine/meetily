'use client';

import React from 'react';
import { Home, Calendar, Settings, Mic, Square } from 'lucide-react';
import { useRouter, usePathname } from 'next/navigation';
import { useSidebar } from '@/components/Sidebar/SidebarProvider';
import { useRecordingState } from '@/contexts/RecordingStateContext';

const BottomNavBar: React.FC = () => {
  const router = useRouter();
  const pathname = usePathname();
  const { handleRecordingToggle } = useSidebar();
  const { isRecording } = useRecordingState();

  const isHomePage = pathname === '/';
  const isNotesPage = pathname?.startsWith('/meeting-details') || pathname?.startsWith('/notes');
  const isSettingsPage = pathname === '/settings';

  return (
    <nav className="fixed bottom-0 left-0 right-0 h-16 bg-white/75 backdrop-blur-lg border-t border-gray-200/30 flex items-center justify-around px-4 z-50 md:hidden shadow-[0_-2px_10px_rgba(0,0,0,0.05)]">
      <button
        onClick={() => router.push('/')}
        className={`flex flex-col items-center justify-center w-12 h-12 rounded-xl transition-all duration-200 ${
          isHomePage ? 'text-blue-600 bg-blue-50/50 scale-105' : 'text-gray-500 hover:text-gray-700'
        }`}
        aria-label="Home"
      >
        <Home className="w-5 h-5" />
        <span className="text-[10px] mt-1 font-medium">Home</span>
      </button>

      {/* Floating Record Button in center */}
      <button
        onClick={handleRecordingToggle}
        className={`relative -top-3 flex items-center justify-center w-14 h-14 rounded-full transition-all duration-300 shadow-lg ${
          isRecording 
            ? 'bg-gradient-to-tr from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 animate-pulse text-white' 
            : 'bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white'
        }`}
        aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
      >
        <div className="absolute inset-0 rounded-full bg-inherit filter blur-md opacity-40 scale-110 -z-10"></div>
        {isRecording ? (
          <Square className="w-6 h-6" />
        ) : (
          <Mic className="w-6 h-6" />
        )}
      </button>

      <button
        onClick={() => router.push('/settings')}
        className={`flex flex-col items-center justify-center w-12 h-12 rounded-xl transition-all duration-200 ${
          isSettingsPage ? 'text-blue-600 bg-blue-50/50 scale-105' : 'text-gray-500 hover:text-gray-700'
        }`}
        aria-label="Settings"
      >
        <Settings className="w-5 h-5" />
        <span className="text-[10px] mt-1 font-medium">Settings</span>
      </button>
    </nav>
  );
};

export default BottomNavBar;
