import { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';

const SOCKET_SERVER_URL = 'http://localhost:3000';

export default function useMultiplayer() {
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [room, setRoom] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);

  useEffect(() => {
    const socket = io(SOCKET_SERVER_URL, {
      autoConnect: false,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
      console.log('[Multiplayer] Connected to Arena');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('[Multiplayer] Disconnected');
    });

    socket.on('room:update', (roomData) => {
      setRoom(roomData);
    });

    socket.on('score:sync', ({ leaderboard }) => {
      setLeaderboard(leaderboard);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const connect = () => {
    if (socketRef.current && !socketRef.current.connected) {
      socketRef.current.connect();
    }
  };

  const joinRoom = (joinCode, userName) => {
    if (socketRef.current) {
      socketRef.current.emit('room:join', { joinCode, userName });
    }
  };

  const pushMetrics = (joinCode, signals) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('trace:push', { 
        joinCode, 
        ...signals 
      });
    }
  };

  return {
    isConnected,
    room,
    leaderboard,
    connect,
    joinRoom,
    pushMetrics
  };
}
