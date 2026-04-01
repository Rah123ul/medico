import { useState } from 'react';
import OnboardingScreen from './components/Onboarding/OnboardingScreen';
import DashboardScreen from './components/Dashboard/DashboardScreen';
import './index.css';

import useMultiplayer from './hooks/useMultiplayer';

function App() {
  const [currentScreen, setCurrentScreen] = useState('onboarding');
  const multiplayer = useMultiplayer();

  return (
    <>
      <div className="ambient-bg" aria-hidden="true">
        <div className="ambient-orb orb-1"></div>
        <div className="ambient-orb orb-2"></div>
        <div className="ambient-orb orb-3"></div>
      </div>

      {currentScreen === 'onboarding' && (
        <OnboardingScreen 
          onStartSession={() => setCurrentScreen('dashboard')} 
          multiplayer={multiplayer}
        />
      )}
      
      {currentScreen === 'dashboard' && (
        <DashboardScreen 
          onBack={() => setCurrentScreen('onboarding')} 
          multiplayer={multiplayer}
        />
      )}
    </>
  );
}

export default App;
