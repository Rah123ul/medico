import React from 'react';

export default function MultiplayerLeaderboard({ leaderboard, currentUserId }) {
  if (!leaderboard || leaderboard.length === 0) return null;

  return (
    <div className="leaderboard-hud">
      <div className="leaderboard-title">
        <span>Arena Rankings</span>
        <span>{leaderboard.length} Online</span>
      </div>
      
      <div className="leaderboard-list">
        {leaderboard.map((player, index) => {
          const isSelf = player.id === currentUserId;
          return (
            <div 
              key={player.id} 
              className={`leaderboard-item ${isSelf ? 'leaderboard-item--self' : ''}`}
            >
              <div className="leaderboard-rank">#{index + 1}</div>
              <div className="leaderboard-info">
                <div className="leaderboard-name">
                  {player.name} {isSelf && '(You)'}
                </div>
                <div className="leaderboard-status">
                  {player.stress || 'Calm'}
                </div>
              </div>
              <div className="leaderboard-score">
                {player.score}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
