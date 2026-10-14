import React from 'react';
import { LuSprout, LuCpu, LuShieldCheck, LuCircleAlert } from 'react-icons/lu';

export function Header({ backendStatus }) {
  const isOnline = backendStatus === 'ok';

  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon-box">
          <LuSprout size={28} />
        </div>
        <div>
          <div className="brand-title">
            CropMedic <span style={{ fontSize: '1rem', color: 'var(--green-primary)' }}>AI</span>
          </div>
          <div className="brand-tagline">
            Crop Disease Diagnostics & Agronomic Recommendations
          </div>
        </div>
      </div>

      <div className="header-badges">
        <div className="tag-badge">
          <LuCpu size={14} />
          <span>CNN Architecture</span>
        </div>

        <div className={`tag-badge ${isOnline ? 'status-ok' : 'status-err'}`}>
          {isOnline ? (
            <>
              <LuShieldCheck size={14} />
              <span>Backend API Online</span>
            </>
          ) : (
            <>
              <LuCircleAlert size={14} />
              <span>Backend Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
