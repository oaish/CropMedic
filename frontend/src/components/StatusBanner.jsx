import React from 'react';
import { LuTriangleAlert, LuX } from 'react-icons/lu';

export function StatusBanner({ error, onDismiss }) {
  if (!error) return null;

  return (
    <div className="error-banner" role="alert">
      <div className="error-left">
        <LuTriangleAlert size={22} style={{ flexShrink: 0 }} />
        <span>{error}</span>
      </div>
      {onDismiss && (
        <button
          type="button"
          className="error-btn"
          onClick={onDismiss}
          title="Dismiss notification"
        >
          <LuX size={16} />
        </button>
      )}
    </div>
  );
}
