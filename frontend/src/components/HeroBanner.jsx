import React from 'react';
import { LuScan, LuLayers, LuTarget } from 'react-icons/lu';

export function HeroBanner() {
  return (
    <section className="hero-banner">
      <div className="hero-text">
        <h1>Plant Pathology Diagnostic System</h1>
        <p>
          Upload a high-resolution leaf photograph to detect agricultural pathogens, estimate confidence, and receive prescriptive treatments.
        </p>
      </div>

      <div className="hero-specs">
        <div className="spec-chip">
          <LuTarget style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
          <span>8 Disease Classes</span>
        </div>
        <div className="spec-chip">
          <LuLayers style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
          <span>Tomato / Potato / Pepper</span>
        </div>
        <div className="spec-chip">
          <LuScan style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
          <span>FastAPI + React.js</span>
        </div>
      </div>
    </section>
  );
}
