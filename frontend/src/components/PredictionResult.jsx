import React from 'react';
import {
  LuActivity,
  LuShieldAlert,
  LuShieldCheck,
  LuHeartHandshake,
  LuListChecks,
  LuSparkles,
  LuCircleCheck,
  LuCircleHelp,
  LuFileSearch
} from 'react-icons/lu';

export function PredictionResult({ result, isLoading }) {
  // 1. Loading State
  if (isLoading) {
    return (
      <div className="neo-card">
        <div className="card-header-bar">
          <div className="card-title-group">
            <LuActivity size={20} />
            <h2>Pathology Diagnosis</h2>
          </div>
          <span className="step-indicator">STEP 02</span>
        </div>

        <div className="loading-box">
          <div className="loading-spinner-box">
            <LuFileSearch size={28} style={{ color: 'var(--green-primary)' }} />
          </div>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, textTransform: 'uppercase' }}>
              Processing Leaf Specimen
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Running mock convolutional classifier & extracting symptoms...
            </p>
          </div>
          <div className="progress-stripes" />
        </div>
      </div>
    );
  }

  // 2. Initial / No Result State
  if (!result) {
    return (
      <div className="neo-card">
        <div className="card-header-bar">
          <div className="card-title-group">
            <LuActivity size={20} />
            <h2>Diagnostic Report</h2>
          </div>
          <span className="step-indicator">STEP 02</span>
        </div>

        <div className="placeholder-result">
          <div className="placeholder-icon">
            <LuCircleHelp size={32} style={{ color: 'var(--text-muted)' }} />
          </div>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, textTransform: 'uppercase' }}>
              No Specimen Analyzed Yet
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Upload or pick a sample leaf on the left and click "Predict Disease" to run diagnosis.
            </p>
          </div>

          <div className="guide-steps">
            <div className="guide-item">
              <span className="guide-number">01</span>
              <span>Select or drop leaf photograph</span>
            </div>
            <div className="guide-item">
              <span className="guide-number">02</span>
              <span>Click Predict Disease button</span>
            </div>
            <div className="guide-item">
              <span className="guide-number">03</span>
              <span>Review confidence & treatment plan</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 3. Prediction Result Available
  const { disease, confidence, treatment } = result;
  const isHealthy = treatment?.status?.toLowerCase() === 'healthy';
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="neo-card">
      <div className="card-header-bar">
        <div className="card-title-group">
          <LuActivity size={20} />
          <h2>Diagnostic Report</h2>
        </div>
        <span className="step-indicator">RESULT</span>
      </div>

      {/* Disease Banner */}
      <div className={`result-header-card ${isHealthy ? 'healthy' : 'infected'}`}>
        <div className="disease-label-group">
          <span className="crop-category">{treatment?.crop || 'Crop'} Specimen</span>
          <div className="disease-name" id="detected-disease-title">{disease}</div>
        </div>

        <div className={`status-badge ${isHealthy ? 'healthy' : 'infected'}`}>
          {isHealthy ? <LuShieldCheck size={16} /> : <LuShieldAlert size={16} />}
          <span>{treatment?.status || (isHealthy ? 'Healthy' : 'Infected')}</span>
        </div>
      </div>

      {/* Confidence Score Bar */}
      <div className="confidence-box">
        <div className="confidence-top">
          <span>Classifier Confidence</span>
          <span className="confidence-value" id="confidence-percentage">{confidencePercent}%</span>
        </div>
        <div className="meter-track">
          <div
            className="meter-fill"
            style={{
              width: `${confidencePercent}%`,
              backgroundColor: isHealthy ? '#16a34a' : confidencePercent > 80 ? '#eab308' : '#ef4444'
            }}
          />
        </div>
      </div>

      {/* Primary Recommended Treatment */}
      <div className="treatment-card">
        <div className="section-tag">
          <LuHeartHandshake size={15} />
          <span>Recommended Treatment Action</span>
        </div>
        <div className="treatment-text" id="treatment-recommendation-text">
          {treatment?.recommendation}
        </div>
      </div>

      {/* Agronomic Preventative Measures */}
      {treatment?.preventative_measures && treatment.preventative_measures.length > 0 && (
        <div className="measures-card">
          <div className="section-tag" style={{ color: 'var(--green-dark)' }}>
            <LuListChecks size={15} />
            <span>Preventative & Cultural Practices</span>
          </div>
          <ul className="measures-list">
            {treatment.preventative_measures.map((measure, idx) => (
              <li key={idx}>
                <LuCircleCheck size={16} className="measure-icon" />
                <span>{measure}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Organic / Biological Option */}
      {treatment?.organic_options && (
        <div className="organic-card">
          <LuSparkles size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Organic Alternative: </strong>
            <span>{treatment.organic_options}</span>
          </div>
        </div>
      )}
    </div>
  );
}
