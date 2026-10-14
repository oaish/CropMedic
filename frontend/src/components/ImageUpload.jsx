import React, { useRef, useState } from 'react';
import {
  LuUpload,
  LuImage,
  LuRefreshCw,
  LuCheck,
  LuSparkles,
  LuFileText,
  LuPlay,
  LuTrash2
} from 'react-icons/lu';
import { SAMPLE_PRESETS, dataUrlToFile } from '../utils/sampleImages';

export function ImageUpload({
  selectedFile,
  previewUrl,
  isLoading,
  onFileSelect,
  onPredict,
  onReset,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [activeSampleId, setActiveSampleId] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setActiveSampleId(null);
      onFileSelect(file);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setActiveSampleId(null);
      onFileSelect(file);
    }
  };

  const handleSelectSample = async (sample) => {
    setActiveSampleId(sample.id);
    const file = await dataUrlToFile(sample.dataUrl, sample.filename);
    onFileSelect(file, sample.dataUrl);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 KB';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  };

  return (
    <div className="neo-card">
      <div className="card-header-bar">
        <div className="card-title-group">
          <LuImage size={20} />
          <h2>Leaf Specimen Upload</h2>
        </div>
        <span className="step-indicator">STEP 01</span>
      </div>

      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/png, image/jpeg, image/jpg, image/webp"
        style={{ display: 'none' }}
        id="crop-image-input"
      />

      {/* Upload Dropzone or Preview */}
      {!previewUrl ? (
        <div
          className={`dropzone-container ${isDragging ? 'dragging' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          id="dropzone-area"
        >
          <div className="dropzone-icon">
            <LuUpload size={28} />
          </div>
          <div>
            <div className="dropzone-title">Click to browse or drop leaf image</div>
            <div className="dropzone-subtitle">Supports JPG, PNG, WEBP (Max 10MB)</div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div className="preview-box">
            <img src={previewUrl} alt="Leaf Preview" id="specimen-preview" />
          </div>
          <div className="preview-meta-bar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              <LuFileText size={16} />
              <span>{selectedFile?.name || 'specimen_image.jpg'}</span>
            </div>
            <span>{formatFileSize(selectedFile?.size)}</span>
          </div>
        </div>
      )}

      {/* Quick Test Samples */}
      <div className="samples-section">
        <div className="samples-label">
          <LuSparkles size={14} />
          <span>Or Test With Sample Leaves</span>
        </div>
        <div className="samples-grid">
          {SAMPLE_PRESETS.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className={`sample-chip ${activeSampleId === sample.id ? 'active' : ''}`}
              onClick={() => handleSelectSample(sample)}
            >
              {activeSampleId === sample.id ? (
                <LuCheck size={14} style={{ color: 'var(--green-primary)' }} />
              ) : (
                <LuImage size={14} />
              )}
              <span>{sample.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Buttons: Predict & Reset */}
      <div className="btn-group">
        <button
          type="button"
          className="neo-btn btn-primary"
          onClick={onPredict}
          disabled={!selectedFile || isLoading}
          id="predict-button"
        >
          {isLoading ? (
            <>
              <LuRefreshCw className="animate-spin" size={18} />
              <span>Analyzing Specimen...</span>
            </>
          ) : (
            <>
              <LuPlay size={18} />
              <span>Predict Disease</span>
            </>
          )}
        </button>

        {previewUrl && (
          <button
            type="button"
            className="neo-btn btn-secondary"
            onClick={onReset}
            disabled={isLoading}
            title="Clear current selection"
            id="reset-button"
          >
            <LuTrash2 size={18} />
            <span>Reset</span>
          </button>
        )}
      </div>
    </div>
  );
}
