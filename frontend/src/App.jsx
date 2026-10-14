import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { HeroBanner } from './components/HeroBanner';
import { ImageUpload } from './components/ImageUpload';
import { PredictionResult } from './components/PredictionResult';
import { StatusBanner } from './components/StatusBanner';
import { predictCropDisease, checkBackendHealth } from './services/api';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Verify backend health on mount
  useEffect(() => {
    async function verifyHealth() {
      const res = await checkBackendHealth();
      setBackendStatus(res.status === 'ok' ? 'ok' : 'offline');
    }
    verifyHealth();

    const interval = setInterval(verifyHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle image file selection
  const handleFileSelect = (file, customDataUrl = null) => {
    setErrorMessage(null);
    setSelectedFile(file);

    if (customDataUrl) {
      setPreviewUrl(customDataUrl);
    } else {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  // Handle Predict button click
  const handlePredict = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload or select a leaf image first.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const data = await predictCropDisease(selectedFile);
      setPredictionResult(data);
    } catch (err) {
      setErrorMessage(
        err.message || 'Could not connect to FastAPI backend server. Ensure backend is running on port 8000.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle reset to upload another image
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="app-layout">
      {/* 1. Top Navigation & Status */}
      <Header backendStatus={backendStatus} />

      {/* 2. Hero Information Banner */}
      <HeroBanner />

      {/* 3. Error / Warning Banner */}
      <StatusBanner
        error={errorMessage}
        onDismiss={() => setErrorMessage(null)}
      />

      {/* 4. Main 2-Column Neo-Brutalist Layout */}
      <main className="main-grid">
        {/* Left Column: Image Upload & Preview */}
        <ImageUpload
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          isLoading={isLoading}
          onFileSelect={handleFileSelect}
          onPredict={handlePredict}
          onReset={handleReset}
        />

        {/* Right Column: Prediction & Treatment Output */}
        <PredictionResult
          result={predictionResult}
          isLoading={isLoading}
        />
      </main>

      {/* 5. Minimal Technical Footer */}
      <footer className="app-footer">
        <div>
          Crop Disease Detection & Treatment Recommendation Mini-Project
        </div>
        <div>
          By: Joaquin, Ronal, Oaish &amp; Hayden
        </div>
      </footer>
    </div>
  );
}
