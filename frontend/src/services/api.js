const API_BASE_URL = 'http://localhost:8000/api';

/**
 * Sends image file to backend disease prediction endpoint.
 * @param {File} file - The image file to analyze
 * @returns {Promise<Object>} The prediction and treatment recommendation response
 */
export async function predictCropDisease(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = 'Failed to analyze crop image.';
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Use fallback error message
    }
    throw new Error(errorMessage);
  }

  return await response.json();
}

/**
 * Checks backend health status.
 * @returns {Promise<Object>}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
    });
    if (!response.ok) {
      return { status: 'error' };
    }
    return await response.json();
  } catch {
    return { status: 'offline' };
  }
}
