// Global variables
let cameraInitialized = false;
let analysisInProgress = false;
let notificationTimeout = null;

// DOM elements
const captureBtn = document.getElementById('capture-btn');
const cameraFeed = document.getElementById('camera-feed');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const progressFill = document.getElementById('progress-fill');
const faceOverlay = document.getElementById('face-overlay');
const loadingScreen = document.getElementById('loading-screen');
const initProgress = document.getElementById('init-progress');

// Status elements
const cameraStatus = document.getElementById('camera-status');
const resultsStatus = document.getElementById('results-status');
const notificationContainer = document.getElementById('notification-container');

// Feature toggles
const ageToggle = document.getElementById('age-toggle');
const emotionToggle = document.getElementById('emotion-toggle');
const genderToggle = document.getElementById('gender-toggle');
const raceToggle = document.getElementById('race-toggle');

// Result elements
const resultsContainer = document.getElementById('results-container');
const emptyState = document.getElementById('empty-state');
const analysisSummary = document.getElementById('analysis-summary');
const processingTime = document.getElementById('processing-time');
const detectionConfidence = document.getElementById('detection-confidence');

// Individual result cards
const ageResult = document.getElementById('age-result');
const emotionResult = document.getElementById('emotion-result');
const genderResult = document.getElementById('gender-result');
const raceResult = document.getElementById('race-result');

// Application state
const appState = {
    cameraReady: false,
    analysisInProgress: false,
    lastAnalysisTime: null,
    selectedFeatures: ['age', 'emotion', 'gender', 'race']
};

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeApplication();
});

async function initializeApplication() {
    try {
        showLoadingScreen(true);
        updateLoadingProgress(20, 'Initializing application...');

        // Initialize event listeners
        initializeEventListeners();

        updateLoadingProgress(40, 'Setting up camera...');
        await initializeCamera();

        updateLoadingProgress(60, 'Loading AI models...');
        await simulateModelLoading();

        updateLoadingProgress(80, 'Finalizing setup...');
        setupInitialState();

        updateLoadingProgress(100, 'Ready!');

        setTimeout(() => {
            showLoadingScreen(false);
            showNotification('Pehechan AI is ready for facial analysis!', 'success');
        }, 500);

    } catch (error) {
        console.error('Initialization error:', error);
        showLoadingScreen(false);
        showNotification('Failed to initialize application. Please refresh the page.', 'error');
    }
}

function initializeEventListeners() {
    // Capture button
    captureBtn.addEventListener('click', handleCaptureAndAnalyze);

    // Feature toggles
    ageToggle.addEventListener('change', updateSelectedFeatures);
    emotionToggle.addEventListener('change', updateSelectedFeatures);
    genderToggle.addEventListener('change', updateSelectedFeatures);
    raceToggle.addEventListener('change', updateSelectedFeatures);

    // Window events
    window.addEventListener('beforeunload', cleanup);
    window.addEventListener('resize', handleResize);
}

async function initializeCamera() {
    try {
        const response = await fetch('/initialize_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.success) {
            appState.cameraReady = true;
            cameraFeed.src = '/camera_feed';
            cameraFeed.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
            updateCameraStatus('active', 'Camera Active');
            captureBtn.disabled = false;

            // Show focus improvement tips
            showNotification('Camera initialized! Please ensure good lighting and hold steady for best results.', 'info', 5000);

            // Show face detection overlay after camera settles
            setTimeout(() => {
                faceOverlay.style.display = 'block';
                // Start focus monitoring
                startFocusMonitoring();
            }, 2000);

        } else {
            throw new Error(data.error || 'Failed to initialize camera');
        }
    } catch (error) {
        console.error('Camera initialization error:', error);
        updateCameraStatus('error', 'Camera Error');
        throw error;
    }
}

function startFocusMonitoring() {
    // Monitor camera feed quality and provide feedback
    let focusCheckInterval = setInterval(() => {
        if (!appState.cameraReady) {
            clearInterval(focusCheckInterval);
            return;
        }

        // Check if camera feed is loaded and visible
        if (cameraFeed.complete && cameraFeed.naturalHeight !== 0) {
            // Update status to indicate camera is ready for capture
            updateCameraStatus('active', 'Ready - Hold Steady');
        }
    }, 3000);

    // Clear interval after 30 seconds
    setTimeout(() => {
        clearInterval(focusCheckInterval);
    }, 30000);
}

async function simulateModelLoading() {
    // Simulate AI model loading time
    return new Promise(resolve => {
        setTimeout(resolve, 1000);
    });
}

function setupInitialState() {
    updateSelectedFeatures();
    resetResults();
    updateResultsStatus('Ready for analysis');
}

function updateSelectedFeatures() {
    const features = [];
    if (ageToggle.checked) features.push('age');
    if (emotionToggle.checked) features.push('emotion');
    if (genderToggle.checked) features.push('gender');
    if (raceToggle.checked) features.push('race');

    appState.selectedFeatures = features;

    // Update capture button state
    if (features.length === 0) {
        captureBtn.disabled = true;
        updateResultsStatus('Select at least one feature');
    } else if (appState.cameraReady) {
        captureBtn.disabled = false;
        updateResultsStatus('Ready for analysis');
    }
}

async function handleCaptureAndAnalyze() {
    if (appState.analysisInProgress || !appState.cameraReady) return;

    if (appState.selectedFeatures.length === 0) {
        showNotification('Please select at least one feature to analyze', 'warning');
        return;
    }

    try {
        appState.analysisInProgress = true;
        showAnalysisProgress(true);
        updateResultsStatus('Analyzing...');
        updateCameraStatus('active', 'Capturing...');

        // Provide user guidance
        showNotification('Hold still for best results...', 'info', 2000);

        const startTime = Date.now();

        // Simulate analysis steps with focus feedback
        await simulateAnalysisSteps();

        const response = await fetch('/capture_and_analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: appState.selectedFeatures
            })
        });

        const data = await response.json();

        if (data.success) {
            const analysisTime = (Date.now() - startTime) / 1000;
            appState.lastAnalysisTime = analysisTime;

            displayResults(data.results, data.metadata);
            showNotification('Analysis completed successfully! High-quality image captured.', 'success');
            updateResultsStatus('Analysis complete');
            updateCameraStatus('active', 'Ready for next capture');
        } else {
            // Handle specific error cases
            if (data.error.includes('blurry') || data.error.includes('focus')) {
                showNotification('Image too blurry. Please ensure camera is in focus and hold steady.', 'warning', 6000);
                updateCameraStatus('active', 'Focus and try again');
            } else if (data.error.includes('lighting') || data.error.includes('dark') || data.error.includes('bright')) {
                showNotification('Lighting issue detected. Please adjust lighting and try again.', 'warning', 6000);
                updateCameraStatus('active', 'Adjust lighting');
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
            updateResultsStatus('Ready for analysis');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
        updateResultsStatus('Analysis failed');
        updateCameraStatus('active', 'Ready to retry');
    } finally {
        appState.analysisInProgress = false;
        showAnalysisProgress(false);
    }
}

async function simulateAnalysisSteps() {
    const steps = [
        { progress: 20, text: 'Detecting face...' },
        { progress: 40, text: 'Extracting features...' },
        { progress: 60, text: 'Running AI analysis...' },
        { progress: 80, text: 'Processing results...' }
    ];

    for (const step of steps) {
        updateAnalysisProgress(step.progress, step.text);
        await new Promise(resolve => setTimeout(resolve, 300));
    }
}

function showAnalysisProgress(show) {
    if (show) {
        loadingOverlay.style.display = 'flex';
        captureBtn.disabled = true;
        faceOverlay.style.display = 'none';
        updateAnalysisProgress(0, 'Starting analysis...');
    } else {
        loadingOverlay.style.display = 'none';
        captureBtn.disabled = false;
        if (appState.cameraReady) {
            faceOverlay.style.display = 'block';
        }
    }
}

function updateAnalysisProgress(percentage, text) {
    if (progressFill) {
        progressFill.style.width = `${percentage}%`;
    }
    if (loadingText) {
        loadingText.textContent = text;
    }
}

function showLoadingScreen(show) {
    if (show) {
        loadingScreen.style.display = 'flex';
        loadingScreen.classList.remove('hidden');
    } else {
        loadingScreen.classList.add('hidden');
        setTimeout(() => {
            loadingScreen.style.display = 'none';
        }, 500);
    }
}

function updateLoadingProgress(percentage, text) {
    if (initProgress) {
        initProgress.style.width = `${percentage}%`;
    }
    // Update loading text if element exists
    const loadingTextEl = document.querySelector('.loading-content p');
    if (loadingTextEl && text) {
        loadingTextEl.textContent = text;
    }
}

function updateCameraStatus(status, text) {
    const statusDot = cameraStatus.querySelector('.status-dot');
    const statusText = cameraStatus.querySelector('.status-text');

    if (statusDot) {
        statusDot.className = 'status-dot';
        if (status === 'active') {
            statusDot.classList.add('active');
        }
    }

    if (statusText) {
        statusText.textContent = text;
    }
}

function updateResultsStatus(text) {
    const statusText = resultsStatus.querySelector('.status-text');
    if (statusText) {
        statusText.textContent = text;
    }
}

function showNotification(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    notificationContainer.appendChild(notification);

    // Auto remove after duration
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
}

function displayResults(results, metadata = {}) {
    // Hide empty state
    emptyState.style.display = 'none';

    // Show analysis summary
    if (metadata.processing_time) {
        processingTime.textContent = metadata.processing_time;
        analysisSummary.style.display = 'block';
    }

    if (metadata.detection_confidence) {
        detectionConfidence.textContent = `${metadata.detection_confidence}%`;
    }

    // Display individual results
    displayFeatureResult('age', results.age);
    displayFeatureResult('emotion', results.emotion);
    displayFeatureResult('gender', results.gender);
    displayFeatureResult('race', results.race);
}

function displayFeatureResult(feature, result) {
    const resultCard = document.getElementById(`${feature}-result`);
    const valueElement = document.getElementById(`${feature}-value`);
    const confidenceElement = document.getElementById(`${feature}-confidence`);
    const confidenceBar = document.getElementById(`${feature}-confidence-bar`);
    const extraElement = document.getElementById(`${feature}-extra`);

    if (!result || !resultCard) return;

    // Show the result card
    resultCard.style.display = 'block';

    // Enhanced handling for age with new structure
    if (feature === 'age') {
        if (valueElement) {
            // Use category_emoji if available, otherwise use display
            const emoji = result.category_emoji || '';
            const category = result.category || result.group || '';
            const ageValue = result.value || 0;

            valueElement.innerHTML = `
                <div class="age-main">${emoji} ${ageValue} years</div>
                <div class="age-category">${category}</div>
            `;
        }

        // Enhanced extra information for age
        if (extraElement) {
            const detailedInfo = result.detailed_info || {};
            const stability = detailedInfo.prediction_stability || 1;
            const lifeStage = detailedInfo.life_stage || '';

            extraElement.innerHTML = `
                <div class="age-details">
                    <div class="age-range">
                        <i class="fas fa-chart-line"></i> Range: ${result.range || 'N/A'}
                    </div>
                    ${lifeStage ? `<div class="age-stage">
                        <i class="fas fa-user-clock"></i> ${lifeStage}
                    </div>` : ''}
                    <div class="age-stability">
                        <i class="fas fa-check-circle"></i>
                        Stability: ${stability > 1 ? 'High' : 'Standard'}
                        ${stability > 1 ? `(${stability} predictions)` : ''}
                    </div>
                </div>
            `;
        }
    } else {
        // Standard handling for other features
        if (valueElement) {
            valueElement.innerHTML = result.display || result.value || 'Unknown';
        }

        // Update extra information for other features
        if (extraElement) {
            if (result.all_scores) {
                // Show top 3 scores for emotions, gender, race
                const scores = Object.entries(result.all_scores)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 3);

                extraElement.innerHTML = `
                    <div class="feature-scores">
                        ${scores.map(([key, value]) =>
                            `<div class="score-item">
                                <span class="score-label">${key.charAt(0).toUpperCase() + key.slice(1)}</span>
                                <span class="score-value">${value}%</span>
                            </div>`
                        ).join('')}
                    </div>
                `;
            } else if (result.range) {
                extraElement.innerHTML = `<div class="feature-range">Range: ${result.range}</div>`;
            }
        }
    }

    // Update confidence with enhanced styling
    if (result.confidence !== undefined && confidenceElement) {
        const confidence = Math.round(result.confidence);
        confidenceElement.textContent = `${confidence}%`;

        // Add confidence level indicator
        let confidenceClass = 'confidence-low';
        if (confidence >= 85) confidenceClass = 'confidence-high';
        else if (confidence >= 70) confidenceClass = 'confidence-medium';

        confidenceElement.className = `confidence-text ${confidenceClass}`;

        // Animate confidence bar with color coding
        if (confidenceBar) {
            setTimeout(() => {
                confidenceBar.style.width = `${confidence}%`;
                confidenceBar.className = `confidence-fill ${confidenceClass}`;
            }, 100);
        }
    }

    // Enhanced animation with stagger effect
    resultCard.style.animation = 'none';
    resultCard.style.opacity = '0';
    resultCard.style.transform = 'translateY(20px)';

    setTimeout(() => {
        resultCard.style.animation = 'slideInUp 0.6s ease-out forwards';
        resultCard.style.opacity = '1';
        resultCard.style.transform = 'translateY(0)';
    }, 50);
}

function resetResults() {
    // Hide all result cards
    ageResult.style.display = 'none';
    emotionResult.style.display = 'none';
    genderResult.style.display = 'none';
    raceResult.style.display = 'none';

    // Show empty state
    emptyState.style.display = 'block';

    // Hide analysis summary
    analysisSummary.style.display = 'none';

    // Reset confidence bars
    const confidenceBars = document.querySelectorAll('.confidence-fill');
    confidenceBars.forEach(bar => {
        bar.style.width = '0%';
    });
}

function handleResize() {
    // Handle responsive adjustments if needed
    if (window.innerWidth <= 768) {
        // Mobile adjustments
        if (notificationContainer) {
            notificationContainer.style.left = '1rem';
            notificationContainer.style.right = '1rem';
        }
    } else {
        // Desktop adjustments
        if (notificationContainer) {
            notificationContainer.style.left = 'auto';
            notificationContainer.style.right = '2rem';
        }
    }
}

async function cleanup() {
    try {
        if (appState.cameraReady) {
            await fetch('/cleanup_camera', { method: 'POST' });
        }
    } catch (error) {
        console.error('Cleanup error:', error);
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Performance monitoring
function logPerformance(label, startTime) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    console.log(`${label}: ${duration.toFixed(2)}ms`);
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'error');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('A network error occurred. Please check your connection.', 'error');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Space bar to capture (when camera is ready and not analyzing)
    if (event.code === 'Space' && appState.cameraReady && !appState.analysisInProgress) {
        event.preventDefault();
        handleCaptureAndAnalyze();
    }

    // Escape to reset results
    if (event.code === 'Escape') {
        resetResults();
        updateResultsStatus('Ready for analysis');
    }
});

// Touch events for mobile
let touchStartY = 0;
let touchEndY = 0;

document.addEventListener('touchstart', function(event) {
    touchStartY = event.changedTouches[0].screenY;
});

document.addEventListener('touchend', function(event) {
    touchEndY = event.changedTouches[0].screenY;
    handleSwipe();
});

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartY - touchEndY;

    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            // Swipe up - could trigger capture
            if (appState.cameraReady && !appState.analysisInProgress) {
                handleCaptureAndAnalyze();
            }
        }
    }
}

// Initialize performance observer if available
if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            if (entry.entryType === 'navigation') {
                console.log(`Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
            }
        }
    });
    observer.observe({ entryTypes: ['navigation'] });
}

// Service worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment when service worker is implemented
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered'))
        //     .catch(error => console.log('SW registration failed'));
    });
}
