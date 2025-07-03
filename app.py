from flask import Flask, render_template, request, jsonify, Response
import cv2
import threading
import time
import logging
import atexit
from analysis import analyze_face_features_advanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

camera = None
camera_active = False
camera_lock = threading.Lock()

class AdvancedVideoCamera:
    def __init__(self):
        self.video = None
        self.is_initialized = False

        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_V4L2, "Video4Linux2")
        ]
        for backend, name in backends:
            try:
                logger.info(f"Trying camera backend: {name}")
                self.video = cv2.VideoCapture(0, backend)
                if self.video.isOpened():
                    logger.info(f"Successfully opened camera with {name} backend")
                    break
                self.video.release()
            except Exception as e:
                logger.warning(f"Backend {name} failed: {e}")
                continue

        if self.video is None or not self.video.isOpened():
            raise Exception("Could not open camera with any available backend")

        self._configure_camera()
        self._test_camera()
        self.is_initialized = True
        logger.info("Camera initialized successfully")

    def _configure_camera(self):
        try:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.video.set(cv2.CAP_PROP_FPS, 30)

            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            logger.info("Camera settings configured")
        except Exception as e:
            logger.warning(f"Some camera settings could not be applied: {e}")

    def _test_camera(self):
        time.sleep(1) 

        for attempt in range(3):
            ret, frame = self.video.read()
            if ret and frame is not None:
                logger.info("Camera test successful")
                return
            time.sleep(0.5)

        raise Exception("Camera opened but cannot read frames after multiple attempts")

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        display_image = cv2.resize(image, (640, 480))

        ret, jpeg = cv2.imencode('.jpg', display_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jpeg.tobytes()

    def capture_high_quality_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        return image

def generate_frames():
    global camera, camera_active

    try:
        while True:
            with camera_lock:
                if camera is None or not camera_active:
                    break

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033) 
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        with camera_lock:
            camera_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/initialize_camera', methods=['POST'])
def initialize_camera():
    global camera, camera_active
    try:
        with camera_lock:
            if camera is not None:
                try:
                    camera.video.release()
                except:
                    pass
            camera = AdvancedVideoCamera()
            camera_active = True
        return jsonify({'success': True, 'message': 'Camera initialized'})
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/capture_and_analyze', methods=['POST'])
def capture_and_analyze():
    global camera

    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            })

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            })

        selected_features = data.get('features', [])
        if not selected_features:
            return jsonify({
                'success': False,
                'error': 'Please select at least one feature to analyze'
            })

        valid_features = ['age', 'gender', 'emotion', 'race']
        invalid_features = [f for f in selected_features if f not in valid_features]
        if invalid_features:
            return jsonify({
                'success': False,
                'error': f'Invalid features: {invalid_features}'
            })

        logger.info(f"Starting analysis for features: {selected_features}")

        with camera_lock:
            if camera is None or not camera.is_initialized:
                return jsonify({
                    'success': False,
                    'error': 'Camera not initialized. Please refresh the page.'
                })

            try:
                frame = camera.capture_high_quality_frame()
                if frame is None:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to capture frame. Please ensure camera is working.'
                    })
            except Exception as e:
                logger.error(f"Frame capture failed: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Camera capture failed. Please try again.'
                })

        logger.info("Starting facial analysis...")
        results = analyze_face_features_advanced(frame, selected_features)

        if results['success']:
            logger.info("Analysis completed successfully")
            return jsonify({
                'success': True,
                'results': results['data'],
                'metadata': results.get('metadata', {}),
                'timestamp': time.time()
            })
        else:
            logger.warning(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': results.get('error', 'Analysis failed')
            })

    except Exception as e:
        logger.error(f"Error in capture_and_analyze: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/cleanup_camera', methods=['POST'])
def cleanup_camera():
    global camera, camera_active
    try:
        with camera_lock:
            camera_active = False
            if camera:
                del camera
                camera = None
        return jsonify({'success': True, 'message': 'Camera cleaned up'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'HUFMAN AI - Where faces speak and AI listens (Built through DRDO PXE)'})

import atexit

def cleanup_on_exit():
    global camera, camera_active
    try:
        with camera_lock:
            camera_active = False
            if camera:
                del camera
                camera = None
    except:
        pass

atexit.register(cleanup_on_exit)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
