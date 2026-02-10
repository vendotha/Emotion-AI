from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import base64

app = Flask(__name__)

# 1. SETUP PATHS DYNAMICALLY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# 2. LOAD AI MODEL (Bridging Keras 2 to Keras 3)
json_path = os.path.join(BASE_DIR, 'emotion_model.json')
weights_path = os.path.join(BASE_DIR, 'emotion_model.h5')

with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Custom object scope ensures 'Sequential' is recognized in newer TensorFlow versions
with tf.keras.utils.custom_object_scope({'Sequential': tf.keras.Sequential}):
    emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights(weights_path)

# 3. LOAD FACE DETECTOR
cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(cascade_path)


def process_frame(frame):
    """
    Processes a single frame for emotion detection.
    Returns (processed_frame, face_found_boolean).
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization improves detection in poor lighting
    gray_frame = cv2.equalizeHist(gray_frame)

    # Detect faces with tuned parameters for busy backgrounds
    num_faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(30, 30)
    )

    face_found = len(num_faces) > 0

    for (x, y, w, h) in num_faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)

        # Preprocess ROI for CNN
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        input_data = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

        # Predict
        prediction = emotion_model.predict(input_data, verbose=0)
        maxindex = int(np.argmax(prediction))
        label = emotion_dict[maxindex]

        # Draw Label with Shadow for high-contrast visibility
        cv2.putText(frame, label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, face_found


def generate_frames():
    """
    Generator function that initializes camera ONLY when requested by the browser.
    """
    # Initialize camera inside the generator to prevent page-load blocking
    cap = cv2.VideoCapture(0)

    # Set lower resolution for significant performance boost on MacBook
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Process frame for emotions
            processed_frame, _ = process_frame(frame)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        # Release hardware immediately when the user stops the stream/closes tab
        cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. The 'src' attribute of <img> tag calls this.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # Resize large uploads to prevent AI lag
    if img.shape[1] > 1280:
        img = cv2.resize(img, (1280, int(img.shape[0] * (1280 / img.shape[1]))))

    processed_img, face_found = process_frame(img)

    if not face_found:
        return jsonify({'error': 'AI could not detect a face. Use a clearer front-facing photo.'}), 400

    _, buffer = cv2.imencode('.jpg', processed_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': encoded_img})


if __name__ == '__main__':
    # threaded=True allows multiple routes (UI + Stream) to work simultaneously
    app.run(debug=True, threaded=True)