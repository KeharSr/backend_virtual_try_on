

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from rembg import remove
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(base_dir, 'uploaded_images')
PROCESSED_FOLDER = os.path.join(base_dir, 'processed_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Cache for preprocessed T-shirt images
tshirt_cache = {}


def remove_background(image):
    """Remove the background from the T-shirt image."""
    return remove(image)


def overlay_tshirt(frame, tshirt, landmarks):
    """Overlay a T-shirt on the frame based on body landmarks using masking."""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

    # Calculate T-shirt dimensions based on shoulder width and torso height
    tshirt_width = int(1.5 * abs(right_shoulder.x - left_shoulder.x) * frame.shape[1])
    tshirt_height = int(abs(left_hip.y - left_shoulder.y) * frame.shape[0])

    # Resize the T-shirt image
    tshirt_resized = cv2.resize(tshirt, (tshirt_width, tshirt_height), interpolation=cv2.INTER_AREA)

    if tshirt_resized.shape[2] == 3:
        tshirt_resized = cv2.cvtColor(tshirt_resized, cv2.COLOR_RGB2RGBA)
        tshirt_resized[:, :, 3] = 255

    # Calculate overlay position (align T-shirt with shoulders and offset upward)
    inch_offset = int(frame.shape[0] * 0.05)  # Adjust upward offset for neck alignment
    x1 = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1] - tshirt_width / 2)
    y1 = max(0, int(left_shoulder.y * frame.shape[0]) - inch_offset)
    x2, y2 = x1 + tshirt_width, y1 + tshirt_height

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    tshirt_resized = tshirt_resized[0:y2 - y1, 0:x2 - x1]

    if tshirt_resized.shape[0] > 0 and tshirt_resized.shape[1] > 0:
        tshirt_alpha = tshirt_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (tshirt_alpha * tshirt_resized[:, :, c] +
                                       (1 - tshirt_alpha) * frame[y1:y2, x1:x2, c])

    return frame


def download_and_process_tshirt(tshirt_image_url):
    """Download the T-shirt image and process it by removing the background."""
    if tshirt_image_url in tshirt_cache:
        return tshirt_cache[tshirt_image_url], None

    try:
        # Download T-shirt image
        response = requests.get(tshirt_image_url)
        if response.status_code != 200:
            return None, "Failed to download T-shirt image"

        # Save the T-shirt image in the uploaded_images folder
        tshirt_image = np.frombuffer(response.content, np.uint8)
        tshirt_image = cv2.imdecode(tshirt_image, cv2.IMREAD_UNCHANGED)
        tshirt_image_path = os.path.join(UPLOAD_FOLDER, "tshirt_image.jpg")
        cv2.imwrite(tshirt_image_path, tshirt_image)
        print(f"T-shirt image saved at: {tshirt_image_path}")

        # Apply background removal
        _, encoded_tshirt_image = cv2.imencode('.png', tshirt_image)  # Encode image for `rembg`
        tshirt_no_bg = remove_background(encoded_tshirt_image.tobytes())  # Remove background
        np_tshirt_no_bg = np.frombuffer(tshirt_no_bg, np.uint8)  # Convert to numpy array
        tshirt_no_bg = cv2.imdecode(np_tshirt_no_bg, cv2.IMREAD_UNCHANGED)  # Decode back to image

        # Save processed T-shirt image in the processed_images folder
        processed_tshirt_image_path = os.path.join(PROCESSED_FOLDER, "processed_tshirt.png")
        cv2.imwrite(processed_tshirt_image_path, tshirt_no_bg)
        print(f"Processed T-shirt image saved at: {processed_tshirt_image_path}")

        # Cache the processed T-shirt image
        tshirt_cache[tshirt_image_url] = tshirt_no_bg
        return tshirt_no_bg, None
    except Exception as e:
        return None, f"Error processing T-shirt image: {str(e)}"


@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the uploaded T-shirt image URL
    tshirt_image_url = request.form.get('tshirt_image')
    if not tshirt_image_url:
        return jsonify({"error": "T-shirt image URL is missing"}), 400

    # Process the T-shirt image (download + background removal)
    tshirt_future = executor.submit(download_and_process_tshirt, tshirt_image_url)
    tshirt_no_bg, error = tshirt_future.result()  # Wait for the T-shirt processing to finish

    if error:
        return jsonify({"error": error}), 400

    # Get the uploaded frame
    file = request.files['frame']
    frame = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Pose detection and T-shirt overlay
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame = overlay_tshirt(frame, tshirt_no_bg, landmarks)

    # Encode the processed frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_image})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
