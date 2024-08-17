

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import dlib
# import numpy as np
# import base64
# import os
# import concurrent.futures

# app = Flask(__name__)
# CORS(app)

# # Load dlib's face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# base_dir = os.path.dirname(os.path.abspath(__file__))
# predictor = dlib.shape_predictor(os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat"))

# def overlay_glasses(frame):
#     glasses_path = os.path.join(base_dir, 'glasses3.png')
#     glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

#     if glasses is None:
#         raise FileNotFoundError(f"Could not read the image file at {glasses_path}")

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)

#         # Get the positions of the left and right eye
#         left_eye = (landmarks.part(36).x, landmarks.part(36).y)
#         right_eye = (landmarks.part(45).x, landmarks.part(45).y)

#         # Calculate the center point between the eyes
#         eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

#         # Calculate the angle of rotation for the glasses
#         dY = right_eye[1] - left_eye[1]
#         dX = right_eye[0] - left_eye[0]
#         angle = np.degrees(np.arctan2(dY, dX))

#         # Calculate the size of the glasses to fit the width between the eyes
#         glasses_width = int(1.6 * np.sqrt((dX ** 2) + (dY ** 2)))
#         glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

#         # Resize the glasses
#         glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

#         # Rotate the glasses
#         M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), angle, 1)
#         glasses_rotated = cv2.warpAffine(glasses_resized, M, (glasses_width, glasses_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

#         # Determine the position to place the glasses
#         y1 = int(eyes_center[1] - glasses_height / 2)
#         y2 = y1 + glasses_height
#         x1 = int(eyes_center[0] - glasses_width / 2)
#         x2 = x1 + glasses_width

#         # Ensure the coordinates are within the image dimensions
#         y1, y2 = max(0, y1), min(frame.shape[0], y2)
#         x1, x2 = max(0, x1), min(frame.shape[1], x2)
        
#         # Extract the region of interest
#         glasses_rotated = glasses_rotated[0:y2-y1, 0:x2-x1]

#         # Extract the alpha mask of the glasses and the inverse mask
#         alpha_s = glasses_rotated[:, :, 3] / 255.0  # Alpha channel
#         alpha_l = 1.0 - alpha_s

#         # Overlay the glasses on the frame
#         for c in range(3):
#             frame[y1:y2, x1:x2, c] = (alpha_s * glasses_rotated[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

#     return frame

# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     file = request.files['frame']
#     npimg = np.frombuffer(file.read(), np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     # Use a thread pool to process the frame asynchronously
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(overlay_glasses, frame)
#         processed_frame = future.result()

#     _, buffer = cv2.imencode('.jpg', processed_frame)
#     encoded_image = base64.b64encode(buffer).decode('utf-8')

#     return jsonify({'image': encoded_image})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, threaded=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import base64
import os
import concurrent.futures

app = Flask(__name__)
CORS(app)

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
base_dir = os.path.dirname(os.path.abspath(__file__))
predictor = dlib.shape_predictor(os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat"))

def overlay_glasses(frame):
    glasses_path = os.path.join(base_dir, 'glasses2.png')
    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

    if glasses is None:
        raise FileNotFoundError(f"Could not read the image file at {glasses_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get the positions of the left and right eye
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Calculate the center point between the eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Calculate the angle of rotation for the glasses
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the size of the glasses to fit the width between the eyes
        glasses_width = int(1.6 * np.sqrt((dX ** 2) + (dY ** 2)))
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

        # Resize the glasses
        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

        # Rotate the glasses
        M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), angle, 1)
        glasses_rotated = cv2.warpAffine(glasses_resized, M, (glasses_width, glasses_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Determine the position to place the glasses
        y1 = int(eyes_center[1] - glasses_height / 2)
        y2 = y1 + glasses_height
        x1 = int(eyes_center[0] - glasses_width / 2)
        x2 = x1 + glasses_width

        # Ensure the coordinates are within the image dimensions
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        
        # Extract the region of interest
        glasses_rotated = glasses_rotated[0:y2-y1, 0:x2-x1]

        # Extract the alpha mask of the glasses and the inverse mask
        alpha_s = glasses_rotated[:, :, 3] / 255.0  # Alpha channel
        alpha_l = 1.0 - alpha_s

        # Overlay the glasses on the frame
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * glasses_rotated[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

    return frame

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Use a thread pool to process the frame asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(overlay_glasses, frame)
        processed_frame = future.result()

    _, buffer = cv2.imencode('.jpg', processed_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
