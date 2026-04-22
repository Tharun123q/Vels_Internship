import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import string
import copy
from collections import deque
from typing import List

# --- 1. Project Configuration (Must Match Training) ---

SEQUENCE_LEN= 30  # Number of frames (time steps) needed for one prediction
MAX_HANDS = 2
FEATURES_PER_HAND = 21 * 2  # 21 landmarks * (x, y) coordinates
FEATURES_PER_FRAME = FEATURES_PER_HAND * MAX_HANDS

# Sign Definitions (ASCII letters + three common dynamic signs)
SIGN_LABELS = list(string.ascii_uppercase) + ['HELLO', 'GOODBYE', 'I_LOVE_YOU']
MODEL_PATH = "model/asl_dynamic_gru_classifier_2hand.keras"


# --- 2. Model and MediaPipe Initialization ---

try:
    # Disable Keras/TensorFlow verbose output during inference
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    model = load_model(MODEL_PATH)
    print(f"✅ Classification model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model. Ensure the model has been trained and the path is correct.")
    print(f"Details: {e}")
    exit()

mp_hands = mp.solutions.hands
# Use MediaPipe solution with configuration parameters
hand_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# --- Visual History Trace Setup ---
HISTORY_LENGTH = 16
point_history = deque(maxlen=HISTORY_LENGTH)
# Pre-fill the deque with zero points
for _ in range(HISTORY_LENGTH):
    point_history.append([0, 0])


# --- 3. Utility Functions ---


def calculate_fps(last_tick_time: int) -> tuple:
    """Calculates FPS and returns (fps, current_tick_count)."""
    current_tick_count = cv.getTickCount()
    
    # Correctly calculates the difference from the last recorded time
    time_diff_ticks = current_tick_count - last_tick_time
    
    # Converts tick difference to milliseconds
    time_diff_ms = (time_diff_ticks / cv.getTickFrequency()) * 1000.0
    
    fps = 1000.0 / time_diff_ms if time_diff_ms > 0 else 0.0
    
    # Return FPS and the current tick count for the next iteration
    return fps, current_tick_count

def calc_bounding_rect(image: np.ndarray, landmarks) -> List[int]:
    """Calculates the minimal bounding box [x_min, y_min, x_max, y_max] for a hand."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # Use np.append is generally slow, but acceptable here as it runs max 2 times per frame
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0) 
        
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    """Converts normalized landmarks to pixel coordinates for drawing."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def draw_hand_label(image, brect, handedness):
    """Draws the 'Left' or 'Right' label above the bounding box."""
    label_text = handedness.classification[0].label[0:]
    
    # Draw background rectangle for the text
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    
    # Draw text
    cv.putText(image, label_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_index_trace(image, point_history):
    """Draws a fading trail of the index fingertip movement."""
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Circle size decreases with age (index in deque) for a fading effect
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                     (152, 251, 152), 2) # Light green color
    return image

def draw_status_overlay(image: np.ndarray, predicted_sign: str, confidence: float, sequence_len: int, fps: float) -> np.ndarray:
    """Draws the main prediction, buffer status, and FPS on the video feed."""
    h, w, c = image.shape
    
    # Draw dark background for status bar
    cv.rectangle(image, (0, 0), (w, 80), (0, 0, 0), -1) 
    
    # FPS Display
    cv.putText(image, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)

    # Sequence Buffer Status
    buffer_status = f"Buffer: {sequence_len}/{SEQUENCE_LEN}"
    cv.putText(image, buffer_status, (w - 200, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)

    # Prediction Text
    # Color changes based on confidence
    color = (0, 255, 0) if confidence >= 0.8 else ((0, 165, 255) if confidence > 0.5 else (255, 255, 255))
    text_status = f"Sign: {predicted_sign}"
    conf_status = f"Confidence: {confidence:.2f}"
    
    cv.putText(image, text_status, (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)
    cv.putText(image, conf_status, (w - 300, 65), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
    
    return image


def get_normalized_features(results) -> np.ndarray:
    """
    Extracts, normalizes (wrist-relative), and zero-pads keypoints for a two-hand system.
    This logic MUST match the training data pre-processing exactly.
    """
    all_features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    
    if not results.multi_hand_landmarks:
        return all_features

    # Order hands: Left='Hand 1', Right='Hand 2' for fixed feature vector order (Index 0: Left, Index 1: Right)
    ordered_landmarks = [None] * MAX_HANDS
    
    for i, classification in enumerate(results.multi_handedness):
        label = classification.classification[0].label
        # Map 'Left' hand to index 0, 'Right' hand to index 1
        hand_idx = 0 if label == 'Left' else 1
        ordered_landmarks[hand_idx] = results.multi_hand_landmarks[i]

    feature_start_index = 0
    
    for hand_landmarks in ordered_landmarks:
        if hand_landmarks:
            landmarks_flat = []
            for landmark in hand_landmarks.landmark:
                # Only use X and Y coordinates (MediaPipe uses Z, but typically discarded for 2D feature vector)
                landmarks_flat.append(landmark.x)
                landmarks_flat.append(landmark.y)
            
            landmarks = np.array(landmarks_flat, dtype=np.float32)
            
            # --- Wrist-Relative Normalization ---
            landmarks_2d = landmarks.reshape(-1, 2)
            # Wrist is landmark 0
            wrist_x, wrist_y = landmarks_2d[0] 
            relative_landmarks = landmarks_2d - [wrist_x, wrist_y]
            relative_landmarks_flat = relative_landmarks.flatten()

            # --- Global Scaling (Normalize to -1.0 to 1.0 based on max deviation from wrist) ---
            max_abs_value = np.max(np.abs(relative_landmarks_flat))
            
            if max_abs_value != 0:
                normalized_landmarks = relative_landmarks_flat / max_abs_value
            else:
                normalized_landmarks = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
            
            # Place the normalized features into the correct position in the final feature vector
            all_features[feature_start_index : feature_start_index + FEATURES_PER_HAND] = normalized_landmarks
            
        feature_start_index += FEATURES_PER_HAND
    
    return all_features


# --- 4. Main Inference Loop ---


def run_inference():
    # Deque to store the sequence of feature vectors
    sequence_buffer = deque(maxlen=SEQUENCE_LEN)
    current_prediction = "..."
    confidence_score = 0.0
    
    # Timing variables for FPS and Cooldown
    last_tick_time = cv.getTickCount()  # For FPS calculation
    last_prediction_time = 0.0          # For Cooldown logic
    PREDICTION_COOLDOWN_SECONDS = 1.5   # Time to wait after a confident prediction

    cap = cv.VideoCapture(0)
    # Set preferred resolution (e.g., 1280x720) for better MediaPipe stability, if supported
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Error: Camera not accessible. Check camera index or permissions.")
        return

    print("\n--- Starting Real-Time 2-Hand Dynamic Recognition ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Update FPS timer and Cooldown timer
        current_tick_count = cv.getTickCount()
        current_time_seconds = current_tick_count / cv.getTickFrequency()
        
        # Calculate FPS and update the last recorded tick time
        fps, last_tick_time = calculate_fps(last_tick_time)
        
        # Mirror the image for a more intuitive user experience
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe (set flags to read-only for slight speedup)
        image_rgb.flags.writeable = False
        results = hand_model.process(image_rgb)
        image_rgb.flags.writeable = True

        # Extract normalized features and update the sequence buffer
        features = get_normalized_features(results)
        sequence_buffer.append(features)

        # --- Visual Feedback and Pointer Trace Update ---
        if results.multi_hand_landmarks:
            # Draw landmarks, bounding box, and handedness for all detected hands
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Bounding box & Handedness Label
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                debug_image = draw_hand_label(debug_image, brect, handedness)
                
                # Draw MediaPipe landmarks
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Use the first detected hand for the Index Finger Trace (Index fingertip is landmark 8)
            first_hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = calc_landmark_list(debug_image, first_hand_landmarks)
            point_history.append(landmark_list[8])
        
        else:
            # If no hands are detected, log a zero point for the trace
            point_history.append([0, 0])

        # Draw the visual trace trail
        debug_image = draw_index_trace(debug_image, point_history)

        # --- Perform Prediction Logic ---
        is_cooldown_expired = (current_time_seconds - last_prediction_time > PREDICTION_COOLDOWN_SECONDS)

        # 1. Only attempt prediction if buffer is full AND cooldown is over
        if len(sequence_buffer) == SEQUENCE_LEN and is_cooldown_expired:
            input_sequence = np.array(list(sequence_buffer))
            # Reshape to (1, SEQUENCE_LENGTH, NUM_FEATURES_PER_FRAME) for the model
            input_data = input_sequence[np.newaxis, :, :] 

            # Perform inference
            predictions = model.predict(input_data, verbose=0)[0]
            
            predicted_index = np.argmax(predictions)
            new_confidence_score = predictions[predicted_index]
            new_prediction_label = SIGN_LABELS[predicted_index]
            
            # If a high-confidence prediction is made:
            if new_confidence_score > 0.7:
                current_prediction = new_prediction_label
                confidence_score = new_confidence_score
                last_prediction_time = current_time_seconds  # Reset cooldown timer
                sequence_buffer.clear()                     # Clear buffer to start capturing the next sign
            else:
                 # If prediction is low confidence, we can optionally clear the buffer
                 # or simply wait for the next prediction attempt after a further delay.
                 pass

        # 2. Aggressive buffer reset (if buffer is full and no confident prediction was made for double the cooldown time)
        elif len(sequence_buffer) == SEQUENCE_LEN and (current_time_seconds - last_prediction_time > PREDICTION_COOLDOWN_SECONDS * 2):
            sequence_buffer.clear()
            current_prediction = "Waiting for Sign..."
            confidence_score = 0.0

        # Display Main Status (FPS, Prediction, Buffer)
        debug_image = draw_status_overlay(debug_image, current_prediction, confidence_score, len(sequence_buffer), fps)
        
        # Display the result window
        cv.imshow('ASL Dynamic Recognizer', debug_image)

        # Exit on 'q' press
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    run_inference()