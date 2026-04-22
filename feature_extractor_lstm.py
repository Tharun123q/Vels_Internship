import mediapipe as mp

import cv2


import numpy as np
import pandas as pd
import os
import time
import string
import sys


# --- Project Configuration ---
SEQUENCE_LEN = 30              # Fixed number of frames per action sequence.
MAX_HANDS = 2                     # Max hands MediaPipe will track.
FEATURES_PER_HAND = 21 * 2    # X, Y coordinates for 21 landmarks (42 features).
FEATURES_PER_FRAME = FEATURES_PER_HAND * MAX_HANDS # 84 total features per frame.
FEATURES_PER_SEQUENCE = SEQUENCE_LEN * FEATURES_PER_FRAME # 2520 features per sequence.

# Sign Definitions
SIGN_LABELS = list(string.ascii_uppercase) + ['HELLO', 'GOODBYE', 'I_LOVE_YOU']
NUM_CLASSES = len(SIGN_LABELS)

# File Paths
DATA_DIR = 'raw_data_2hand'
OUTPUT_FILE = 'asl_sequence_data_2hand.csv'
NUM_SAMPLES_PER_SIGN = 30

# --- Mediapipe Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)

def get_normalized_features(results):
    """
    Extracts, normalizes (wrist-relative), and zero-pads keypoints for a 2-hand system.
    Returns a fixed-size vector (84 features) for every frame.
    """
    all_features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    
    if not results.multi_hand_landmarks:
        return all_features

    # Sort hands by handedness to ensure Left Hand features (0-41) always precede Right Hand features (42-83).
    ordered_landmarks = [None] * MAX_HANDS
    
    for i, classification in enumerate(results.multi_handedness):
        label = classification.classification[0].label # 'Left' or 'Right'
        key_index = 0 if label == 'Left' else 1
        ordered_landmarks[key_index] = results.multi_hand_landmarks[i]

    feature_start_index = 0
    
    for hand_landmarks in ordered_landmarks:
        if hand_landmarks:
            # Flatten the X, Y coordinates
            landmarks_flat = []
            for landmark in hand_landmarks.landmark:
                landmarks_flat.append(landmark.x)
                landmarks_flat.append(landmark.y)
            
            landmarks = np.array(landmarks_flat, dtype=np.float32)
            
            # --- Wrist-Relative Normalization ---
            landmarks_2d = landmarks.reshape(-1, 2)
            wrist_x, wrist_y = landmarks_2d[0]
            relative_landmarks = landmarks_2d - [wrist_x, wrist_y]
            relative_landmarks_flat = relative_landmarks.flatten()

            # Scale to prevent large coordinate changes from camera movement
            max_abs_value = np.max(np.abs(relative_landmarks_flat))
            
            if max_abs_value != 0:
                normalized_landmarks = relative_landmarks_flat / max_abs_value
            else:
                normalized_landmarks = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
            
            # Insert the 42 normalized features into the 84-feature vector
            all_features[feature_start_index : feature_start_index + FEATURES_PER_HAND] = normalized_landmarks
            
        feature_start_index += FEATURES_PER_HAND
    
    return all_features

def collect_data():
    """Main function to manage the sequential data capture process."""
    
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    
    # Define the output CSV header dynamically
    header = ['label'] + [f'feature_{i}' for i in range(FEATURES_PER_SEQUENCE)]
    
    # Load existing data to resume collection
    try:
        df = pd.read_csv(output_path)
        print(f"Loaded existing dataset with {len(df)} recorded sequences.")
        df = df[df['label'] < NUM_CLASSES] # Clean up potential extra classes
    except FileNotFoundError:
        df = pd.DataFrame(columns=header)
        print("Initiating a new dataset from scratch.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be accessed.")
        sys.exit(1)

    print(f"\nCommencing dynamic data capture for {NUM_CLASSES} signs.")

    # Iterate through all defined signs
    for sign_idx, sign_name in enumerate(SIGN_LABELS):
        
        collected_count = len(df[df['label'] == sign_idx])
        required_count = NUM_SAMPLES_PER_SIGN - collected_count
        
        if required_count <= 0:
            print(f" > Sign '{sign_name}' already has {collected_count} samples. Skipping.")
            continue
            
        print(f"\n--- Preparing to record {required_count} samples for SIGN: '{sign_name}' (Class {sign_idx}) ---")
        time.sleep(3) # Short pause to prepare for the first sequence
        
        while collected_count < NUM_SAMPLES_PER_SIGN:
            
            sequence_buffer = [] 
            print(f"  > Starting sequence capture for {sign_name}...")
            
            # Capture the full sequence of 30 frames
            while len(sequence_buffer) < SEQUENCE_LEN:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # --- Drawing and Feature Extraction ---
                current_status = f"Capturing: {len(sequence_buffer)}/{SEQUENCE_LEN}"

                features = get_normalized_features(results)
                sequence_buffer.append(features)
                time.sleep(0.033) # Enforce a steady frame rate (~30 FPS)
                
                # Draw landmarks on the frame for visual feedback
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display current status
                cv2.putText(frame, current_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"SIGN: {sign_name} ({collected_count + 1}/{NUM_SAMPLES_PER_SIGN})", 
                            (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('Dynamic Data Collector', frame)
                
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

            # --- Sequence Capture Complete ---
            if len(sequence_buffer) == SEQUENCE_LEN:
                
                # Flatten the 30x84 sequence into a single 2520-feature vector
                sequence_array = np.array(sequence_buffer)
                flattened_features = sequence_array.flatten().tolist()

                # Create and append the new data row
                row_data = [sign_idx] + flattened_features
                new_row = pd.DataFrame([row_data], columns=header)
                df = pd.concat([df, new_row], ignore_index=True)
                
                collected_count += 1
                print(f"  > Sample {collected_count}/{NUM_SAMPLES_PER_SIGN} successfully recorded and saved.")
                
                # Save the cumulative data to prevent loss
                df.to_csv(output_path, index=False)
                
                time.sleep(1.0) # Pause between sequences for user to reset
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAll required sequences collected. FINAL DATASET SAVED to: {output_path}")

if __name__ == '__main__':
    collect_data()