import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import pyautogui
import keyboard 
from math import atan2, degrees, sqrt
import re
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from pykalman import KalmanFilter  # Ensure pykalman is installed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde


#eyegaze--------------------------------------------------------------------------------------------------------

# Function to enhance the eye images using histogram equalization
def enhance_eye_image(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    enhanced_eye = cv2.equalizeHist(gray_eye)
    return enhanced_eye

# Function to extract eye regions from facial landmarks
def extract_eye_regions(frame, landmarks):
    h, w, _ = frame.shape
    left_eye_indices = [33, 133, 158, 153, 144, 145]  # Indices for left eye landmarks
    right_eye_indices = [362, 263, 387, 382, 373, 374]  # Indices for right eye landmarks

    left_eye_coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices], dtype=np.int32)
    right_eye_coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices], dtype=np.int32)

    # Get bounding box for left eye
    left_x, left_y, left_w, left_h = cv2.boundingRect(left_eye_coords)
    left_eye_image = frame[left_y - 5:left_y + left_h + 5, left_x - 5:left_x + left_w + 5]

    # Get bounding box for right eye
    right_x, right_y, right_w, right_h = cv2.boundingRect(right_eye_coords)
    right_eye_image = frame[right_y - 5:right_y + right_h + 5, right_x - 5:right_x + right_w + 5]

    return left_eye_image, right_eye_image

# Function to save and resize images
def save_resized_image(image, path):
    resized_image = cv2.resize(image, (244, 244))  # Resize to 244x244 pixels
    cv2.imwrite(path, resized_image)

# Create directories to save images if they don't exist
# face_image_dir = 'face_images'
# eye_image_dir = 'eye_images'

def eyeGazing():
    # Constants (update with the values from Script 1)
    real_world_eye_distance = 8.5  # Distance between eyes in cm
    focal_length = 282.40  # Replace with your calculated focal length from Script 1

    # os.makedirs(face_image_dir, exist_ok=True)
    # os.makedirs(eye_image_dir, exist_ok=True)
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Set up Face Mesh with 478 landmarks
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # CSV file to log the data
    csv_file_path = 'data/trail.csv'
    data_log = []  # List to hold the data for logging

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Frame counter

    # Step 5: Additional coordinates for detecting head roll
    head_roll_landmarks = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0]

    # Step 6: Pairs for detecting yaw movement
    yaw_pairs = {
        '10 to 109': (10,109),
        '10 to 338': (10,338), 
        '151 to 103': (151,103),
        '151 to 332': (151,332),
        '9 to 21': (9, 21),
        '9 to 251': (9,251),
        '8 to 162': (8,162),
        '8 to 389':(8,389),
        '6 to 127': (6,127),
        '6 to 356': (6,356),
        '195 to 454': (195,454),
        '195 to 234': (195,234),
        '5 to 93': (5, 93),
        '5 to 366': (5,366)
    }

    # Step 7: Pairs for detecting pitch movement
    pitch_reference = 10
    pitch_landmarks = [116, 118, 47, 195, 277, 349, 345]


    while cap.isOpened():
        if not keyboard.is_pressed('A'):
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty frame from camera.")
                    continue

                # Flip and convert the color of the frame (since OpenCV uses BGR)
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width = frame.shape[:2]

                # Process the frame with Face Mesh
                results = face_mesh.process(rgb_frame)

                # Draw only the eye landmarks if detected
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        # # Extract eye regions
                        # left_eye_image, right_eye_image = extract_eye_regions(frame, face_landmarks.landmark)

                        # # Enhance the eye images
                        # enhanced_left_eye = enhance_eye_image(left_eye_image)
                        # enhanced_right_eye = enhance_eye_image(right_eye_image)
                        # Define left and right eye bounding boxes and pupil centers

                        cursor_x, cursor_y = pyautogui.position()
                        print(f"Cursor Position: X = {cursor_x}, Y = {cursor_y}")
                    
                        left_eye_center = 473
                        right_eye_center = 468

                        # Define the additional landmarks for calculating relative positions
                        left_eye_landmarks = [463, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 53, 21, 162, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0]
                        right_eye_landmarks = [33, 7, 163, 144, 153, 154, 155, 145, 133, 173, 157, 158, 159, 160, 161, 246, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 53, 21, 162, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0]

                        # Get coordinates for left and right pupil centers
                        left_pupil_center = face_landmarks.landmark[left_eye_center]
                        right_pupil_center = face_landmarks.landmark[right_eye_center]

                        # Draw the pupil centers
                        left_eye_x,left_eye_y = (int(left_pupil_center.x * frame_width), int(left_pupil_center.y * frame_height))
                        right_eye_x,right_eye_y = (int(right_pupil_center.x * frame_width), int(right_pupil_center.y * frame_height))

                        # Calculate pixel distance between left and right eye
                        pixel_eye_distance = ((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) ** 0.5
                        
                        # Calculate distance from camera
                        distance_btn_screen_and_face = (real_world_eye_distance * focal_length) / pixel_eye_distance
                        print(f'distance between screen and face is:{distance_btn_screen_and_face}')

                        left_pupil_center_2d=()
                        right_pupil_center_2d=()
                        left_pupil_center_2d+=(left_eye_x,)
                        left_pupil_center_2d+=(left_eye_y,)
                        right_pupil_center_2d+=(right_eye_x,)
                        right_pupil_center_2d+=(right_eye_y,)
                        

                        cv2.circle(frame, left_pupil_center_2d, 2, (0, 255, 255), -1)
                        cv2.circle(frame, right_pupil_center_2d, 2, (0, 255, 255), -1)

                        # Calculate EAR for left eye
                        left_y1 = face_landmarks.landmark[386].y  # Landmark above the eye
                        left_y2 = face_landmarks.landmark[374].y  # Landmark below the eye
                        left_x1 = face_landmarks.landmark[463].x  # Left outer eye landmark
                        left_x2 = face_landmarks.landmark[263].x  # Right outer eye landmark
                        left_ear = (left_y1 - left_y2) / (left_x2 - left_x1) if (left_y1 - left_y2) != 0 else 0

                        # Calculate EAR for right eye
                        right_y1 = face_landmarks.landmark[159].y  # Landmark above the eye
                        right_y2 = face_landmarks.landmark[145].y  # Landmark below the eye
                        right_x1 = face_landmarks.landmark[33].x  # Left outer eye landmark
                        right_x2 = face_landmarks.landmark[133].x  # Right outer eye landmark
                        right_ear = (right_y1 - right_y2) / (right_x2 - right_x1) if (right_y2 - right_y1) != 0 else 0

                        # Calculate and draw lines for left eye relative positions and distances
                        left_distances = []
                        print("Left eye relative positions and distances:")
                        for landmark in left_eye_landmarks:
                            landmark_data = face_landmarks.landmark[landmark]
                            landmark_x = landmark_data.x
                            landmark_y = landmark_data.y
                            landmark_z = landmark_data.z

                            # Calculate the distance in 3D space
                            distance = np.sqrt((landmark_x - left_pupil_center.x) ** 2 +
                                            (landmark_y - left_pupil_center.y) ** 2 +
                                            (landmark_z - left_pupil_center.z) ** 2)

                            # Convert landmark coordinates to pixel space
                            landmark_pixel = (int(landmark_x * frame_width), int(landmark_y * frame_height))
                            print(f"Landmark {landmark}: Distance from left pupil center = {distance:.2f}")
                            left_distances.append(distance)

                            # Draw line from left pupil center to this landmark
                            cv2.line(frame, left_pupil_center_2d, landmark_pixel, (0, 0, 0), 2)

                        # Calculate and draw lines for right eye relative positions and distances
                        right_distances = []
                        print("Right eye relative positions and distances:")
                        for landmark in right_eye_landmarks:
                            landmark_data = face_landmarks.landmark[landmark]
                            landmark_x = landmark_data.x
                            landmark_y = landmark_data.y
                            landmark_z = landmark_data.z

                            # Calculate the distance in 3D space
                            distance = np.sqrt((landmark_x - right_pupil_center.x) ** 2 +
                                            (landmark_y - right_pupil_center.y) ** 2 +
                                            (landmark_z - right_pupil_center.z) ** 2)

                            # Convert landmark coordinates to pixel space
                            landmark_pixel = (int(landmark_x * frame_width), int(landmark_y * frame_height))
                            print(f"Landmark {landmark}: Distance from right pupil center = {distance:.2f}")
                            right_distances.append(distance)

                            # Draw line from right pupil center to this landmark
                            cv2.line(frame, right_pupil_center_2d, landmark_pixel, (0, 0, 0), 2)

                        # Prepare data for logging
                        left_distances_str = ', '.join(f'{d:.2f}' for d in left_distances)
                        right_distances_str = ', '.join(f'{d:.2f}' for d in right_distances)

                        # Collect head roll data and draw lines
                        head_roll_data_x = []
                        head_roll_data_y = []
                        head_roll_data_z = []
                        head_roll_pixel_coords = []  # To hold pixel coordinates of head roll landmarks
                        for landmark in head_roll_landmarks:
                            landmark_data = face_landmarks.landmark[landmark]
                            head_roll_data_x.append(landmark_data.x)
                            head_roll_data_y.append(landmark_data.y)
                            head_roll_data_z.append(landmark_data.z)
                            if landmark==10:
                                start_coords=[]
                                start_coords.append(landmark_data.x)
                                start_coords.append(landmark_data.y)
                                start_coords.append(landmark_data.z)
                            if landmark==0:
                                end_coords=[]
                                end_coords.append(landmark_data.x)
                                end_coords.append(landmark_data.y)
                                end_coords.append(landmark_data.z)
                            
                                
                            

                                # Create a vector from start to end point
                                line_vector = np.array(end_coords) - np.array(start_coords)

                                # Define the horizontal axis in 3D (assuming x-axis: [1, 0, 0])
                                horizontal_axis = np.array([1, 0, 0])

                                # Calculate the angle between the line vector and the horizontal axis
                                dot_product = np.dot(line_vector, horizontal_axis)
                                line_magnitude = np.linalg.norm(line_vector)
                                horizontal_magnitude = np.linalg.norm(horizontal_axis)
                                cos_theta = dot_product / (line_magnitude * horizontal_magnitude)

                                # Angle in degrees
                                angle = degrees(atan2(sqrt(1 - cos_theta**2), cos_theta))
                                # List to store calculated angles


                            # Convert landmark coordinates to pixel space for drawing
                            landmark_pixel = (int(landmark_data.x * frame_width), int(landmark_data.y * frame_height))
                            head_roll_pixel_coords.append(landmark_pixel)

                            # Draw the landmarks on the frame
                            cv2.circle(frame, landmark_pixel, 2,(255, 0, 0),1)

                        # Draw lines connecting head roll landmarks
                        for i in range(len(head_roll_pixel_coords) - 1):
                            cv2.line(frame, head_roll_pixel_coords[i], head_roll_pixel_coords[i + 1], (255, 0, 0), 2)

                        # Calculate and log yaw movements
                        yaw_data = []
                        for label, (ref,landmark_index) in yaw_pairs.items():

                            landmark_data = face_landmarks.landmark[landmark_index]
                            ref_data=face_landmarks.landmark[ref]
                    
                            yaw_diff1 = round(np.sqrt((landmark_data.x - ref_data.x) ** 2 + (landmark_data.y - ref_data.y) ** 2),4)


                            yaw_data.append(yaw_diff1)
                            # Convert to pixel space
                            ref_pixel = (int(ref_data.x * frame_width), int(ref_data.y * frame_height))
                            target_pixel = (int(landmark_data.x * frame_width), int(landmark_data.y * frame_height))

                            # Draw the purple line
                            cv2.line(frame, ref_pixel, target_pixel, (255, 0, 255), 1)  # Purple color with thickness of 2

                            print(f"{label}: Yaw Difference = {yaw_diff1:.2f}")
                        
                                    # Calculate pitch based on specified landmarks
                        pitch_values = []
                        for landmark in pitch_landmarks:
                            landmark_data = face_landmarks.landmark[landmark]
                            pitch_value = landmark_data.y - face_landmarks.landmark[pitch_reference].y
                            pitch_values.append(pitch_value)

                            # Convert to pixel space
                            ref_pitch = (int(face_landmarks.landmark[pitch_reference].x * frame_width), int(face_landmarks.landmark[pitch_reference].y * frame_height))
                            target_pitch = (int(face_landmarks.landmark[landmark].x * frame_width), int(face_landmarks.landmark[landmark].y * frame_height))

                            # Draw the purple line
                            cv2.line(frame, ref_pitch, target_pitch, (0, 100, 0), 2)  # Purple color with thickness of 2

                            # face_image_path = os.path.join(face_image_dir, f"face_{frame_count}.png")
                            # left_eye_image_path = os.path.join(eye_image_dir, f"left_eye_{frame_count}.png")
                            # right_eye_image_path = os.path.join(eye_image_dir, f"right_eye_{frame_count}.png")
                            # # Save the full face image and eye images
                            # save_resized_image(frame, face_image_path)
                            # save_resized_image(enhanced_left_eye, left_eye_image_path)
                            # save_resized_image(enhanced_right_eye, right_eye_image_path)
                        



                    

                        # Append data to the log
                        data_log.append({
                            'Frame':frame_count,
                            'cursor_x':cursor_x,
                            'cursor_y': cursor_y,
                            'Distance':distance_btn_screen_and_face,
                            # 'face_image_path: face_image_path,
                            # 'left_eye_image_path': left_eye_image_path,
                            # 'right_eye_image_path': right_eye_image_path,
                            'LeftEAR': left_ear,
                            'RightEAR': right_ear,
                            **{f'Landmark{landmark}vsLeft Pupil': d for landmark, d in zip(left_eye_landmarks, left_distances)},
                            **{f'Landmark{landmark}vsRight Pupil': d for landmark, d in zip(right_eye_landmarks, right_distances)},
                            **{f'yaw_diff{label}':yaw_diff1 for label,yaw_diff1 in zip(yaw_pairs.keys(), yaw_data)},
                            **{f'pitch_diff{pitch_reference}vs{pitch_land}':pitch_data for pitch_land,pitch_data in zip(pitch_landmarks,pitch_values)},
                            'roll_angle':angle, 
                            **{f'head_roll{landmark}_{axis}': data[i] for i, landmark in enumerate(head_roll_landmarks) for axis, data in zip(['x', 'y', 'z'], [head_roll_data_x, head_roll_data_y, head_roll_data_z])}
                
                        })

                        frame_count += 1

                # Show the resulting frame
                cv2.imshow('Face Mesh', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Convert the data to a DataFrame and save to CSV
    df = pd.DataFrame(data_log)
    df.to_csv(csv_file_path, index=False)

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


    # Load the scaler, PCA model, and trained model for cursor_x
    scaler_x = joblib.load('model/scaler_x_updates_70000.pkl')
    pca_x = joblib.load('model/pca_model_x_updated_70000.pkl')
    model_x = load_model("model/gaze_detection_model_x1_updated_70000.h5", custom_objects={"mse": MeanSquaredError()})

    # Load the scaler, PCA model, and trained model for cursor_y
    scaler_y = joblib.load('model/scaler_y_updates_70000.pkl')
    pca_y = joblib.load('model/pca_model_y_updated_70000.pkl')
    model_y = load_model("model/gaze_detection_model_y1_updated_70000.h5", custom_objects={"mse": MeanSquaredError()})

    # Load new input data for testing
    input_file_path = 'data/trail.csv'  # Replace with your input data file


    data = pd.read_csv(input_file_path)
    columns_to_drop=['cursor_x','cursor_y']

    input_data=data.drop(columns=columns_to_drop)

    # Remove non-numeric and unnecessary columns (if needed)
    input_data = input_data.select_dtypes(include=[np.number])

    # Handle missing values if necessary
    input_data.fillna(input_data.mean(), inplace=True)

    # Scale and transform the data for cursor_x prediction
    X_test_scaled_x = scaler_x.transform(input_data)
    X_test_pca_x = pca_x.transform(X_test_scaled_x)
    predictions_x = model_x.predict(X_test_pca_x)

    # Scale and transform the data for cursor_y prediction
    X_test_scaled_y = scaler_y.transform(input_data)
    X_test_pca_y = pca_y.transform(X_test_scaled_y)
    predictions_y = model_y.predict(X_test_pca_y)

    # Applying Kalman filter to smooth the predictions
    kf_x = KalmanFilter(initial_state_mean=predictions_x[0], n_dim_obs=1)
    kf_y = KalmanFilter(initial_state_mean=predictions_y[0], n_dim_obs=1)

    # Fit the Kalman filter and smooth the predictions
    predictions_x_smoothed, _ = kf_x.smooth(predictions_x)
    predictions_y_smoothed, _ = kf_y.smooth(predictions_y)

    # Convert smoothed predictions to a DataFrame
    predicted_df = pd.DataFrame({
        'predicted_cursor_x': predictions_x_smoothed.flatten(),
        'predicted_cursor_y': predictions_y_smoothed.flatten()
    })

    # Save predictions to a CSV file
    predicted_df.to_csv('data/predictions_smoothed1.csv', index=False)

    print("Smoothed predictions saved to 'predictions_smoothed.csv'.")

    # Load data from predictions_smoothed.csv
    data1 = pd.read_csv('data/predictions_smoothed1.csv')
    cursor_x = data1['predicted_cursor_x'].to_numpy()
    cursor_y = data1['predicted_cursor_y'].to_numpy()

    # Initialize heatmap parameters for 1920x1080 screen
    screen_width, screen_height = 1920, 1080
    heatmap = np.zeros((screen_height, screen_width), dtype=np.float32)

    # Populate heatmap with cursor positions
    for x, y in zip(cursor_x, cursor_y):
        # Accumulate points, increasing intensity for repeated positions
        cv2.circle(heatmap, (int(x), int(y)), 15, 1, -1)

    # Apply Gaussian blur to create smoother heat zones
    heatmap = cv2.GaussianBlur(heatmap, (35, 35), 0)

    # Normalize heatmap to [0, 1] for color mapping
    heatmap_normalized = np.clip(heatmap / heatmap.max(), 0, 1)

    # Apply a color map for visualization
    heatmap_colored = cm.get_cmap('jet')(heatmap_normalized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Load the background image and resize to match heatmap dimensions
    background_image = cv2.imread('GeminiModel\s1.webp')
    background_image = cv2.resize(background_image, (screen_width, screen_height))

    brightness_offset = 100  # Adjust brightness level (higher value = brighter)
    background_image = cv2.add(background_image, np.full_like(background_image, brightness_offset))

    # Convert heatmap to BGR for blending
    heatmap_colored_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    # Blend the heatmap with the background image
    alpha = 0.6  # Transparency level of the heatmap
    blended_image = cv2.addWeighted(heatmap_colored_bgr, alpha, background_image, 1 - alpha, 0)

    # Save the final blended image
    cv2.imwrite('heatmap_with_background.png', blended_image)

    # Display the blended image using matplotlib
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    plt.imshow(blended_image_rgb)
    plt.axis('off')  # Hide axes
    plt.show()



    csv_file = 'data/predictions_smoothed1.csv'  # Replace with your file
    data = pd.read_csv(csv_file)

    # Extract x and y coordinates
    x = data['predicted_cursor_x'].values
    y = data['predicted_cursor_y'].values

    # Estimate density using Kernel Density Estimation
    kde = gaussian_kde([x, y])
    density = kde([x, y])

    # Find the region with the highest density
    threshold = 0.9 * np.max(density)  # Set a density threshold
    dense_points = np.where(density >= threshold)[0]

    # Get bounding box for the densest region
    x_dense = x[dense_points]
    y_dense = y[dense_points]

    x_min, x_max = int(np.min(x_dense)), int(np.max(x_dense))
    y_min, y_max = int(np.min(y_dense)), int(np.max(y_dense))

    # Expand the bounding box slightly for better cropping
    padding = 500  # Adjust padding as needed
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max += padding
    y_max += padding

    # Load the image
    image_path = 'GeminiModel\s1.webp'  # Replace with your image path
    image = cv2.imread(image_path)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Save or display the cropped image
    cv2.imwrite('cropped_image.jpg', cropped_image)  # Save the cropped image
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # Display with matplotlib
    plt.title("Cropped Region")
    plt.show()
