import cv2
import os
import time
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define actions
actions = ['zoom_in', 'zoom_out', 'prev_slide', 'next_slide', 'annotation', 'pointer']

# Parameters
video_folder = "videos_5"  # Root folder to save videos
frame_limit = 30
  # Number of frames per video
videos_per_action = 100  # Number of videos per action

# Create root folder if not exists
os.makedirs(video_folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 30  # Frames per second

last_valid_frame_count = 0  # Store valid frames from the last sequence

for action in actions:
    action_path = os.path.join(video_folder, action)
    os.makedirs(action_path, exist_ok=True)  # Create action folder

    for video_num in range(1, videos_per_action + 1):
        video_name = f"video_{video_num}.avi"
        video_path = os.path.join(action_path, video_name)

        # Show countdown before recording starts (with last valid frame count)
        for i in range(2, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display last valid frame count
            message = f"Get Ready! Recording {action} in {i} sec..."
            if video_num > 1:  # Show valid frame count only after the first video
                message += f" (Last Valid Frames: {last_valid_frame_count}/{frame_limit})"
            
            #v2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Get Ready! Recording {action} in {i} sec...", (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if video_num > 1:
                cv2.putText(frame, f"(Last Valid Frames: {last_valid_frame_count}/{frame_limit})", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Recording...", frame)
            cv2.waitKey(1000)  # 1-second delay

        # Set up video writer
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        print(f"Recording {video_name} for action '{action}' ({video_num}/{videos_per_action})")
        frame_count = 0
        valid_frame_count = 0  # Count frames where a hand is detected

        while frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (for MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)  # Detect hand landmarks

            # Check if hand is detected
            if results.multi_hand_landmarks:
                valid_frame_count += 1  # Count valid frames

            out.write(frame)  # Save frame to video
            frame_count += 1

            # Display recording message with valid frame count
            cv2.putText(frame, f"Recording: {action} ({video_num}/{videos_per_action})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Valid Frames: {valid_frame_count}/{frame_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Recording...", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
                break

        out.release()
        print(f"Saved: {video_path} | Valid Frames: {valid_frame_count}/{frame_count}")

        # Store last valid frame count for display in the next sequence
        last_valid_frame_count = valid_frame_count

cap.release()
cv2.destroyAllWindows()
print("All videos recorded successfully!")
