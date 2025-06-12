import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from scipy import stats
import pyautogui
import subprocess
import time
import pygetwindow as gw
from cvzone.HandTrackingModule import HandDetector
import pandas as pd
from datetime import datetime


model = load_model('final_model_30_frames_200_videos.h5')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

actions = np.array(['zoom_in', 'reset_zoom', 'next_slide', 'prev_slide', 'annotation', 'pointer'])

colors = [(245,117,16), (117,245,16), (16,117,245), (200,100,10), (300,200,20), (45,150,100)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def extract_hand(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame.flags.writeable = False                  
    results = model.process(frame)                 
    frame.flags.writeable = True                   
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR

    # Check if any hand is detected
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks.landmark
    elif results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks.landmark
    else:
        return None, results

    # Get bounding box
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    h, w, _ = frame.shape
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)

    # Add padding
    padding = 20
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

    # Crop and resize
    hand_crop = frame[y_min:y_max, x_min:x_max]

    # Check if the crop is empty
    if hand_crop.size == 0:
        return None, results

    return hand_crop, results

pyautogui.FAILSAFE = False

width, height = 1280, 720 

ppt_window_cache = None
annotation_active = False
pointer_active = False 
prev_annotation_point= None
panning_active = False
zoom_level = 0

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


folderPath = "PPTFiles"           
pptx_file_path = os.path.join(folderPath, 'presentation.pptx')
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Launch PowerPoint and start slideshow
subprocess.Popen(['start', pptx_file_path], shell=True)
time.sleep(1) 
pyautogui.hotkey('fn', 'f5')
time.sleep(1)

def focus_ppt_window():
    global ppt_window_cache
    try:
        if ppt_window_cache is not None:
            return ppt_window_cache
        windows = gw.getWindowsWithTitle("PowerPoint")
        if windows:
            ppt_window = windows[0]
            ppt_window.restore()  # Ensure PowerPoint is not minimized
            ppt_window.activate()
            ppt_window.maximize()
            time.sleep(0.1)
            ppt_window_cache = ppt_window
            return ppt_window
        else:
            print("No PowerPoint window found.")
            return None
    except Exception as e:
        print("Error focusing PowerPoint window:", e)
        return None
    
def map_coords(indexFinger, ppt_window):
    ppt_left, ppt_top = ppt_window.left, ppt_window.top
    ppt_width, ppt_height = ppt_window.width, ppt_window.height
    # Map full camera coordinates to PowerPoint window
    xVal = int(np.interp(indexFinger[0], [0, width], [ppt_left, ppt_left + ppt_width]))
    yVal = int(np.interp(indexFinger[1], [0, height], [ppt_top, ppt_top + ppt_height]))
    return xVal, yVal

def perform_annotation():
    global annotation_active, pointer_active, prev_annotation_point

    ppt_window = focus_ppt_window()
    if ppt_window is None:
        print("[ERROR] PowerPoint window not found. Aborting annotation update.")
        return
    
    fingers = [1, 1, 0, 0, 1]
    while fingers == [1, 1, 0, 0, 1]:  
        success, img = cap.read()
        if not success:
            print("[ERROR] Failed to capture frame. Exiting annotation loop.")
            break

        img = cv2.flip(img, 1)
        hands, img = detectorHand.findHands(img)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detectorHand.fingersUp(hand)
            indexFinger = (lmList[8][0], lmList[8][1])

            # print values for verification
            '''print("\n\n\n\n############################################")
            print(f'hand value is {hand}\n')
            print(f'landmark list value is {lmList}\n')
            print(f'fingers value is {fingers}\n')
            print(f'indexfinger coordinates are {indexFinger}\n')
            print("############################################\n\n\n\n")'''

        else:
            print("hands not detected")

        try:
            xVal, yVal = map_coords(indexFinger, ppt_window)
        except Exception as e:
            print(f"[ERROR] Mapping coordinates for annotation failed: {e}")
            return
        
        # print values for verification
        '''print("\n\n\n\n############################################")
        print(f'mapped x value is {xVal}\n')
        print(f'mapped y value is {yVal}\n')
        print("############################################\n\n\n\n")'''

        if pointer_active:
            pointer_active = False
            try:
                pyautogui.mouseUp()
            except Exception as e:
                print("Error releasing pointer mode:", e)

        if not annotation_active:
            try:
                pyautogui.hotkey('ctrl', 'p')  
                time.sleep(0.05)
                annotation_active = True
                prev_annotation_point = (xVal, yVal)
                print("[INFO] Annotation mode activated.")
            except Exception as e:
                print(f"[ERROR] Activating annotation mode failed: {e}")
                return
        else:
            if prev_annotation_point is not None:
                try:
                    pyautogui.mouseDown(button='left')
                    prev_x, prev_y = prev_annotation_point
                    pyautogui.moveTo(prev_x, prev_y)

                    print(f"[INFO] Annotation drawn from {prev_annotation_point} to: ({xVal}, {yVal})")
                    prev_annotation_point = (xVal, yVal)

                except Exception as e:
                    print(f"[ERROR] Dragging for annotation failed: {e}")
            else:
                prev_annotation_point = (xVal, yVal)

def perform_pointer():
    print("pointer code must be here")

    #take the coordinates here only
    global annotation_active, pointer_active, prev_annotation_point

    ppt_window = focus_ppt_window()
    if ppt_window is None:
        print("[ERROR] PowerPoint window not found. Aborting Pointer update.")
        return
    
    print("PPT window focused...\n")
    
    fingers = [0, 1, 0, 0, 0]
    while fingers == [0, 1, 0, 0, 0]:
        success, img = cap.read()
        if not success:
            print("[ERROR] Failed to capture frame. Exiting Pointer loop.")
            break

        img = cv2.flip(img, 1)
        hands, img = detectorHand.findHands(img)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detectorHand.fingersUp(hand)
            indexFinger = (lmList[8][0], lmList[8][1])

            # print values for verification
            '''print("\n\n\n\n############################################")
            print(f'hand value is {hand}\n')
            print(f'landmark list value is {lmList}\n')
            print(f'fingers value is {fingers}\n')
            print(f'indexfinger coordinates are {indexFinger}\n')
            print("############################################\n\n\n\n")'''

        else:
            print("hands not detected")

        try:
            xVal, yVal = map_coords(indexFinger, ppt_window)
        except Exception as e:
            print(f"[ERROR] Mapping coordinates for pointer failed: {e}")
            return
        
        # print values for verification
        '''print("\n\n\n\n############################################")
        print(f'mapped x value is {xVal}\n')
        print(f'mapped y value is {yVal}\n')
        print("############################################\n\n\n\n")'''

        # Only activate pointer mode if it's not already active
        if not pointer_active:
            if annotation_active:
                try:
                    pyautogui.mouseUp()  # Release annotation mode if it was active
                except Exception as e:
                    print("Error releasing annotation mode:", e)
                annotation_active = False
                prev_annotation_point = None
                print("Annotation mode deactivated.")

            try:
                pyautogui.hotkey('ctrl', 'l')  # Activate pointer mode once
                pointer_active = True
                print("Pointer mode activated.")
            except Exception as e:
                print("Error activating pointer mode:", e)
        else:
            print("Pointer mode already active; moving pointer.")

        try:
            pyautogui.moveTo(xVal, yVal, duration=0.1)
            print(f"Pointer moved to: ({xVal}, {yVal})")
        except Exception as e:
            print("Error moving pointer:", e)

    pointer_active = False
    pyautogui.hotkey('ctrl', 'l')

def perform_pann():
    global panning_active, zoom_level
    print(f'performing pann zoom lvl is {zoom_level}')

    ppt_window = focus_ppt_window()
    if ppt_window is None:
        print("PowerPoint window not found. Aborting pan function.")
        return

    # Ensure PowerPoint is focused
    ppt_window.activate()
    time.sleep(0.1)

    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to capture frame. Exiting Pointer loop.")
        return

    img = cv2.flip(img, 1)
    hands, img = detectorHand.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detectorHand.fingersUp(hand)

        if fingers != [1, 1, 1, 1, 1] and fingers[1] == 1:  # not zoom in action and index finger must be up
            x, y = lmList[8][0], lmList[8][1]  # Index finger tip position

            # Convert hand coordinates to screen coordinates
            screen_x, screen_y = map_coords((x, y), ppt_window)

            if not panning_active:  # Start panning
                pyautogui.mouseDown()  # Click and hold
                panning_active = True
                print("Panning started")

            pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Move mouse
            print(f"Panning to: ({screen_x}, {screen_y})")

        else:
            if fingers == [1, 1, 1, 1, 1]:
                print("zoom_in action")
            elif fingers[1] == 0:
                print("index finger isn't up")
            if panning_active:  # Stop panning
                pyautogui.mouseUp()  
                panning_active = False
                print("Panning stopped")

    else:
        print("hands not detected, panning cannot be done... exiting")

def perform_slide_nav(action):
    global zoom_level

    if zoom_level > 0 and (action[0] == "zoom_in" or action[0] == "reset_zoom") :
        print(f"zoom_level is {zoom_level}, and action is {action[0]}")
        perform_pann()

    if action[0] == "next_slide":
        pyautogui.mouseUp()
        pyautogui.press('right')
        print("Next slide")

    elif action[0] == "prev_slide":
        pyautogui.mouseUp()
        pyautogui.press('left')
        print("Previous slide")

    elif action[0] == "zoom_in":
        if zoom_level == 3: # max zoom level reached, so return
            perform_pann()
            return
        zoom_level+=1 # increase zoom level
        pyautogui.mouseUp()
        pyautogui.hotkey('ctrl', '+')
        print("Zoom In")
        if zoom_level > 0:
            perform_pann()

    elif action[0] == "reset_zoom":
        if zoom_level == 0: # min zoom level reached, so return
            return
        zoom_level-=1 # decrease zoom level
        pyautogui.mouseUp()
        pyautogui.hotkey('ctrl', '-')
        print("Zoom Out")
        if zoom_level > 0:
            perform_pann()

    elif action[0] == "pointer":
        perform_pointer()

    elif action[0] == "annotation":
        perform_annotation()

# Using the model here
sequence = []
sentence = []
predictions = []
threshold = 0.8
confidence_score = 0


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if frame is not captured properly

        # Extract hand region
        hand_crop, results = extract_hand(frame, holistic)

        # Draw landmarks on the original frame (for visualization)
        draw_styled_landmarks(frame, results)

        # Default to an empty keypoints array if no hand is detected
        keypoints = np.zeros((126,)) if hand_crop is None else extract_keypoints(results)

        if np.all(keypoints == 0):  
            print("No hands detected")
            sequence = []
            predictions = []
            # DEACTIVATE THE POINTER/ANNOTATION IF HAND ISN'T DETECTED
            if pointer_active or annotation_active:
                try:
                    pyautogui.mouseUp()
                except Exception as e:
                    print("Error releasing pointer/annotation mode:", e)
                pointer_active = False
                annotation_active = False

        else:
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:

                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    
                    confidence_score = res[np.argmax(res)]

                    if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:

                        sentence.append(actions[np.argmax(res)])

                    predictions = [] # resets the prediction value
                    sequence = []

                if len(sentence) > 1: 

                    sentence = sentence[-1:]

                # Probability visualization on frame
                frame = prob_viz(res, actions, frame, colors)

                if len(sentence) > 0:
                    print(f'current action is {sentence[0]}')
                    perform_slide_nav(sentence)
        
        # Overlay text on full frame
        cv2.putText(frame, f'{sentence}: {confidence_score:.2f}', (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the full frame, not just the cropped hand
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
