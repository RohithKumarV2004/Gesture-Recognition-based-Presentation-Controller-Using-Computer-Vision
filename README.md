# PowerPoint Gesture Controller using LSTM-based Hand Tracking

This project implements a deep learning–based gesture recognition system that enables hands-free control of PowerPoint presentations using dynamic hand gestures. By leveraging LSTM networks and real-time hand landmark detection using MediaPipe, users can navigate slides, annotate, zoom, and use a virtual pointer—all through intuitive hand movements captured via webcam.

## Folder Structure

```
Final Codes/
├── final_model_30_frames_200_videos.h5       # Trained LSTM model
├── only_hands_training_code.ipynb            # Jupyter notebook for training the model
├── ppt_control_code.py                       # Script to control PowerPoint using gestures
├── PPTFiles/
│   └── presentation.pptx                     # Sample PowerPoint file for testing
├── recording_training_videos.py              # Script to record training videos
├── training_videos/
│   └── [Action_Folder(s)]                    # Contains videos organized by gesture action
└── Data_30_Frame_Limit_200_Videos/           # Auto-generated during training; contains .npy files
```

## How It Works

### 1. Data Collection (`recording_training_videos.py`)
- Uses webcam to record short (1s) gesture videos at 120 FPS.
- Each recording captures **30 frames**, stored in folders named after each gesture.
- A separate folder is created per action containing all videos of that gesture.

### 2. Data Preprocessing & Model Training (`only_hands_training_code.ipynb`)
- Extracts **hand landmarks** using MediaPipe Holistic (only hand keypoints used).
- Saves each frame's 3D landmarks (21 keypoints × 3D) as `.npy` files → shape: `(30, 126)`.
- Trains an LSTM-based model:
  - 3 LSTM layers (64, 128, 128 units)
  - Followed by Dense layers with Dropout
  - Final output: Softmax (6 gesture classes)
- Accuracy: **~99%** (achieved in under 40 epochs).
- Saved model: `final_model_30_frames_200_videos.h5`.

### 3. Gesture Classification & Presentation Control (`ppt_control_code.py`)
- Loads the trained model.
- Detects real-time hand gestures from webcam.
- Maps gesture predictions to PowerPoint actions using `pyautogui`.
- Sample presentation file included in `PPTFiles/`.

## Defined Gestures and Actions

| Gesture         | Action Performed              |
|-----------------|-------------------------------|
| Swipe Right     | Next Slide                    |
| Swipe Left      | Previous Slide                |
| Flex Fingers    | Zoom In                       |
| Close Fist      | Reset Zoom                    |
| Thumb + Index + Little Finger | Annotate      |
| Index Finger Only | Pointer Mode               |

## Model Input Specification

- **Input shape**: `(30, 126)`  
  → 30 frames per gesture, 126 features per frame  
  (21 hand landmarks × 3 coordinates × 2 hands)

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **MediaPipe Holistic** (for real-time hand tracking)
- **OpenCV** (video capture & processing)
- **NumPy** (data handling)
- **pyautogui** (keyboard/mouse emulation)

## How to Run

1. **Record Training Videos:**
   ```bash
   python recording_training_videos.py
   ```

2. **Train Model:**
   - Open `only_hands_training_code.ipynb` in Jupyter Notebook.
   - Run all cells to generate `.npy` data and train the model.
   - The model will be saved as `final_model_30_frames_200_videos.h5`.

3. **Run Presentation Controller:**
   ```bash
   python ppt_control_code.py
   ```

## Results

- Final model achieves over **95% accuracy**.
- Real-time gesture control of presentations.
- Supports 6 distinct hand gestures for complete slide control.

## Notes

- **Data Folder (`Data_30_Frame_Limit_200_Videos/`)** is auto-generated during training.
- Ensure webcam is properly connected and accessible.
- `pyautogui` may require admin permissions for simulating inputs on some systems.


## Acknowledgements

- [Google MediaPipe](https://github.com/google/mediapipe)
- [TensorFlow](https://www.tensorflow.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
