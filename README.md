# Real-Time Hand Gesture Recognition

A computer vision project exploring hand gesture recognition using **MediaPipe hand landmarks**, with both rule-based and machine learning approaches.

The project uses **OpenCV** for webcam capture and visualization and **MediaPipe Hand Landmarker** to detect 21 landmarks on each hand.

This repository explores two approaches:

1. **Rule-based real-time recognition** — gestures are identified using manually defined relationships between hand landmarks.
2. **Machine learning classification experiment** — landmark coordinates are used as features to train and evaluate a gesture classifier.

---

## Features

- Real-time webcam hand tracking
- Detection of up to two hands
- 21-point hand landmark visualization
- Left/right hand identification
- Rule-based gesture recognition
- Landmark data collection and feature engineering
- Machine learning gesture classification
- Model evaluation using classification metrics
- Prediction on unseen hand images

---

## Supported Gestures

### Rule-Based Real-Time Recognizer

| Gesture | Prediction |
| --- | --- |
| ✋ Open Palm | `OPEN PALM` |
| ✊ Fist | `FIST` |
| ✌️ Peace | `PEACE` |
| ☝️ Pointing | `POINTING` |
| 👍 Thumbs Up | `THUMBS UP` |

### Machine Learning Experiment

The current dataset contains four gesture classes:

- `OPEN_PALM`
- `FIST`
- `PEACE`
- `POINTING`

---

# Approach 1: Real-Time Rule-Based Recognition

## How It Works

The real-time application follows this pipeline:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Landmark-Based Gesture Rules
   ↓
Gesture Prediction
```

MediaPipe provides 21 normalized `(x, y, z)` landmark coordinates for each detected hand.

For example, landmark `8` represents the index fingertip while landmark `6` represents a lower joint of the index finger.

For an upright hand:

```python
index_up = landmarks[8].y < landmarks[6].y
```

can be used as a simple approximation of whether the index finger is raised.

The states of multiple fingers are then combined to identify gestures such as Peace, Open Palm and Fist.

## Running the Real-Time Application

After completing the installation steps below:

```bash
python src/main.py
```

A webcam window should open.

Try showing different supported gestures to the camera.

Press `q` to exit.

That's it! Once you're all set up, have fun trying out the different gestures ✋✌️👍

---

# Approach 2: Machine Learning Gesture Classification

The second part of the project explores a more **data-driven approach** to gesture recognition rather than relying entirely on manually defined rules.

The pipeline is:

```text
Hand Image / Webcam Frame
        ↓
MediaPipe Hand Landmarker
        ↓
21 Hand Landmarks
        ↓
Feature Engineering
        ↓
Train Gesture Classifier
        ↓
Model Evaluation
        ↓
Prediction on Unseen Images
```

MediaPipe is used as a **pretrained computer vision feature extractor**. Instead of training directly from raw image pixels, the detected hand is first represented using 21 hand landmarks.

---

## How Was the Dataset Obtained?

The prepared dataset was generated using:

```bash
python src/collect_data.py
```

The data collection script opens the webcam and uses MediaPipe to detect the hand.

The controls are:

```text
O → Open Palm
P → Peace
F → Fist
I → Pointing
Q → Quit
```

For every detected hand, MediaPipe returns:

```text
21 landmarks × (x, y, z)
```

giving **63 landmark values**.

### Feature Engineering

Raw landmark coordinates depend on where the hand appears in the image.

To reduce this dependence, each landmark is represented relative to the wrist (landmark `0`):

```python
relative_x = landmark.x - wrist.x
relative_y = landmark.y - wrist.y
relative_z = landmark.z - wrist.z
```

Each sample therefore contains:

```text
63 landmark features + gesture label
```

The resulting data is stored in:

```text
data/gesture_landmarks.csv
```

A prepared dataset is already included in the repository, so you **do not need to collect your own data** to run the classification experiment.

However, you can run `collect_data.py` if you want to collect additional samples and experiment with your own data.

---

## Step 1: Train the Gesture Classifier

The trained gesture classifier is **not included in the repository**.

You must first train it locally by running:

```bash
python src/train_classifier.py
```

The script will:

1. Load the prepared landmark dataset.
2. Separate the landmark features and gesture labels.
3. Split the data into training and test sets.
4. Train a Random Forest classifier.
5. Evaluate predictions on the test set.
6. Display:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
7. Save the trained classifier locally for use during prediction.

Once this step is complete, you can move on to testing your own unseen images.

---

## Step 2: Test an Unseen Hand Image

Find or take a hand gesture image that was **not part of the training dataset**.

Then run:

```bash
python src/predict_image.py <path-to-image>
```

For example:

```bash
python src/predict_image.py "path/to/fist.png"
```

The script will:

1. Load the unseen image.
2. Detect the hand using MediaPipe.
3. Extract the 21 hand landmarks.
4. Visualize the detected landmarks.
5. Apply the same feature transformation used during training.
6. Load the classifier you trained in Step 1.
7. Predict the gesture.
8. Display the prediction probabilities.

Example:

```text
Hand detected: Yes

Prediction: FIST

Prediction probabilities:
FIST         : 0.82
OPEN_PALM    : 0.08
PEACE        : 0.06
POINTING     : 0.04
```

Try experimenting with:

- Different people
- Left and right hands
- Different camera angles
- Different backgrounds
- Different distances
- Slightly rotated gestures

Then compare the results with the evaluation metrics you obtained during training.

---

## Understanding the Evaluation

The current dataset was collected from short webcam recording sessions. This means that many samples correspond to **consecutive video frames** and may be highly similar.

For example:

```text
Frame 1 → PEACE
Frame 2 → PEACE
Frame 3 → PEACE
Frame 4 → PEACE
```

A random train/test split may place very similar frames into both the training and test sets.

This can result in **overly optimistic evaluation results**.

Therefore, strong performance on the initial test split should not automatically be interpreted as strong generalization to completely new hands or images.

When testing unseen images, consider:

- Does the prediction match the actual gesture?
- How confident is the classifier?
- Are the MediaPipe landmarks correctly detected?
- If the prediction is wrong, did the landmark extraction fail or did the classifier fail?
- Does the evaluation method reflect how the model would be used in practice?

---

# Tech Stack

- Python
- OpenCV
- MediaPipe
- pandas
- scikit-learn

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/aleeax/real-time-hand-gesture-recognition.git
cd real-time-hand-gesture-recognition
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project expects the MediaPipe Hand Landmarker model to be available at:

```text
models/hand_landmarker.task
```

---

# Limitations

## Rule-Based Recognition

The rule-based recognizer relies on simple geometric relationships between landmarks and may be affected by:

- Hand orientation
- Camera angle
- Partial occlusion
- Ambiguous finger positions

## Machine Learning Classification

The current classification experiment is an initial prototype. The dataset has limited diversity because samples were collected from short webcam sessions and a limited number of hands.

The classifier may therefore be affected by:

- Highly correlated consecutive frames
- Limited variation between users
- Hand orientation
- Left/right hand differences
- Camera distance and scale
- Different ways of performing the same gesture

The current evaluation should therefore be treated as a **baseline experiment rather than a measure of production-level performance**.

---

# Future Improvements

- Collect data from multiple participants and recording sessions
- Use participant/session-based train/test splits
- Build a larger independent test set
- Normalize landmarks for hand scale
- Explore rotation-invariant features
- Engineer landmark distances and joint-angle features
- Compare different classifiers
- Tune classifier hyperparameters
- Add temporal smoothing for real-time predictions
- Support additional gestures
- Integrate the trained classifier into the real-time webcam application

---

## Acknowledgements

Hand landmark detection is powered by Google's **MediaPipe Hand Landmarker**.