# Task 04: Real-Time Hand Gesture Demo

## Overview

This project uses a webcam and **MediaPipe Hands** to detect hand landmarks in real time. It then applies simple geometric rules to label a few upright-hand poses: Open Palm, Victory, Thumbs Up, and Fist.

This is a small **landmark-based, rule-driven prototype**. MediaPipe supplies the hand landmarks; the gesture labels are assigned by this project's rules. The code does not train its own machine-learning gesture classifier, and the rules may not work reliably for every hand orientation, camera angle, or lighting condition.

## How it works

1. Open the default webcam with OpenCV.
2. Mirror each video frame and convert it from BGR to RGB.
3. Pass the frame to MediaPipe Hands to detect hand landmarks.
4. Compare fingertip and joint positions to estimate which fingers are extended.
5. Display the detected landmarks and the rule-based gesture label.
6. Press **q** to exit and release the webcam.

## Requirements

- Python 3.10 (recommended; verify compatibility for your installed MediaPipe version)
- OpenCV
- MediaPipe

Install dependencies:

```bash
python -m pip install opencv-python mediapipe
```

## Run

```bash
python SCT_04.py
```

Allow webcam access if prompted. Show one hand upright in front of the camera. Press **q** while the preview window is active to quit.

## Gesture rules

- **Open Palm:** all four non-thumb fingers appear extended.
- **Victory:** index and middle fingers appear extended while ring and little fingers appear folded.
- **Thumbs Up:** the four non-thumb fingers appear folded and the thumb tip is above its interphalangeal joint.
- **Fist:** the four non-thumb fingers appear folded and the thumbs-up rule does not match.
- **Unknown:** the detected pose does not match these simple rules.

These are geometric heuristics, not a learned classifier. They are sensitive to orientation and should be treated as a demonstration.

## Limitations and possible improvements

- Only one hand is processed.
- The rules assume a mostly upright hand and can mislabel rotated or partially occluded poses.
- There is no trained model, gesture dataset, accuracy benchmark, or confidence score.
- Next steps could include testing across users and lighting conditions, adding orientation-robust landmark features, collecting labelled examples, and evaluating a trained classifier on a held-out dataset.
