import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

print("opencv", cv2.__version__)
print("mediapipe", mp.__version__)
print("PoseLandmarker ready:", vision.PoseLandmarker is not None)