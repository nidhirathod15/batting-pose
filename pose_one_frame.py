from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker_lite.task"
VIDEO = ROOT / "videos" / "swing1.MP4"
OUT = ROOT / "outputs" / "frame_with_skeleton.jpg"
FRAME_NUMBER = 30  # which frame of the video to test on


def read_frame(video_path, frame_number):
    """Return one frame (BGR image) from the video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_number}. The video has {total} frames - lower FRAME_NUMBER.")
    return frame


def draw_skeleton(frame, landmarks):
    """Draw dots on the 33 body points and lines between connected ones."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return points


def main():
    if not MODEL.exists():
        raise SystemExit(f"Model file not found: {MODEL}\nDownload pose_landmarker_lite.task into the models/ folder.")
    if not VIDEO.exists():
        raise SystemExit(f"Video not found: {VIDEO}\nSave your clip as videos/swing.mp4.")

    frame = read_frame(VIDEO, FRAME_NUMBER)

    # MediaPipe expects RGB colours; OpenCV gives BGR, so convert.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    if not result.pose_landmarks:
        raise SystemExit("No person detected in this frame. Try another FRAME_NUMBER, or check your clip.")

    landmarks = result.pose_landmarks[0]  # first (only) person
    points = draw_skeleton(frame, landmarks)

    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), frame)
    print(f"Detected {len(landmarks)} body points.")
    print(f"Right elbow (point 14) is at pixel {points[14]}.")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()