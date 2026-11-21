import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker_lite.task"
VIDEO_DIR = ROOT / "videos"
OUT_DIR = ROOT / "outputs"

NUM_POINTS = 33
CSV_HEADER = ["frame", "time_s"] + [f"p{i}_{a}" for i in range(NUM_POINTS) for a in ("x", "y", "vis")]


def draw_skeleton(frame, landmarks):
    """Draw dots on the 33 body points and lines between connected ones."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def process_clip(video_path):
    """Run pose detection on every frame of one clip. Returns (frames_total, frames_detected)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = OUT_DIR / f"{video_path.stem}_skeleton.mp4"
    out_csv = OUT_DIR / f"{video_path.stem}_landmarks.csv"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # VIDEO mode remembers the previous frame, so the person is tracked
    # between frames. It needs a fresh landmarker for each clip, because
    # its timestamps must only ever go up.
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
    )
    total = detected = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            open(out_csv, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(CSV_HEADER)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(total * 1000 / fps)
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)
                row = [total, round(total / fps, 3)]
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.visibility]
                csv_writer.writerow(row)
                detected += 1

            writer.write(frame)
            total += 1

    cap.release()
    writer.release()
    return total, detected


def main():
    if not MODEL.exists():
        raise SystemExit(f"Model file not found: {MODEL}")
    clips = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    if not clips:
        raise SystemExit(f"No .mp4 clips found in {VIDEO_DIR}")

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Found {len(clips)} clip(s).\n")
    for clip in clips:
        total, detected = process_clip(clip)
        pct = 100 * detected / total if total else 0
        print(f"{clip.name}: person found in {detected} of {total} frames ({pct:.1f}%)")
    print(f"\nSaved skeleton videos and landmark CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()