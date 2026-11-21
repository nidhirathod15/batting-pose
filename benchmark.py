import platform
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker_lite.task"
VIDEO_DIR = ROOT / "videos"
WARMUP_FRAMES = 5  # the first few frames are slower while the model warms up; don't count them


def time_clip(video_path):
    """Time every frame of one clip. Returns two lists of seconds: model-only and whole-pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
    )
    model_times, pipeline_times = [], []
    n = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            t_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t_model = time.perf_counter()
            landmarker.detect_for_video(image, int(n * 1000 / fps))
            t_end = time.perf_counter()

            if n >= WARMUP_FRAMES:
                model_times.append(t_end - t_model)
                pipeline_times.append(t_end - t_start)
            n += 1
    cap.release()
    return model_times, pipeline_times


def main():
    if not MODEL.exists():
        raise SystemExit(f"Model file not found: {MODEL}")
    clips = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    if not clips:
        raise SystemExit(f"No .mp4 clips found in {VIDEO_DIR}")

    print(f"Machine: {platform.processor() or platform.machine()} | Python {platform.python_version()}\n")
    all_model, all_pipeline = [], []
    for clip in clips:
        model_t, pipe_t = time_clip(clip)
        if not pipe_t:
            print(f"{clip.name}: too short to time, skipping")
            continue
        all_model += model_t
        all_pipeline += pipe_t
        print(f"{clip.name}: {len(pipe_t)} frames | model only {statistics.mean(model_t)*1000:.1f} ms/frame "
              f"({1/statistics.mean(model_t):.0f} FPS) | whole pipeline {1/statistics.mean(pipe_t):.0f} FPS")

    print(f"\nAll clips: model only {1/statistics.mean(all_model):.0f} FPS, "
          f"whole pipeline (read + convert + detect) {1/statistics.mean(all_pipeline):.0f} FPS")
    print("Whole pipeline = reading the frame, converting colours and running the model. Drawing and saving are not included.")


if __name__ == "__main__":
    main()