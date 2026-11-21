import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save the plot as an image, don't open a window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

MIN_VISIBILITY = 0.5  # ignore a point the model isn't confident about

# joint name -> (point before, the joint itself, point after), using MediaPipe point numbers
JOINTS = {
    "left elbow": (11, 13, 15),    # shoulder, elbow, wrist
    "right elbow": (12, 14, 16),
    "left knee": (23, 25, 27),     # hip, knee, ankle
    "right knee": (24, 26, 28),
}


def load_landmarks(csv_path):
    """Read a landmarks CSV. Returns frame numbers and an array shaped (rows, 33, 3) of x, y, visibility."""
    frames, data = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            data.append([[float(row[f"p{i}_x"]), float(row[f"p{i}_y"]), float(row[f"p{i}_vis"])]
                         for i in range(33)])
    return np.array(frames), np.array(data)


def video_info(stem):
    """Width, height and frames-per-second of the clip, read from its skeleton video."""
    cap = cv2.VideoCapture(str(OUT_DIR / f"{stem}_skeleton.mp4"))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return width, height, fps


def angle_at(a, b, c):
    """Angle in degrees at point b, between the line b->a and the line b->c. Inputs have shape (n, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(data, width, height):
    """Angle of every joint in every row. Unsure points give NaN (not a number)."""
    # Points are fractions of the picture (0 to 1). Turn them into pixels first,
    # otherwise a tall or wide picture squashes the angles.
    xy = data[:, :, :2] * np.array([width, height])
    vis = data[:, :, 2]
    angles = {}
    for name, (a, b, c) in JOINTS.items():
        ang = angle_at(xy[:, a], xy[:, b], xy[:, c])
        sure = (vis[:, a] >= MIN_VISIBILITY) & (vis[:, b] >= MIN_VISIBILITY) & (vis[:, c] >= MIN_VISIBILITY)
        ang[~sure] = np.nan
        angles[name] = ang
    return angles


def process(csv_path):
    stem = csv_path.name[: -len("_landmarks.csv")]
    frames, data = load_landmarks(csv_path)
    if len(frames) == 0:
        print(f"{stem}: no detected frames, skipping.")
        return
    width, height, fps = video_info(stem)
    found = joint_angles(data, width, height)

    # Put the angles on a full timeline; frames with no person stay NaN.
    n_frames = frames.max() + 1
    timeline = {name: np.full(n_frames, np.nan) for name in found}
    for name, values in found.items():
        timeline[name][frames] = values
    time_s = np.arange(n_frames) / fps

    out_csv = OUT_DIR / f"{stem}_angles.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + list(JOINTS))
        for i in range(n_frames):
            w.writerow([i, round(time_s[i], 3)] + ["" if np.isnan(timeline[n][i]) else round(float(timeline[n][i]), 1)
                                                    for n in JOINTS])

    plt.figure(figsize=(9, 4.5))
    for name, values in timeline.items():
        plt.plot(time_s, values, label=name)
    plt.xlabel("time (s)")
    plt.ylabel("joint angle (degrees)")
    plt.title(f"{stem}: joint angles through the clip")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{stem}_angles.png", dpi=120)
    plt.close()

    print(f"\n{stem}:")
    for name, values in timeline.items():
        good = values[~np.isnan(values)]
        if len(good):
            print(f"  {name:12s} usable in {len(good)} frames, range {good.min():.0f} to {good.max():.0f} degrees")
        else:
            print(f"  {name:12s} no usable frames")


def main():
    csvs = sorted(OUT_DIR.glob("*_landmarks.csv"))
    if not csvs:
        raise SystemExit("No landmark CSVs in outputs/. Run pose_video.py first.")
    for csv_path in csvs:
        process(csv_path)
    print(f"\nSaved angle CSVs and plots in {OUT_DIR}")


if __name__ == "__main__":
    main()