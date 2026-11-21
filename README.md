# Batting pose analysis

A Python pipeline that finds a batter's body pose in cricket clips, works out
knee and elbow angles for every frame, and measures how fast it runs on a
laptop CPU.

## Why I built this
Batting technique is usually judged by eye, frame by frame. I wanted to see
how far an off-the-shelf pose-estimation model gets on real match footage,
which is harder than a clean lab video: the camera angle is fixed by the
broadcaster, the batter moves fast, and the arms and bat overlap each other.
The questions were: does the model keep finding the batter, can joint angles
be pulled out of the output, how far can those angles be trusted, and is it
fast enough to run on an ordinary laptop?

## How it works
1. **Read the video.** OpenCV reads each clip frame by frame and converts the
   colours from BGR (OpenCV's order) to RGB (what MediaPipe expects).
2. **Find the pose.** MediaPipe's Pose Landmarker (the lite model) returns 33
   body points per frame (shoulders, elbows, wrists, hips, knees, ankles and
   so on). Each point has an x and y position (as a fraction of the picture)
   and a visibility score for how sure the model is. The pipeline runs the
   model in video mode, which tracks the person from one frame to the next
   instead of searching from scratch every time.
3. **Save the results.** Every clip gets a copy with the skeleton drawn on it
   and a CSV of all 33 points for each frame.
4. **Compute joint angles.** The angle at a joint is the angle between the two
   body segments that meet there: elbow = shoulder, elbow, wrist; knee = hip,
   knee, ankle. It is calculated from the two segment vectors with a dot
   product and an arccosine. Positions are converted from fractions to pixels
   first, because a tall or wide frame would otherwise distort the angles. If
   the model is unsure about any of the three points (visibility under 0.5),
   that frame's angle is left blank instead of guessed.
5. **Plot and measure speed.** Angles are plotted over time for each clip. For
   speed, each frame is timed with `time.perf_counter`, skipping the first 5
   frames while the model warms up. Two numbers are reported: the model alone,
   and the whole pipeline (read frame, convert colours, run model). Drawing
   and saving are not included.

## Tools
- Python
- MediaPipe (Pose Landmarker, Tasks API, lite model)
- OpenCV: video reading, colour conversion, drawing the skeleton
- NumPy: angle maths
- Matplotlib: angle plots

## Results
Tested on 3 cricket match clips of 3 different batters.
- The batter was found in 538 of 539 frames (99.8%).
- Speed: 41 FPS for the whole pipeline and 51 FPS for the model alone, on a
  laptop CPU (<your CPU model>), no GPU.
- Example angle output: `docs/swing1_angles.png`

## Limitations
- **Elbow angles were unreliable.** When the arms were raised or overlapping,
  the angle jumped by over 100 degrees between consecutive frames. A stricter
  confidence filter removed most frames and still left jumps.
- Knee angles were steadier in two clips. In one clip the right knee also had
  some bad frames.
- The clips are broadcast footage from a front or three-quarter angle, so
  angles worked out from 2D points are approximate.
- The clips are not in this repo.
- Timings are from one laptop, so treat them as a rough guide.

## Possible next steps
- Smooth the angles and reject impossible jumps, so elbow values become usable.
- Film side-on clips of my own, where angles from 2D points are more accurate.
- Use the angles to tell shots apart (needs many labelled clips).

## How to run
Tested on Python 3.12.10
1. `pip install -r requirements.txt`
2. Download the model and put it in a `models/` folder:
   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
3. Put your .mp4 clips in a `videos/` folder.
4. Run in this order:
   - `python src/pose_video.py` saves a skeleton video and a landmarks CSV per clip
   - `python src/angles.py` saves an angles CSV and a plot per clip
   - `python src/benchmark.py` prints the speed measurements

## Files
- `src/pose_one_frame.py`: skeleton on a single frame (quick check)
- `src/pose_video.py`: pose for every frame of every clip
- `src/angles.py`: joint angles and plots
- `src/benchmark.py`: speed measurement

## Files
- `src/pose_one_frame.py`: skeleton on one frame 
- `src/pose_video.py`: pose for every frame of every clip
- `src/angles.py`: joint angles and plots
- `src/benchmark.py`: speed measurement
