
# AprilTag Position Estimation

A computer vision system that uses **five AprilTags** to estimate the position and height of a middle marker.

Four AprilTags are placed at the four corners/edges. These four markers are used to establish a reference plane and coordinate system. The fifth, middle marker can then be tracked relative to this plane to estimate its **X/Y position, depth, and approximate height**.

The system uses OpenCV and `pupil_apriltags` for real-time detection.

## Setup

* 4 reference AprilTags: IDs `0–3`
* 1 middle AprilTag: ID `4`
* AprilTag family: `tagStandard41h12`
* Camera input 

## Requirements

```bash
pip install opencv-python numpy pupil-apriltags
```

## Run

```bash
python main.py
```
