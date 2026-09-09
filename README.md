
# AprilTag Position Estimation

A computer vision system that uses **five AprilTags** to estimate the position and height of a middle marker.

Four AprilTags are placed at the four corners/edges. These four markers are used to establish a reference plane . The fifth, middle marker can then be tracked relative to this plane to estimate its 3d position. 

## Setup

* 4 reference AprilTags: IDs `0–3`
* 1 middle AprilTag: ID `4`
* AprilTag family: `tagStandard41h12`
* Camera input 

## Requirements

- opencv-python 
- numpy 
- pupil-apriltags
