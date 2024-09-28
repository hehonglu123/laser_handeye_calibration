# 2D Laser Camera Handeye Calibration using a Triangle Target.

## Underline Algo
<img src="images/calibration_algo.png" alt="Alt Text" width="600"  height="auto">


## Procedures

### Points Teaching
Start RR robot driver. Jog to triangle vertices in Teach mode and use `save_q.py` to log the joint angles.

### Scan Motion
`robot_laser_motion_triangle.py` will generate a rough motion to scan over the trangle. Monitor the live scan to make sure the scan covers the entire triangle. Also make move the scan motion up and down.
[View 3D Model](model/triangle.stl)

<img src="images/calibration_motion.jpg" alt="Alt Text" width="300"  height="auto">

### Processing
Stack up all frames in time axis. Use `scan_playback.py`, `pc_playback.py` and `edge_detection.py` to check each step's result.
The scripts will identify triangle vertices in time and space, and display as 2D+time 3D pointcloud.

<img src="images/2d+time.png" alt="Alt Text" width="800"  height="auto">

<img src="images/edge_detection.png" alt="Alt Text" width="300"  height="auto">


Run `calibration.py` to generate the transformation matrix. 
