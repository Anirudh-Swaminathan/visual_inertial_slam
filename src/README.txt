# ECE276A Project 3 - Visual Intertial SLAM

Project implemented by Anirudh Swaminathan - A53316083 - UCSD - Winter 2020.

The following are the list of source files relevant to the proper implementation of the project.

### Main Source Files

These are the main, final files for the project.

 - src/slam.py              -> Final SLAM implementation, with joint IMU and Landmark Covariance update step, along with the IMU predict step.
 - src/hw3_main.py          -> The main file provided to students to run the SLAM codes. This has the same contents as src/slam.py
 - src/utils.py             -> Provided source files to load data and display trajectory. I modified it to plot the landmarks also
 - src/imu.py               -> Contains the IMU() class source code to track IMU poses over time, which is their mean inverse pose and the covariance associated with it
 - src/landmark.py          -> Contains the Landmarks() class source code to keep track of the landmark means and covariances over time. Performs EKF update step on the Landmarks in part b) mapping and part c) SLAM

### Experiments Source Files

These are the source files that I developed for different experiments.
These reflect the progress of my experiments from just the dead reckoning for IMU, to the codes for the Mapping, and finally part c) SLAM.
 - src/dead_reckoning.py    -> Dead Reckoning, which is EKF prediction step applied to IMU without noise, nor covariance
 - src/visual_mapping_dr.py -> Implements the mean initialization on landmarks for the EKF update step first part for part b) Mapping
 - src/visual_mapping.py    -> Visual Mapping, which implements the landmark initialization, as well as the EKF update step on the landmarks based on IMU dead reckoning inverse poses.
 - src/predict_mapping.py   -> Implements the complete prediction step for IMU mean and covariance, with the EKF update step for mapping.
 - src/pre_slam.py          -> A step before SLAM, with both IMU update and predict, and the landmarks update, but with separate covariances for IMU and landmarks

### Auxillary Source Files
 - src/speed_test.py     -> Analyzes the speed of scipy.expm() vs Rodrigues formula for matrix exponentiation
 - src/hw3_main_cpy.py      -> A copy of the original file provided to us (IN case I mess up my code)
