# ECE276A Project 3 - Visual Intertial SLAM

Project implemented by Anirudh Swaminathan - A53316083 - UCSD - Winter 2020.

The following are the list of source files relevant to the proper implementation of the project.

 - src/hw3_main.py       -> The main file provided to students to run the SLAM codes
 - src/utils.py          -> Provided source files to load data and display trajectory
 - src/dead_reckoning.py -> Dead Reckoning -> EKF prediction step applied to IMU without noise
 - src/imu.py            -> Contains the IMU() class source code to track IMU poses over time
 - src/hw3_main_cpy.py   -> A copy of the original file provided to us (IN case I mess up my code)
 - src/speed_test.py     -> Analyzes the speed of scipy.expm() vs Rodrigues formula for matrix exponentiation
