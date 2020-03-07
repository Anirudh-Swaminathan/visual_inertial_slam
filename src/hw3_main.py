import numpy as np
from utils import *


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	print(t.dtype, t.shape, t.min(), t.max())
	diff = t[0][1] - t[0][0]
	print(diff)
	print("Checking if the difference in time stamps is constant")
	for i in range(1, t.shape[1]):
		d = t[0][i] - t[0][i-1]
		print(d)
		assert(np.abs(d - diff) <= 5*1e-2)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
