#!/usr/bin/python

# Created by anicodebreaker on March 10, 2020

import numpy as np
from utils import *
from imu import IMU
from landmark import LandMarks as LMS

if __name__ == '__main__':
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    print(features.shape)
    print(K.shape)
    print(b)
    print(cam_T_imu.shape)

    # Extract every 10th feature for faster computation
    ori_num_feats = features.shape[1]
    new_inds = np.arange(0, ori_num_feats, 10)
    new_feats = features[:, new_inds, :]
    features = np.copy(new_feats)

    # COMPUTE the Stereo Camera Calibration matrix
    c_mat = np.zeros((3, 4))
    c_mat[:, :3] = np.copy(K)
    c_mat[2][2] = 0
    c_mat[2][3] = K[0][0] * b
    print(K)
    print(c_mat)

    save_base = "./outputs/visual_mapping/d0027/try_005/"

    imu_base = "outputs/dead_reckoning/d0027/try_001/"
    imu_poses = np.load(imu_base + "path_" + str(t.shape[1]) + ".npy")
    imu_inv_poses = np.load(imu_base + "hist_" + str(t.shape[1]) + ".npy")
    print(imu_poses.shape)

    # recover the dead reckoned landmark points
    dead_map_base = "outputs/visual_mapping_dr/d0027/try_001/landmarks_"
    dead_map = np.load(dead_map_base + str(t.shape[1]) + "_means.npy")
    dead_map = dead_map[:, new_inds]
    print(dead_map.shape)
    assert(dead_map.shape[1] == features.shape[1])

    # instantiate landmarks
    # M - Number of landmarks
    M = features.shape[1]
    lms = LMS(M)

    # check if the only negative value is -1
    assert (len(np.unique(features[features < 0])) == 1)

    # loop through all time steps
    for ti in range(1, t.shape[1]):
        # time discretization
        tau = t[0][ti] - t[0][ti - 1]

        # current IMU inverse pose and pose
        cur_inv_pose = imu_inv_poses[ti, :, :]
        cur_pose = imu_poses[ti, :, :]

        # current observations
        z_t = features[:, :, ti]
        lms.update(z_t, c_mat, cam_T_imu, cur_pose)

        if ti % 500 == 0 or ti == 1:
            # save landmark means and covariances
            lms.save_landmarks(save_base + "landmarks_" + str(ti) + "_")

        if ti % 100 == 0 or ti == 1:
            # You can use the function below to visualize the robot pose over time
            world_T_imu = imu_poses[:ti, :, :]
            old_start = world_T_imu[0, :, :]
            # world_T_imu = world_T_imu.reshape((4, 4, ti + 1))
            world_T_imu = np.moveaxis(world_T_imu, [0, 1, 2], [2, 0, 1])
            new_start = world_T_imu[:, :, 0]
            # print(world_T_imu.shape)
            # print(old_start)
            # print(new_start)
            assert (world_T_imu.shape[0] == 4)
            assert (world_T_imu.shape[1] == 4)
            assert (np.sum(np.abs(old_start - new_start)) <= 1e-6)
            # assert (world_T_imu.shape[2] == t.shape[1])
            land_means = lms.get_obs_means()
            land_covs = lms.get_obs_covs()
            print("So far observed landmarks!!")
            print("Covariances are shaped as ", land_covs.shape)
            # print(land_covs)
            visualize_trajectory_2d(world_T_imu, landmarks=land_means,
                                    path_name="mp_0027", show_ori=True,
                                    save_pth=save_base + "map_img_" + str(ti) + ".png")

    # final save
    # save IMU inverse poses and history over all time steps
    lms.save_landmarks(save_base + "landmarks_" + str(t.shape[1]) + "_")

    # Final visualization
    # You can use the function below to visualize the robot pose over time
    world_T_imu = imu_poses[:, :, :]
    old_start = world_T_imu[0, :, :]
    # world_T_imu = world_T_imu.reshape((4, 4, ti + 1))
    world_T_imu = np.moveaxis(world_T_imu, [0, 1, 2], [2, 0, 1])
    new_start = world_T_imu[:, :, 0]
    assert (world_T_imu.shape[0] == 4)
    assert (world_T_imu.shape[1] == 4)
    assert (np.sum(np.abs(old_start - new_start)) <= 1e-6)
    # assert (world_T_imu.shape[2] == t.shape[1])
    land_means = lms.get_obs_means()
    visualize_trajectory_2d(world_T_imu, landmarks=land_means, path_name="mp_0027", show_ori=True,
                            save_pth=save_base + "map_img_" + str(t.shape[1]) + ".png")

    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM
