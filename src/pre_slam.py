#!/usr/bin/python

# Created by anicodebreaker on March 12, 2020

import numpy as np
from utils import *
from imu import IMU
from landmark import LandMarks as LMS

if __name__ == '__main__':
    filename = "./data/0034.npz"
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

    save_base = "./outputs/pre_slam/d0034/try_001/"
    ani_imu = IMU()

    # recover the dead reckoned landmark points
    dead_map_base = "outputs/visual_mapping_dr/d0034/try_001/landmarks_"
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
        print("Timestep: ", ti)
        # extract this time steps linear and angular velocities
        lin_t = linear_velocity[:, ti]
        ang_t = rotational_velocity[:, ti]

        # construct the control input for this time step
        cont_t = np.concatenate((lin_t, ang_t))
        assert (len(cont_t.shape) == 1)
        assert (cont_t.shape[0] == lin_t.shape[0] + ang_t.shape[0])

        # time discretization
        tau = t[0][ti] - t[0][ti - 1]

        # predict step - IMU
        ani_imu.predict(cont_t, tau)

        # UPDATES!
        # current observations
        z_t = features[:, :, ti]

        # update step - IMU(separate update)
        # get current landmark positions
        lm_means = lms.get_means()
        # ani_imu.update(z_t, c_mat, cam_T_imu, lm_means)

        # obtain current pose
        cur_inv_pose = np.copy(ani_imu.mean)
        cur_pose = ani_imu.get_pose_from_inv(cur_inv_pose)
        assert(np.sum(np.sum(np.abs(cur_pose - np.linalg.inv(cur_inv_pose)))) < 1e-6)

        # UPDATE LandMarks(separate)
        lms.update(z_t, c_mat, cam_T_imu, cur_pose, ani_imu)

        if ti % 500 == 0 or ti == 1:
            # save IMU inverse poses and history
            ani_imu.save_history(save_base + "hist_" + str(ti) + ".npy")
            ani_imu.save_path(save_base + "path_" + str(ti) + ".npy")
            ani_imu.save_cov(save_base + "imu_cov_" + str(ti) + ".npy")
            print("Saved IMU covariance was:-")
            print(ani_imu.cov)
            # save landmark means and covariances
            lms.save_landmarks(save_base + "landmarks_" + str(ti) + "_")

        if ti % 100 == 0 or ti == 1:
            # You can use the function below to visualize the robot pose over time
            world_T_imu = ani_imu.get_path()
            world_T_imu = np.array(world_T_imu)
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
            # visualize_trajectory_2d(world_T_imu, dead_means=dead_map[:, :land_means.shape[1]], landmarks=land_means,
            visualize_trajectory_2d(world_T_imu, landmarks=land_means,
                                    path_name="p_sl_0034", show_ori=True,
                                    save_pth=save_base + "map_img_" + str(ti) + ".png")

    # final save
    # save IMU inverse poses and history
    ani_imu.save_history(save_base + "hist_" + str(t.shape[1]) + ".npy")
    ani_imu.save_path(save_base + "path_" + str(t.shape[1]) + ".npy")
    ani_imu.save_cov(save_base + "imu_cov_" + str(t.shape[1]) + ".npy")
    # save IMU inverse poses and history over all time steps
    lms.save_landmarks(save_base + "landmarks_" + str(t.shape[1]) + "_")

    # Final visualization
    # You can use the function below to visualize the robot pose over time
    world_T_imu = ani_imu.get_path()
    world_T_imu = np.array(world_T_imu)
    old_start = world_T_imu[0, :, :]
    # world_T_imu = world_T_imu.reshape((4, 4, ti + 1))
    world_T_imu = np.moveaxis(world_T_imu, [0, 1, 2], [2, 0, 1])
    new_start = world_T_imu[:, :, 0]
    assert (world_T_imu.shape[0] == 4)
    assert (world_T_imu.shape[1] == 4)
    assert (np.sum(np.abs(old_start - new_start)) <= 1e-6)
    # assert (world_T_imu.shape[2] == t.shape[1])
    land_means = lms.get_obs_means()
    # visualize_trajectory_2d(world_T_imu, dead_means=dead_map, landmarks=land_means, path_name="p_sl_0034", show_ori=True,
    visualize_trajectory_2d(world_T_imu, landmarks=land_means, path_name="p_sl_0034", show_ori=True,
                            save_pth=save_base + "map_img_" + str(t.shape[1]) + ".png")

    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM
