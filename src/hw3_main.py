import numpy as np
from utils import *
from imu import IMU

if __name__ == '__main__':
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # print(linear_velocity.shape)
    # print(rotational_velocity.shape)

    # instantiate my IMU class
    ani_imu = IMU()
    save_base = "./outputs/dead_reckoning/d0027/try_002/"

    # loop through all time steps
    for ti in range(1, t.shape[1]):
        # extract this time steps linear and angular velocities
        lin_t = linear_velocity[:, ti]
        ang_t = rotational_velocity[:, ti]

        # construct the control input for this time step
        cont_t = np.concatenate((lin_t, ang_t))
        assert (len(cont_t.shape) == 1)
        assert (cont_t.shape[0] == lin_t.shape[0] + ang_t.shape[0])

        # time discretization
        tau = t[0][ti] - t[0][ti - 1]
        ani_imu.dead_reckon(cont_t, tau)

        if ti % 500 == 0:
            # save IMU inverse poses and history
            ani_imu.save_history(save_base + "hist_" + str(ti) + ".npy")
            ani_imu.save_path(save_base + "path_" + str(ti) + ".npy")

        if ti % 100 == 0:
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
            assert(np.sum(np.abs(old_start - new_start)) <= 1e-6)
            # assert (world_T_imu.shape[2] == t.shape[1])
            visualize_trajectory_2d(world_T_imu, path_name="dr_0027", show_ori=True,
                                    save_pth=save_base + "path_img_" + str(ti) + ".png")

    # final save
    # save IMU inverse poses and history over all time steps
    ani_imu.save_history(save_base + "hist_" + str(t.shape[1]) + ".npy")
    ani_imu.save_path(save_base + "path_" + str(t.shape[1]) + ".npy")

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
    visualize_trajectory_2d(world_T_imu, path_name="dr_0027", show_ori=True,
                            save_pth=save_base + "path_img_" + str(t.shape[1]) + ".png")

    # (a) IMU Localization via EKF Prediction

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM
