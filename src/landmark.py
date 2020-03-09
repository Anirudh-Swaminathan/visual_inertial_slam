#!/usr/bin/python

# Created by anicodebreaker on March 09, 2020
import numpy as np


class LandMarks(object):
    """
    Class to keep track of the means and the covariances of the landmarks
    """
    def __init__(self, m):
        """
        Constructor to initialize the means, and the covariances of each landmark
        :param m: total number of landmarks
        """
        self.M = m
        self.means = np.zeros((3*self.M, 1))
        self.covs = np.zeros((3*self.M, 3*self.M))

        # boolean array to keep track of all the observed landmarks till time t
        self.observed_inds = np.array([False for _ in range(self.M)])

        # boolean array to keep track of landmarks observed at time t
        self.current_inds = np.array([False for _ in range(self.M)])

        # boolean array of indices to update at time t
        self.cur_upds = np.array([False for _ in range(self.M)])

        # boolean array of indices to initialize mean for at time t
        self.cur_init = np.array([False for _ in range(self.M)])

    def _update_inds(self, z_t):
        """
        Computes all binary indices required for this time step t
        :param z_t: input observations at time t
        :return:
        """
        # only observations to be negative are all -1
        row1 = z_t[0, :]
        self.current_inds = (row1 >= 0)
        self.cur_upds = np.logical_and(self.current_inds, self.observed_inds)
        self.cur_init = np.logical_and(self.current_inds, np.logical_not(self.observed_inds))
        self.observed_inds = np.logical_or(self.current_inds, self.observed_inds)

    def _init_means(self, z_t, c_mat, cam_T_imu, imu_pose):
        """
        Initialize the means for the newly observed landmarks using the inverse of Stereo Camera Observation model
        :param z_t: left and right camera pixel coordinates
        :param c_mat: the calibration matrix of camera
        :param cam_T_imu: pose of IMU in Camera frame
        :return:
        """
        uls = z_t[0, :]
        vls = z_t[1, :]
        urs = z_t[2, :]
        ds = uls - urs
        z = c_mat[2][3] / ds
        x = (uls - (c_mat[0][2])) * z / c_mat[0][0]
        y = (vls - (c_mat[1][2])) * z / c_mat[1][1]
        opt_homo = x.reshape((1, x.size))
        opt_homo = np.dstack((opt_homo, y.reshape((1, y.size))))
        opt_homo = np.dstack((opt_homo, z.reshape((1, z.size))))
        opt_homo = np.dstack((opt_homo, np.ones((1, z.size))))
        imu_frame = np.matmul(np.transpose(cam_T_imu), opt_homo)
        world_frame = np.matmul(imu_pose, imu_frame)
        assert(world_frame.shape[1] == np.sum(self.cur_init))
        mean_2d = np.copy(self.means.reshape((3, self.means.shape[0]/3)))
        mean_2d[:, self.cur_init] = np.copy(world_frame[:3, :])
        self.means = np.copy(mean_2d.reshape(self.means.shape))

    def update(self, z_t, c_mat, cam_T_imu, imu_pose):
        self._update_inds(z_t)
        self._init_means(z_t, c_mat, cam_T_imu, imu_pose)
