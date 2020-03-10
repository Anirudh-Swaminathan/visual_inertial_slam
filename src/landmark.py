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
        self.means = np.zeros((3 * self.M, 1))
        self.covs = np.zeros((3 * self.M, 3 * self.M))

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
        pos_inds = np.where(row1 != -1)
        self.current_inds = np.array([False for _ in range(self.M)])
        self.current_inds[pos_inds] = True
        self.cur_upds = np.logical_and(self.current_inds, self.observed_inds)
        self.cur_init = np.logical_and(self.current_inds, np.logical_not(self.observed_inds))
        assert (np.sum(np.logical_and(self.observed_inds, self.cur_init)) == 0)
        self.observed_inds = np.logical_or(self.current_inds, self.observed_inds)

        # DEBUG
        print(self.current_inds)
        print(self.cur_init)
        print(self.observed_inds)

    def _init_means(self, z_t, c_mat, cam_T_imu, imu_pose):
        """
        Initialize the means for the newly observed landmarks using the inverse of Stereo Camera Observation model
        :param z_t: left and right camera pixel coordinates
        :param c_mat: the calibration matrix of camera
        :param cam_T_imu: pose of IMU in Camera frame
        :return:
        """
        print("In in _init_means")
        print(np.sum(self.cur_init), np.sum(self.observed_inds))

        # return if no initializations are required
        if np.sum(self.cur_init) == 0:
            return
        print(z_t.shape)
        uls = z_t[0, self.cur_init]
        vls = z_t[1, self.cur_init]
        urs = z_t[2, self.cur_init]
        ds = uls - urs
        assert(uls.size == np.sum(self.cur_init))
        assert(vls.size == np.sum(self.cur_init))
        assert(ds.size == np.sum(self.cur_init))

        # To avoid division by 0 errors
        # ins = np.where(abs(ds) < 1e-6)
        # ds[ins] = 1e-6
        z = c_mat[2][3] / ds
        x = (uls - (c_mat[0][2])) * z / c_mat[0][0]
        y = (vls - (c_mat[1][2])) * z / c_mat[1][1]
        opt_homo = x.reshape((1, x.size))
        # print("4 OPT HOMO")
        # print(opt_homo.shape)
        opt_homo = np.vstack((opt_homo, y.reshape((1, y.size))))
        # print(opt_homo.shape)
        opt_homo = np.vstack((opt_homo, z.reshape((1, z.size))))
        # print(opt_homo.shape)
        opt_homo = np.vstack((opt_homo, np.ones((1, z.size))))
        # print(opt_homo.shape)
        assert (opt_homo.shape[0] == 4)
        assert (opt_homo.shape[1] == np.sum(self.cur_init))
        assert (cam_T_imu.size == 16)
        imu_frame = np.matmul(np.linalg.inv(cam_T_imu), opt_homo)
        world_frame = np.matmul(imu_pose, imu_frame)
        assert (world_frame.shape[1] == np.sum(self.cur_init))
        mean_2d = np.copy(self.means.reshape((3, self.M), order='F'))
        assert (mean_2d.shape[0] == 3)
        assert (mean_2d.shape[1] == self.M)
        mean_2d[:, self.cur_init] = np.copy(world_frame[:3, :])
        self.means = np.copy(mean_2d.flatten('F').reshape(self.means.shape))

    def update(self, z_t, c_mat, cam_T_imu, imu_pose):
        self._update_inds(z_t)
        self._init_means(z_t, c_mat, cam_T_imu, imu_pose)

    def get_obs_means(self):
        """
        Returns the current mean world frame positions of all landmarks visited so far
        :return:
        """
        mean_2d = np.copy(self.means).reshape((self.M, 3))
        mean_2d = mean_2d.T
        act_means = mean_2d[:, self.observed_inds]
        return act_means

    def get_means(self):
        """
        Returns means for all landmarks for saving
        :return:
        """
        ret = np.copy(self.means.reshape((self.M, 3)).T)
        return ret

    def _save_means(self, pth):
        """
        save the means to specified path
        :param pth: path to save the means to
        :return:
        """
        sv = self.get_means()
        np.save(pth + "means.npy", sv)

    def save_landmarks(self, pth):
        """
        Save the means and the covariances to file
        :param pth: path to save them to
        :return:
        """
        self._save_means(pth)
