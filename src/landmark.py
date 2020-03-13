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
        # self.covs = np.identity(3*self.M)

        # boolean array to keep track of all the observed landmarks till time t
        self.observed_inds = np.array([False for _ in range(self.M)])

        # boolean array to keep track of landmarks observed at time t
        self.current_inds = np.array([False for _ in range(self.M)])

        # boolean array of indices to update at time t
        self.cur_upds = np.array([False for _ in range(self.M)])

        # boolean array of indices to initialize mean for at time t
        self.cur_init = np.array([False for _ in range(self.M)])

        # the observation model noise
        self.V = np.identity(4)

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
        # print(self.current_inds)
        # print(self.cur_init)
        # print(self.observed_inds)

    def _init_landmarks(self, z_t, c_mat, cam_T_imu, imu_pose):
        """
        Initialize the means for the newly observed landmarks using the inverse of Stereo Camera Observation model
        :param z_t: left and right camera pixel coordinates
        :param c_mat: the calibration matrix of camera
        :param cam_T_imu: pose of IMU in Camera frame
        :return:
        """
        # print("In _init_means")
        num_new = np.sum(self.cur_init)
        # print(num_new, np.sum(self.observed_inds), np.sum(self.current_inds))

        # return if no initializations are required
        if num_new == 0:
            return
        # print(z_t.shape)
        uls = z_t[0, self.cur_init]
        vls = z_t[1, self.cur_init]
        urs = z_t[2, self.cur_init]
        ds = uls - urs
        assert (uls.size == num_new)
        assert (vls.size == num_new)
        assert (ds.size == num_new)

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
        assert (opt_homo.shape[1] == num_new)
        assert (cam_T_imu.size == 16)
        imu_frame = np.matmul(np.linalg.inv(cam_T_imu), opt_homo)
        world_frame = np.matmul(imu_pose, imu_frame)
        assert (world_frame.shape[1] == num_new)
        mean_2d = np.copy(self.means.reshape((3, self.M), order='F'))
        assert (mean_2d.shape[0] == 3)
        assert (mean_2d.shape[1] == self.M)
        mean_2d[:, self.cur_init] = np.copy(world_frame[:3, :])
        self.means = np.copy(mean_2d.flatten('F').reshape(self.means.shape))

        # Initialize the covariances of these landmarks
        cov_init = np.identity(3 * num_new) * 1e-2
        intinds = np.where(self.cur_init == True)[0]
        # print(intinds)
        for i in range(num_new):
            self.covs[intinds[i] * 3: (intinds[i] + 1) * 3, intinds[i] * 3: (intinds[i] + 1) * 3] \
                = np.copy(cov_init[i * 3: (i + 1) * 3, i * 3:(i + 1) * 3])

    def _projective_derivative(self, q):
        """
        Returns the matrix representing the differentiation of the projection function
        :param q: the homogeneous coordinate to take the derivative for
        :return:
        """
        ret = np.identity(4)
        ret[:, 2] -= 1.0 * q / q[2]
        return 1.0/q[2] * ret

    def _update_landmarks(self, z_t, c_mat, cam_T_imu, imu_pose, Imu):
        """
        Function to update the already observed landmarks in this time step
        :param z_t: Current time observations
        :param c_mat: Stereo Camera disparity matrix
        :param cam_T_imu: Pose of IMU in camera frame
        :param imu_pose: Pose of IMU in world frame
        :return:
        """
        # print("In Update landmarks!")
        num_upds = np.sum(self.cur_upds)
        # print(num_upds, np.sum(self.observed_inds), np.sum(self.current_inds))
        if num_upds == 0:
            return
        # Construct the required M - 4*4
        M = np.zeros((4, 4))
        M[:2, :] = c_mat[:2, :]
        M[2:, :] = c_mat[:2, :]
        M[2, 3] = -1.0 * c_mat[2, 3]

        # compute Ut - 4*4
        Ut = np.linalg.inv(imu_pose)
        assert(M.shape == Ut.shape)

        # Projection - 3*4
        P = np.zeros((3, 4))
        P[:, :3] = np.identity(3)

        # TUt - 4*4
        tut = np.matmul(cam_T_imu, Ut)
        assert(tut.shape == M.shape)

        # reshape the means
        # mut - 3*M
        mut = self.means.reshape((3, self.M), order='F')
        ones = np.ones((1, self.M))
        # mubt - 4*M
        mubt = np.vstack((mut, ones))
        assert(mubt.shape[0] == 4)
        assert(mubt.shape[1] == self.M)

        # compute the q inside the Jacobian
        # q - 4*M
        q = np.matmul(tut, mubt)
        assert(q.shape == (4, self.M))
        pq = q / q[2, :]
        assert(pq.shape == (4, self.M))
        z_tbar = np.matmul(M, pq)
        assert(z_tbar.shape == (4, self.M))
        assert(np.sum(np.abs(z_tbar[1, :] - z_tbar[3, :])) < 1e-6)

        Imu.update_separate(z_t[:, self.current_inds].flatten('F'), z_tbar[:, self.current_inds].flatten('F'),
                        M, cam_T_imu, mubt[:, self.current_inds])

        # set up Jacobian - 4Nt * 3M
        Nt = np.sum(self.current_inds)
        Ht = np.zeros((4*Nt, 3*self.M))
        inds = np.where(self.cur_upds == True)[0]
        # print(inds)
        for i in range(len(inds)):
            dpq = self._projective_derivative(q[:, inds[i]])
            assert(dpq.shape == M.shape)
            r = np.matmul(tut, np.transpose(P))
            mid = np.matmul(dpq, r)
            htii = np.matmul(M, mid)
            Ht[i*4:(i+1)*4, inds[i]*3:(inds[i]+1)*3] = np.copy(htii)

        # Set up the covariance as a block diagonal
        IV = np.identity(4*Nt)
        sigmaH = np.matmul(self.covs, np.transpose(Ht))
        assert(sigmaH.shape[0] == 3*self.M)
        assert(sigmaH.shape[1] == 4*Nt)
        hsh = np.matmul(Ht, sigmaH)
        assert(hsh.shape == (4*Nt, 4*Nt))
        inside_inv = hsh + IV
        assert(IV.shape == (4*Nt, 4*Nt))
        assert(inside_inv.shape == (4*Nt, 4*Nt))
        ro = np.linalg.inv(inside_inv)
        mido = np.matmul(np.transpose(Ht), ro)
        assert(mido.shape == (3*self.M, 4*Nt))
        Kt = np.matmul(self.covs, mido)
        assert(Kt.shape == (3*self.M, 4*Nt))

        # mean update
        flat_zt = z_t[:, self.current_inds].flatten('F')
        flat_zt_bar = z_tbar[:, self.current_inds].flatten('F')
        # print(z_t.shape)
        # print(z_tbar.shape)
        # print(flat_zt.shape)
        # print(flat_zt_bar.shape)
        fzt = flat_zt.reshape(flat_zt.size, 1)
        fztb = flat_zt_bar.reshape(flat_zt_bar.size, 1)
        assert(fzt.shape == (4*Nt, 1))
        assert(fztb.shape == (4*Nt, 1))
        adt = np.matmul(Kt, (fzt - fztb))
        assert(adt.shape == (3*self.M, 1))
        self.means += np.copy(adt)

        # covariance update
        ktht = np.matmul(Kt, Ht)
        assert(ktht.shape == (3*self.M, 3*self.M))
        I = np.identity(3*self.M)
        self.covs = np.matmul((I - ktht), self.covs)

    def update(self, z_t, c_mat, cam_T_imu, imu_pose, Imu):
        self._update_inds(z_t)
        self._init_landmarks(z_t, c_mat, cam_T_imu, imu_pose)
        self._update_landmarks(z_t, c_mat, cam_T_imu, imu_pose, Imu)

    def get_obs_means(self):
        """
        Returns the current mean world frame positions of all landmarks visited so far
        :return:
        """
        mean_2d = np.copy(self.means).reshape((self.M, 3))
        mean_2d = mean_2d.T
        act_means = mean_2d[:, self.observed_inds]
        return act_means

    def get_obs_covs(self):
        """
        Returns covariances of all observed landmarks so far
        :return:
        """
        cov = np.copy(self.covs)
        ret_inds = np.where(self.observed_inds == True)[0]
        K = np.sum(self.observed_inds)
        ret_mat = np.zeros((K * 3, K * 3))
        for i in range(K):
            ret_mat[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = cov[ret_inds[i] * 3:(ret_inds[i] + 1) * 3,
                                                            ret_inds[i] * 3:(ret_inds[i] + 1) * 3]
        return ret_mat

    def get_means(self):
        """
        Returns means for all landmarks for saving
        :return:
        """
        ret = np.copy(self.means.reshape((self.M, 3)).T)
        return ret

    def get_covs(self):
        """
        Returns the covariances of all landmarks for saving
        :return:
        """
        ret = np.copy(self.covs)
        return ret

    def _save_means(self, pth):
        """
        save the means to specified path
        :param pth: path to save the means to
        :return:
        """
        sv = self.get_means()
        np.save(pth + "means.npy", sv)

    def _save_covs(self, pth):
        """
        save the covariances of all landmarks to specified path
        :param pth: path to save the covariances to
        :return:
        """
        sv = self.get_covs()
        np.save(pth + "covs.npy", sv)

    def save_landmarks(self, pth):
        """
        Save the means and the covariances to file
        :param pth: path to save them to
        :return:
        """
        self._save_means(pth)
        self._save_covs(pth)
