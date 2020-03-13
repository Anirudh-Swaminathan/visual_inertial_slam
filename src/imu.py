#!/usr/bin/python

import numpy as np
from scipy.linalg import expm


class IMU(object):
    """
    IMU Class - Contains the mean, covariance and the Inverse Pose of the IMU Matrix
    """

    def __init__(self):
        """
        Constructor for IMU Class
        """
        # initialize the mean, covariance and the noise
        self.mean = np.identity(4)
        self.cov = 0.1 * np.identity(6)
        self.W = 1e-4 * np.identity(6)

        # history of IMU Inverse means
        self.inv_history = []

        # IMU Mean poses over time
        self.traj = []

    @staticmethod
    def skew_symm_3D(vec_3d):
        """
        Computes the hat map for a 3 dimensional vector
        :param vec_3d: input vector to compute the hat map for
        :return:
        """
        # row 1
        ret_mat = np.zeros((3, 3))
        ret_mat[0][1] = -1.0 * vec_3d[2]
        ret_mat[0][2] = vec_3d[1]

        # row 2
        ret_mat[1][0] = vec_3d[2]
        ret_mat[1][2] = -1.0 * vec_3d[0]

        # row 3
        ret_mat[2][0] = -1.0 * vec_3d[1]
        ret_mat[2][1] = vec_3d[0]
        return ret_mat

    def control_hat(self, control_inp):
        """
        Computes the hat for the control input useful in EKF prediction step for IMU
        :param control_inp: Odometry reading of IMU. It is R^6, with linear velocity followed by angular velocity
        :return: control_hat
        """
        lin_v = control_inp[:3]
        ang_v = control_inp[-3:]

        # angular velocity hat
        ang_v_hat = self.skew_symm_3D(ang_v)

        # final return matrix
        ret_mat = np.zeros((4, 4))
        ret_mat[:3, :3] = np.copy(ang_v_hat)
        ret_mat[:3, 3] = np.copy(lin_v)
        return ret_mat

    def curly_hat(self, control_inp):
        """
        Computes the curly hat for the control input, useful in EKF prediction step for IMU
        :param control_inp: Odometry reading of IMU. It is R^6, with linear velocity followed by angular velocity
        :return: control_curly
        """
        lin_v = control_inp[:3]
        ang_v = control_inp[-3:]

        # angular velocity hat
        ang_v_hat = self.skew_symm_3D(ang_v)

        # linear velocity hat
        lin_v_hat = self.skew_symm_3D(lin_v)

        # final return matrix
        ret_mat = np.zeros((6, 6))
        # block row 1
        ret_mat[:3, :3] = np.copy(ang_v_hat)
        ret_mat[:3, 3:] = np.copy(lin_v_hat)
        # block row 2
        ret_mat[3:, 3:] = np.copy(ang_v_hat)
        return ret_mat

    @staticmethod
    def get_pose_from_inv(inv_pose):
        """
        Invert the Inverse Pose U of IMU to get T at time t
        :return:
        """
        # print(inv_pose.shape)
        ret_mat = np.zeros((4, 4))
        R = inv_pose[:3, :3]
        Rt = np.transpose(R)
        # print(inv_pose[:3, 3].shape)
        p = inv_pose[:3, 3]
        p = p.reshape((3, 1))
        p_n = -1.0 * np.matmul(Rt, p)
        ret_mat[:3, :3] = np.copy(Rt)
        ret_mat[:3, 3] = np.copy(p_n.flatten())
        ret_mat[3, 3] = 1
        return ret_mat

    def dead_reckon(self, control_inp, time):
        """
        Performs dead reckoning, given the control input
        This means no noise, and NO covariance update
        :param control_inp: Odometry reading of IMU. It is R^6, with linear velocity followed by angular velocity
        :param time: time discretization parameter - tau
        :return:
        """
        self.inv_history.append(np.copy(self.mean))
        mp = self.get_pose_from_inv(self.mean)
        self.traj.append(np.copy(mp))
        u_hat = self.control_hat(control_inp)
        fin_map = -1.0 * time * u_hat
        exp_map = expm(fin_map)
        self.mean = np.matmul(exp_map, self.mean)

    def predict(self, control_inp, time):
        """
        Performs the EKF predict step on the IMU
        :param control_inp: the control input as linear velocity and angular velocity
        :param time: the time discretization
        :return:
        """
        # print("Previous mean and cov")
        # print(self.mean)
        # print(self.cov)
        self.inv_history.append(np.copy(self.mean))
        mp = self.get_pose_from_inv(self.mean)
        self.traj.append(np.copy(mp))

        # mean update
        u_hat = self.control_hat(control_inp)
        f_map = -1.0 * time * u_hat
        e_map = expm(f_map)
        self.mean = np.matmul(e_map, self.mean)

        # covariance update
        u_curly = self.curly_hat(control_inp)
        fin_map = -1.0 * time * u_curly
        exp_map = expm(fin_map)
        rit = np.matmul(self.cov, np.transpose(exp_map))
        term1 = np.matmul(exp_map, rit)
        assert(term1.shape == self.W.shape)
        self.cov = np.copy(term1 + self.W)
        # print("Updated Mean and cov")
        # print(self.mean)
        # print(self.cov)

    def _projective_derivative(self, q):
        """
        Returns the matrix representing the differentiation of the projection function
        :param q: the homogeneous coordinate to take the derivative for
        :return:
        """
        ret = np.identity(4)
        ret[:, 2] -= 1.0 * q / q[2]
        return 1.0/q[2] * ret

    def _con_dot(self, s):
        """
        Computes the Concetric Circle dot as described in lecture 13, slide 18
        :param p:
        :return:
        """
        s = s.flatten()
        s3 = s[:3]
        s_hat = self.skew_symm_3D(s3)
        ret_mat = np.zeros((4, 6))
        ret_mat[:3, :3] = np.identity(3)
        ret_mat[:3, 3:] = np.copy(-1.0 * s_hat)
        return ret_mat

    def get_jacobian(self, cam_T_imu, landmarks, M):
        """
        Returns the Jacobian of the IMU wrt each observation
        :param cam_T_imu:
        :param landmarks:
        :param M:
        :return:
        """
        # Set up the Jacobian
        Nt = landmarks.shape[1]
        Ht = np.zeros((4 * Nt, 6))

        # TUt - 4*4
        tut = np.matmul(cam_T_imu, self.mean)
        assert (tut.shape == M.shape)

        # inside the projective derivative
        q = np.matmul(tut, landmarks)

        # inside the concentric circle dot
        in_dot = np.matmul(self.mean, landmarks)

        # loop through all the observed landmarks at time t
        for i in range(landmarks.shape[1]):
            dpq = self._projective_derivative(q[:, i])
            assert (dpq.shape == M.shape)
            r_dot = self._con_dot(in_dot[:, i])
            r = np.matmul(cam_T_imu, r_dot)
            mid = np.matmul(dpq, r)
            htii = np.matmul(M, mid)
            Ht[i*4:(i+1)*4, :] = np.copy(htii)
        assert(Ht.shape == (4*Nt, 6))
        return Ht

    def ekf_mean_update(self, adt):
        """
        Performs the update of the IMU mean using EKF update equations
        :param Kt: the 6*Nt Kalman Gain matrix
        :param fzt: the first term in innovation
        :param fztb: the second term in innovation
        :return:
        """
        # Mean update
        assert (adt.shape == (6, 1))
        adt = adt.flatten()
        hat_kz = self.control_hat(adt)
        exp_kx = expm(hat_kz)
        self.mean = np.matmul(exp_kx, self.mean)

    def update_separate(self, z_t, z_tbar, M, cam_T_imu, landmarks):
        """
        Performs the EKF update step on the IMU mean and Covariance using the input observations and landmarks separately
        :param z_t: Observations at current time step
        :param z_tbar: Computed predicted observation - useful for the Innovation term
        :param M: the stereo camera calibration matrix
        :param cam_T_imu: transforms IMU frame to the camera frame, i.e, IMU to optical frame
        :param landmarks: the current landmark means to use in the EKF update step
        :return:
        """
        Nt = landmarks.shape[1]
        fzt = z_t.reshape(z_t.size, 1)
        fztb = z_tbar.reshape(z_tbar.size, 1)

        # Jacobian wrt IMU
        Ht = self.get_jacobian(cam_T_imu, landmarks, M)

        # Kalman Gain
        # Set up the covariance as a block diagonal Noise
        IV = np.identity(4 * Nt)
        sigmaH = np.matmul(self.cov, np.transpose(Ht))
        assert (sigmaH.shape[0] == 6)
        assert (sigmaH.shape[1] == 4 * Nt)
        hsh = np.matmul(Ht, sigmaH)
        assert (hsh.shape == (4 * Nt, 4 * Nt))
        inside_inv = hsh + IV
        assert (IV.shape == (4 * Nt, 4 * Nt))
        assert (inside_inv.shape == (4 * Nt, 4 * Nt))
        ro = np.linalg.inv(inside_inv)
        mido = np.matmul(np.transpose(Ht), ro)
        assert (mido.shape == (6, 4 * Nt))
        Kt = np.matmul(self.cov, mido)
        assert (Kt.shape == (6, 4 * Nt))

        # Mean update
        adt = np.matmul(Kt, (fzt - fztb))
        assert(adt.shape == (6, 1))
        adt = adt.flatten()
        hat_kz = self.control_hat(adt)
        exp_kx = expm(hat_kz)
        self.mean = np.matmul(exp_kx, self.mean)

        # Covariance Update
        ktht = np.matmul(Kt, Ht)
        assert (ktht.shape == (6, 6))
        I = np.identity(6)
        self.cov = np.matmul((I - ktht), self.cov)

    def get_cov(self):
        """
        Returns the current IMU 6*6 covariance matrix
        :return:
        """
        return np.copy(self.cov)

    def set_cov(self, cov):
        """
        Set the IMU covariance after the EKF update step, used for the predict step here
        :param cov: the covariance value to set the IMU covariance to
        :return:
        """
        self.cov = np.copy(cov)

    def get_history(self):
        """
        return the history of IMU Inverse Poses
        :return:
        """
        ret = list(self.inv_history)
        ret.append(np.copy(self.mean))
        return ret

    def get_path(self):
        """
        Return the means of the IMU poses for all time steps so far
        :return:
        """
        ret = list(self.traj)
        # append mean pose
        mean_pose = self.get_pose_from_inv(self.mean)
        ret.append(np.copy(mean_pose))
        return ret

    def save_history(self, pth):
        """
        Save tht IMU inverse poses history over all time steps so far
        :param pth: file path to save the history to
        :return:
        """
        hist = self.get_history()
        t = np.array(hist)
        np.save(pth, t)

    def save_path(self, pth):
        """
        Save the trajectory of the IMU poses over all time steps so far
        :param pth: file path to save the trajectory to
        :return:
        """
        traj = self.get_path()
        t = np.array(traj)
        np.save(pth, t)

    def save_cov(self, pth):
        """
        Save the covariance of the inverse IMU pose at time t
        :param pth: file path to save the covariance to
        :return:
        """
        np.save(pth, self.cov)
