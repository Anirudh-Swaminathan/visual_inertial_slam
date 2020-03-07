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
        self.mean = np.identity(4)
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

    @staticmethod
    def control_hat(control_inp):
        """
        Computes the hat for the control input useful in EKF prediction step for IMU
        :param control_inp: Odometry reading of IMU. It is R^6, with linear velocity followed by angular velocity
        :return: control_hat
        """
        lin_v = control_inp[:3]
        ang_v = control_inp[-3:]

        # angular velocity hat
        ang_v_hat = skew_symm_3D(ang_v)

        # final return matrix
        ret_mat = np.zeros((4, 4))
        ret_mat[:3, :3] = ang_v_hat
        ret_mat[3][:3] = lin_v
        return ret_mat

    def dead_reckon(self, control_inp, time):
        """
        Performs dead reckoning, given the control input
        This means no noise, and NO covariance update
        :param control_inp: Odometry reading of IMU. It is R^6, with linear velocity followed by angular velocity
        :param time: time discretization parameter - tau
        :return:
        """
        self.traj.append(np.copy(self.mean))
        u_hat = self.control_hat(control_inp)
        fin_map = -1.0 * time * u_hat
        exp_map = expm(fin_map)
        self.mean = np.matmul(exp_map, self.mean)

    def get_path(self):
        """
        Return the means of the IMU poses for all time steps so far
        :return:
        """
        ret = list(self.traj)
        ret.append(np.copy(self.mean))
        return ret

    def save_path(self, pth):
        """
        Save the trajectory of the IMU poses over all time steps so far
        :param pth: file path to save the trajectory to
        :return:
        """
        traj = self.get_path()
        t = np.array(traj)
        np.save(pth, t)
