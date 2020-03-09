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

    def update(self, z_t):
        self._update_inds(z_t)

