# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.transcribe import PitchListWrapper

import amt_tools.tools as tools

import constants
import utils

# Regular imports
import os


class StackedPitchListWrapper():
    """
    TODO
    """


class PitchListWrapper(PitchListWrapper):
    """
    Wrapper for converting continuous multi pitch activation maps to a pitch list.
    """

    def estimate(self, raw_output):
        """
        Convert continuous multi pitch activation maps to a pitch list.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        times : ndarray (N)
          Time in seconds of beginning of each frame
          N - number of time samples (frames)
        pitch_list : list of ndarray (N x [...])
          Array of pitches corresponding to notes
          N - number of pitch observations (frames)
        """

        # Obtain the frame times associated with the activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Obtain the discrete multi pitch activation map
        discrete_multi_pitch = tools.unpack_dict(raw_output, tools.KEY_MULTIPITCH)

        # Obtain the relative multi pitch activation map
        relative_multi_pitch = tools.unpack_dict(raw_output, constants.KEY_MULTIPITCH_REL)

        # Perform the conversion
        pitch_list = utils.continuous_multi_pitch_to_pitch_list(discrete_multi_pitch,
                                                                relative_multi_pitch,
                                                                self.profile)

        # TODO - eventually, this should replace the parent function altogether
        # TODO - if no relative_multi_pitch, just call normal conversion function

        return times, pitch_list
