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

        TODO - eventually, this should replace the parent function altogether

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

        if relative_multi_pitch is None:
            # Perform conversion normally if there are no relative pitch estimates
            pitch_list = tools.multi_pitch_to_pitch_list(discrete_multi_pitch, self.profile)
        else:
            # Combine the discrete/relative pitch to obtain continuous multi pitch estimates
            pitch_list = utils.continuous_multi_pitch_to_pitch_list(discrete_multi_pitch,
                                                                    relative_multi_pitch,
                                                                    self.profile)

        return times, pitch_list
