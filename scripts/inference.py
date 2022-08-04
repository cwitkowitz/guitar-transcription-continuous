# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.transcribe import StackedPitchListWrapper as _StackedPitchListWrapper
from amt_tools.transcribe import PitchListWrapper as _PitchListWrapper

import amt_tools.tools as tools

import constants
import utils

# Regular imports
import os


class StackedPitchListWrapper(_StackedPitchListWrapper):
    """
    Wrapper for converting stacked continuous multi pitch activation maps to a stacked pitch list.
    """

    def estimate(self, raw_output):
        """
        Convert stacked continuous multi pitch activation maps to a stacked pitch list.

        TODO - eventually, this should replace the parent function altogether

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_pitch_list : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        """

        # Obtain the discrete multi pitch activation map
        stacked_discrete_multi_pitch = tools.unpack_dict(raw_output, tools.KEY_MULTIPITCH)

        # Obtain the relative multi pitch activation map
        stacked_relative_multi_pitch = tools.unpack_dict(raw_output, constants.KEY_MULTIPITCH_REL)

        # Obtain the frame times associated with the activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        if stacked_relative_multi_pitch is None:
            # Perform conversion normally if there are no relative pitch estimates
            stacked_pitch_list = \
                tools.stacked_multi_pitch_to_stacked_pitch_list(stacked_discrete_multi_pitch, times, self.profile)
        else:
            # Combine the discrete/relative pitch to obtain continuous multi pitch estimates
            stacked_pitch_list = \
                utils.stacked_continuous_multi_pitch_to_stacked_pitch_list(stacked_discrete_multi_pitch,
                                                                           stacked_relative_multi_pitch,
                                                                           times, self.profile)

        # TODO - remove this after string-level evaluation is in place
        stacked_pitch_list = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

        return stacked_pitch_list


class PitchListWrapper(_PitchListWrapper):
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

        # Obtain the discrete multi pitch activation map
        discrete_multi_pitch = tools.unpack_dict(raw_output, tools.KEY_MULTIPITCH)

        # Obtain the relative multi pitch activation map
        relative_multi_pitch = tools.unpack_dict(raw_output, constants.KEY_MULTIPITCH_REL)

        # Obtain the frame times associated with the activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        if relative_multi_pitch is None:
            # Perform conversion normally if there are no relative pitch estimates
            pitch_list = tools.multi_pitch_to_pitch_list(discrete_multi_pitch, self.profile)
        else:
            # Combine the discrete/relative pitch to obtain continuous multi pitch estimates
            pitch_list = utils.continuous_multi_pitch_to_pitch_list(discrete_multi_pitch,
                                                                    relative_multi_pitch,
                                                                    self.profile)

        return times, pitch_list


class StackedPitchListTablatureWrapper(StackedPitchListWrapper):
    """
    Wrapper for converting continuous tablature activation maps to a stacked pitch list.
    """

    def pre_proc(self, raw_output):
        """
        Convert continuous tablature activation maps to stacked multi pitch representation.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing discrete/relative tablature activations

        Returns
        ----------
        raw_output : dict
          Dictionary containing discrete/relative stacked multi pitch activations
        """

        # Perform the parent's pre-processing steps
        raw_output = super().pre_proc(raw_output)

        # Obtain a stacked multi pitch array representation of the tablature
        raw_output[tools.KEY_MULTIPITCH] = \
            tools.tablature_to_stacked_multi_pitch(raw_output[tools.KEY_TABLATURE], self.profile)

        # Obtain a stacked multi pitch array representation of the relative pitch
        raw_output[constants.KEY_MULTIPITCH_REL] = \
            tools.logistic_to_stacked_multi_pitch(raw_output[constants.KEY_TABLATURE_REL], self.profile, False)

        return raw_output
