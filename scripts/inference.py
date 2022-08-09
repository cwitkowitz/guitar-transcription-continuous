# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.transcribe import StackedPitchListWrapper as _StackedPitchListWrapper

import amt_tools.tools as tools

import constants
import utils

# Regular imports
import os


class StackedPitchListWrapper(_StackedPitchListWrapper):
    """
    Wrapper for converting stacked continuous multi pitch activation maps to a stacked pitch list.

    TODO - eventually, this should replace the parent class altogether
    """

    def __init__(self, profile, multi_pitch_key=None, multi_pitch_rel_key=None, estimates_key=None, save_dir=None):
        """
        Initialize parameters for the estimator.

        Parameters
        ----------
        See Estimator class...

        multi_pitch_rel_key : string or None (optional)
          Key to use when unpacking relative multi pitch data
        """

        super().__init__(profile=profile,
                         multi_pitch_key=multi_pitch_key,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.multi_pitch_rel_key = constants.KEY_MULTIPITCH_REL if multi_pitch_rel_key is None else multi_pitch_rel_key

    def estimate(self, raw_output):
        """
        Convert stacked continuous multi pitch activation maps to a stacked pitch list.

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
        stacked_discrete_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Obtain the relative multi pitch activation map
        stacked_relative_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_rel_key)

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

        return stacked_pitch_list


class PitchListWrapper(StackedPitchListWrapper):
    """
    Wrapper for converting continuous multi pitch activation maps to a pitch list.

    TODO - eventually, this should replace the parent class altogether
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

        # Obtain the discrete multi pitch activation map
        discrete_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Obtain the relative multi pitch activation map
        relative_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_rel_key)

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

    def write(self, pitch_list, track):
        """
        Write pitch estimates to a file.

        Parameters
        ----------
        pitch_list : tuple containing
          times : ndarray (N)
            Time in seconds of beginning of each frame
            N - number of time samples (frames)
          pitch_list : list of ndarray (N x [...])
            Array of pitches corresponding to notes
            N - number of pitch observations (frames)
        track : string
          Name of the track being processed
        """

        # Stack the pitch list
        stacked_pitch_list = tools.pitch_list_to_stacked_pitch_list(*pitch_list)

        # Call the parent function
        super().write(stacked_pitch_list, track)


class StackedPitchListTablatureWrapper(StackedPitchListWrapper):
    """
    Wrapper for converting continuous tablature activation maps to a stacked pitch list.

    TODO - could this should replace TablatureWrapper eventually?
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
        raw_output[self.multi_pitch_key] = \
            tools.tablature_to_stacked_multi_pitch(raw_output[self.multi_pitch_key], self.profile)

        # Obtain a stacked multi pitch array representation of the relative pitch
        raw_output[self.multi_pitch_rel_key] = \
            tools.logistic_to_stacked_multi_pitch(raw_output[self.multi_pitch_rel_key], self.profile, False)

        return raw_output
