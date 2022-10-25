# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.transcribe import StackedPitchListWrapper as _StackedPitchListWrapper

import guitar_transcription_continuous.utils as utils

import amt_tools.tools as tools

# Regular imports
import numpy as np


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
        self.multi_pitch_rel_key = utils.KEY_MULTIPITCH_REL if multi_pitch_rel_key is None else multi_pitch_rel_key

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
         - could this not just extend StackedPitchListWrapper above?
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

        if tools.query_dict(raw_output, self.multi_pitch_rel_key):
            # Obtain a stacked multi pitch array representation of the relative pitch
            raw_output[self.multi_pitch_rel_key] = \
                tools.logistic_to_stacked_multi_pitch(raw_output[self.multi_pitch_rel_key], self.profile, False)

        return raw_output


class TablatureStreamer(StackedPitchListTablatureWrapper):
    """
    TODO
    """

    def __init__(self, profile, notes_key=None, multi_pitch_key=None,
                 multi_pitch_rel_key=None, estimates_key=None, save_dir=None):
        """
        TODO
        """

        super().__init__(profile=profile,
                         multi_pitch_key=multi_pitch_key,
                         multi_pitch_rel_key=multi_pitch_rel_key,
                         estimates_key=estimates_key,
                         save_dir=save_dir)

        # Default the key for unpacking relevant data
        self.notes_key = tools.KEY_NOTES if notes_key is None else notes_key

    @staticmethod
    def get_default_key():
        """
        Default key for note-contour grouping.
        """

        return utils.KEY_GROUPING

    def estimate(self, raw_output):
        """
        TODO
        """

        # Obtain the discrete multi pitch activation map
        stacked_discrete_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Obtain the relative multi pitch activation map
        stacked_relative_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_rel_key)

        if stacked_relative_multi_pitch is None:
            # Default the relative pitch activations to all zeros
            stacked_relative_multi_pitch = np.zeros(stacked_discrete_multi_pitch.shape)

        # Obtain the stacked note estimates
        stacked_notes = tools.unpack_dict(raw_output, self.notes_key)

        # Obtain the frame times associated with the activation map
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Determine the number of slices
        stack_size = len(stacked_notes)

        # Initialize an empty grouping for each slice of notes in the stack
        # TODO - make this more elegant
        stacked_grouping = {0 : {}, 1 : {}, 2 : {}, 3 : {}, 4 : {}, 5 : {}}

        # Keep track of amount of pre-existing notes for unique index
        num_offset = 0

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Loop through all note predictions in the slice
            for i, (pitch, interval) in enumerate(zip(*stacked_notes[slc])):
                # Add back the collapsed dimension
                pitch, interval = np.array([pitch]), np.array([interval])
                # Obtain an independent multi pitch array for each note
                note_multi_pitch = tools.notes_to_multi_pitch(pitch, interval, times, self.profile)
                # Determine which activations correspond to the note
                pitch_idcs, time_idcs = np.where(note_multi_pitch)
                # Extract the relative activations for the note
                relative_activations = stacked_relative_multi_pitch[slc, pitch_idcs, time_idcs]
                # Compute the continuous pitches associated with the note
                continuous_pitches = self.profile.low + pitch_idcs + relative_activations
                # Create a PitchContour for the observations and add an entry to the grouping
                # TODO - investigate prediction on final frame
                stacked_grouping[slc][num_offset + i] = [utils.PitchContour(continuous_pitches, time_idcs[0])]
            # Add the number of notes processed to the
            num_offset += len(stacked_grouping[slc])

        return stacked_grouping

    def write(self, stacked_pitch_list, track):
        """
        TODO
        """

        return NotImplementedError
