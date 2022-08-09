# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.evaluate import StackedPitchListEvaluator as _StackedPitchListEvaluator
from amt_tools.evaluate import StackedNoteEvaluator as _StackedNoteEvaluator
from amt_tools.evaluate import MultipitchEvaluator as _MultipitchEvaluator
from amt_tools.evaluate import PitchListEvaluator as _PitchListEvaluator
from amt_tools.evaluate import NoteEvaluator as _NoteEvaluator

import amt_tools.tools as tools

import constants
import utils

# Regular imports
import numpy as np


class TablaturePitchListEvaluator(_StackedPitchListEvaluator):
    """
    TODO
    """

    def evaluate(self, estimated, reference):
        """
        Evaluate stacked pitch list estimates with respect to a reference,
        where estimates must be matched with the correct degree of freedom.

        Parameters
        ----------
        estimated : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs
        reference : dict
          Dictionary containing (slice -> (times, pitch_list)) pairs

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # TODO

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : 0.,
            tools.KEY_RECALL : 0.,
            tools.KEY_F1 : 0.
        }

        return results


def get_sources_stacked_notes(stacked_notes):
    """
    TODO
    """

    source_keys = list(stacked_notes.keys())

    sources = np.concatenate([[slc] * len(stacked_notes[key][0])
                              for slc, key in enumerate(source_keys)])

    return sources


class TablatureNoteEvaluator(_StackedNoteEvaluator):
    """
    TODO
    """

    def evaluate(self, estimated, reference):
        """
        Evaluate stacked note estimates with respect to a reference,
        where estimates must be matched with the correct degree of freedom.

        Parameters
        ----------
        estimated : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        reference : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs

        Returns
        ----------
        results : dict
          Dictionary containing precision, recall, and f-measure
        """

        # TODO

        sources_est = get_sources_stacked_notes(estimated)
        pitches_est, intervals_est = tools.stacked_notes_to_notes(estimated)

        sources_ref = get_sources_stacked_notes(reference)
        pitches_ref, intervals_ref = tools.stacked_notes_to_notes(reference)

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : 0.,
            tools.KEY_RECALL : 0.,
            tools.KEY_F1 : 0.
        }

        return results


class MultipitchEvaluator(_MultipitchEvaluator):
    """
    TODO
    """

    def unpack(self, estimated, reference):
        """
        TODO
        """

        estimated, reference = super().unpack(estimated, reference)

        estimated = tools.stacked_multi_pitch_to_multi_pitch(estimated)
        reference = tools.stacked_multi_pitch_to_multi_pitch(reference)

        return estimated, reference


class PitchListEvaluator(_PitchListEvaluator):
    """
    TODO
    """

    def unpack(self, estimated, reference):
        """
        TODO
        """

        estimated, reference = super().unpack(estimated, reference)

        estimated = tools.stacked_pitch_list_to_pitch_list(estimated)
        reference = tools.stacked_pitch_list_to_pitch_list(reference)

        return estimated, reference


class NoteEvaluator(_NoteEvaluator):
    """
    TODO
    """

    def unpack(self, estimated, reference):
        """
        TODO
        """

        estimated, reference = super().unpack(estimated, reference)

        estimated = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(estimated))
        reference = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(reference))

        return estimated, reference
