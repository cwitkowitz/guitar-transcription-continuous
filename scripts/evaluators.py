# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.evaluate import StackedPitchListEvaluator as _StackedPitchListEvaluator
from amt_tools.evaluate import StackedNoteEvaluator as _StackedNoteEvaluator
from amt_tools.evaluate import MultipitchEvaluator as _MultipitchEvaluator
from amt_tools.evaluate import PitchListEvaluator as _PitchListEvaluator
from amt_tools.evaluate import NoteEvaluator as _NoteEvaluator

from string_level_metrics import precision_recall_f1_overlap as evaluate_notes

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
    Obtain the sources corresponding to a collection of stacked notes if they were to be collapsed.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs

    Returns
    ----------
    sources : ndarray (N)
      Array of key indices corresponding to each slice
      N - number of notes
    """

    # Obtain a list of the keys for each slice
    source_keys = list(stacked_notes.keys())
    # Repeat each key index (source) for each note associated with the source
    sources = np.concatenate([[slc] * len(stacked_notes[key][0])
                              for slc, key in enumerate(source_keys)])

    return sources


class TablatureNoteEvaluator(_StackedNoteEvaluator):
    """
    Extension of NoteEvaluator which takes the source of
    predictions and ground-truth into account when matching notes.
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

        # Obtain the estimated note sources
        sources_est = get_sources_stacked_notes(estimated)
        # Collapse the estimated notes
        pitches_est, intervals_est = tools.stacked_notes_to_notes(estimated)

        # Obtain the ground-truth note sources
        sources_ref = get_sources_stacked_notes(reference)
        # Collapse the ground-truth notes
        pitches_ref, intervals_ref = tools.stacked_notes_to_notes(reference)

        # Convert notes to Hertz
        pitches_ref = tools.notes_to_hz(pitches_ref)
        pitches_est = tools.notes_to_hz(pitches_est)

        # Calculate precision, recall, and f1 score of
        # source-matched notes with or without offset
        p, r, f, _ = evaluate_notes(ref_intervals=intervals_ref,
                                    ref_pitches=pitches_ref,
                                    ref_sources=sources_ref,
                                    est_intervals=intervals_est,
                                    est_pitches=pitches_est,
                                    est_sources=sources_est,
                                    offset_ratio=self.offset_ratio)

        # Package the results into a dictionary
        results = {
            tools.KEY_PRECISION : 0.,
            tools.KEY_RECALL : 0.,
            tools.KEY_F1 : 0.
        }

        return results


class MultipitchEvaluator(_MultipitchEvaluator):
    """
    Simple wrapper to support stacked multi pitch estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked multi pitch data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked multi pitch estimate
        reference : dict
          Dictionary containing stacked multi pitch ground-truth

        Returns
        ----------
        multi_pitch_est : ndarray (F x N)
          Estimated multi pitch data
          F - number of discrete pitches
          N - number of frames
        multi_pitch_ref : ndarray (F x N)
          Ground-truth multi pitch data
          Same dimensions as multi_pitch_est
        """

        # Call the parent function to unpack the stacked data
        stacked_multi_pitch_est, stacked_multi_pitch_ref = super().unpack(estimated, reference)

        # Collapse the stacked multi pitch arrays
        multi_pitch_est = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch_est)
        multi_pitch_ref = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch_ref)

        return multi_pitch_est, multi_pitch_ref


class PitchListEvaluator(_PitchListEvaluator):
    """
    Simple wrapper to support stacked pitch list estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked pitch list data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked pitch list estimate
        reference : dict
          Dictionary containing stacked pitch list ground-truth

        Returns
        ----------
        pitch_list_est : tuple (times_est, _pitch_list_est)
          Estimated pitch list data
          _pitch_list_est : list of ndarray (T1 x [...])
            Collection of MIDI pitches
          times_est : ndarray (T1)
            Time in seconds associated with each frame
          (T1 - number of observations in estimates (frames))
        pitch_list_ref : tuple (times_ref, _pitch_list_ref)
          Ground-truth pitch list data
          _pitch_list_ref : list of ndarray (T2 x [...])
            Collection of MIDI pitches
          times_ref : ndarray (T2)
            Time in seconds associated with each frame
          (T2 - number of observations in ground-truth (frames))
        """

        # Call the parent function to unpack the stacked data
        stacked_pitch_list_est, stacked_pitch_list_ref = super().unpack(estimated, reference)

        # Collapse the stacked pitch lists
        pitch_list_est = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list_est)
        pitch_list_ref = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list_ref)

        return pitch_list_est, pitch_list_ref


class NoteEvaluator(_NoteEvaluator):
    """
    Simple wrapper to support stacked note estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked note data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked notes estimate
        reference : dict
          Dictionary containing stacked notes ground-truth

        Returns
        ----------
        notes_est : ndarray (K x 3)
          Estimated note intervals and pitches by row
          K - number of estimated notes
        notes_ref : ndarray (L x 3)
          Ground-truth note intervals and pitches by row
          L - number of ground-truth notes
        """

        # Call the parent function to unpack the stacked data
        stacked_notes_est, stacked_notes_ref = super().unpack(estimated, reference)

        # Collapse the stacked notes and convert them to batched representations
        notes_est = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(stacked_notes_est))
        notes_ref = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(stacked_notes_ref))

        return notes_est, notes_ref
