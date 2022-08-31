# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.metrics.string import multipitch_metrics as evaluate_multipitch
from guitar_transcription_continuous.metrics.string import precision_recall_f1_overlap as evaluate_notes
from guitar_transcription_continuous.evaluators.utils import resample_stacked_pitch_list, \
                                                             unroll_sources_stacked_pitch_list, \
                                                             get_sources_stacked_notes
from amt_tools.evaluate import StackedPitchListEvaluator as _StackedPitchListEvaluator
from amt_tools.evaluate import StackedNoteEvaluator as _StackedNoteEvaluator
from amt_tools.evaluate import TablatureEvaluator as _TablatureEvaluator

import amt_tools.tools as tools

# Regular imports
from mir_eval import util

__all__ = [
    'TablaturePitchListEvaluator',
    'TablatureNoteEvaluator',
    'TablatureOnsetEvaluator',
    'TablatureOffsetEvaluator'
]


class TablaturePitchListEvaluator(_StackedPitchListEvaluator):
    """
    Extension of PitchListEvaluator which takes the source of predictions
    and ground-truth into account when matching pitch observations.
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

        # Initialize an empty dictionary to hold results for each tolerance
        results = dict()

        # Make sure both stacked pitch lists are filled uniformly
        stacked_pitch_list_est = resample_stacked_pitch_list(estimated)
        stacked_pitch_list_ref = resample_stacked_pitch_list(reference)

        # Unroll the stacked pitch list estimates and obtain the observation sources
        times_est, pitches_est, sources_est = unroll_sources_stacked_pitch_list(stacked_pitch_list_est)

        # Unroll the stacked pitch list ground-truth and obtain the observation sources
        times_ref, pitches_ref, sources_ref = unroll_sources_stacked_pitch_list(stacked_pitch_list_ref)

        # Convert pitch lists to Hertz
        pitches_ref = tools.pitch_list_to_hz(pitches_ref)
        pitches_est = tools.pitch_list_to_hz(pitches_est)

        for tol in self.pitch_tolerances:
            # Calculate frame-wise precision, recall for continuous pitches
            (p, r, _, _, _, _, _, _, _, _, _, _, _, _) = evaluate_multipitch(ref_time=times_ref,
                                                                             ref_freqs=pitches_ref,
                                                                             ref_sources=sources_ref,
                                                                             est_time=times_est,
                                                                             est_freqs=pitches_est,
                                                                             est_sources=sources_est,
                                                                             window=tol)

            # Calculate the f1-score using the harmonic mean formula
            f = util.f_measure(p, r)

            # Package the results into a dictionary
            results.update({
                f'{tol}' : {
                    tools.KEY_PRECISION : p,
                    tools.KEY_RECALL : r,
                    tools.KEY_F1 : f
                }})

        return results


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
            tools.KEY_PRECISION : p,
            tools.KEY_RECALL : r,
            tools.KEY_F1 : f
        }

        return results


class TablatureOnsetEvaluator(_TablatureEvaluator):
    """
    Simple wrapper to evaluate string-level onsets as tablature.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for onsets activation maps.
        """

        return tools.KEY_ONSETS

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and convert stacked multi pitch data to tablature.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked multi pitch estimate
        reference : dict
          Dictionary containing stacked multi pitch ground-truth

        Returns
        ----------
        tablature_onsets_est : ndarray (S x T)
          Estimated class membership for multiple DOFs
          S - number of strings or degrees of freedom
          T - number of frames
        tablature_onsets_ref : ndarray (S x T)
          Ground-truth class membership for multiple DOFs
          S - number of strings or degrees of freedom
          T - number of frames
        """

        # Call the parent function to unpack the stacked data
        stacked_onsets_est, stacked_onsets_ref = super().unpack(estimated, reference)

        # Collapse the stacked multi pitch arrays
        tablature_onsets_est = tools.stacked_multi_pitch_to_tablature(stacked_onsets_est, self.profile)
        tablature_onsets_ref = tools.stacked_multi_pitch_to_tablature(stacked_onsets_ref, self.profile)

        return tablature_onsets_est, tablature_onsets_ref


class TablatureOffsetEvaluator(TablatureOnsetEvaluator):
    """
    Simple wrapper to evaluate string-level offsets as tablature.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for offsets activation maps.
        """

        return tools.KEY_OFFSETS
