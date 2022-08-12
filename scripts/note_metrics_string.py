# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from mir_eval import transcription
from mir_eval import util

import numpy as np


def validate(intervals_ref, pitches_ref, sources_ref, intervals_est, pitches_est, sources_est):
    """
    Determine if valid note groups have been provided for both reference and estimates.

    Parameters
    ----------
    sources_ref : ndarray (n)
      Source for each reference note
    sources_est : ndarray (m)
      Source for each estimated note
    See mir_eval.transcription.validate for others...
    """

    # Check that sources have length equal to the number of pitches
    if not sources_ref.shape[0] == pitches_ref.shape[0]:
        raise ValueError('Reference sources and pitches have different lengths.')
    if not sources_est.shape[0] == pitches_est.shape[0]:
        raise ValueError('Estimated sources and pitches have different lengths.')

    # Perform standard note validation steps
    transcription.validate(intervals_ref, pitches_ref, intervals_est, pitches_est)


def match_notes(ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches, est_sources,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05,
                strict=False):
    """
    Compute the optimal matching between two sets of notes when taking into account the source of the notes.

    Parameters
    ----------
    ref_sources : ndarray (n)
      Source for each reference note
    est_sources : ndarray (m)
      Source for each estimated note
    See mir_eval.transcription.match_notes for others...

    Returns
    ----------
    See mir_eval.transcription.match_notes...
    """

    # Create a list to hold (reference note, estimate note) matches
    matching = list()

    # Obtain a collection of global indices for estimated and reference notes
    glb_ref_idcs, glb_est_idcs = np.arange(len(ref_pitches)), np.arange(len(est_pitches))

    # Loop through all unique sources in the reference data
    for src in np.unique(ref_sources):
        # Determine which indices in reference/estimate correspond to the source
        src_ref_idcs, src_est_idcs = (ref_sources == src), (est_sources == src)

        # Within scope of the source, compute note matching as usual
        src_matching = transcription.match_notes(
            ref_intervals[src_ref_idcs], ref_pitches[src_ref_idcs],
            est_intervals[src_est_idcs], est_pitches[src_est_idcs],
            onset_tolerance, pitch_tolerance, offset_ratio, offset_min_tolerance,
            strict)

        # Convert within-source indices to global note indices
        matching += [(glb_ref_idcs[src_ref_idcs][i],
                      glb_est_idcs[src_est_idcs][j]) for (i, j) in src_matching]

    return matching


def precision_recall_f1_overlap(ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches,
                                est_sources, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2,
                                offset_min_tolerance=0.05, strict=False, beta=1.0):
    """
    Evaluate intervallic note predictions when taking into account the source of the notes.

    Parameters
    ----------
    ref_sources : ndarray (n)
      Source for each reference note
    est_sources : ndarray (m)
      Source for each estimated note
    See mir_eval.transcription.precision_recall_f1_overlap for others...

    Returns
    ----------
    See mir_eval.transcription.precision_recall_f1_overlap...
    """

    # Make sure the provided reference and estimated notes are valid
    validate(ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches, est_sources)

    # Count number of reference and estimated notes
    n_ref = len(ref_pitches)
    n_est = len(est_pitches)

    # When reference notes are empty,
    # metrics are undefined, return 0's
    if n_ref == 0 or n_est == 0:
        return 0., 0., 0., 0.

    # Obtain a list of (reference note, estimate note) match estimates
    matching = match_notes(ref_intervals, ref_pitches, ref_sources,
                           est_intervals, est_pitches, est_sources,
                           onset_tolerance, pitch_tolerance, offset_ratio,
                           offset_min_tolerance, strict)

    # Count number of true positive estimates
    true_positives = len(matching)

    # Compute precision, recall, and f1-score
    precision = float(true_positives) / n_est
    recall = float(true_positives) / n_ref
    f_measure = util.f_measure(precision, recall, beta=beta)

    # Compute the overlap ration between all matched notes
    avg_overlap_ratio = transcription.average_overlap_ratio(ref_intervals, est_intervals, matching)

    return precision, recall, f_measure, avg_overlap_ratio
