"""
TODO - if opening mir_eval pull request, update documentation in similar manner to
       https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
"""

import numpy as np
from mir_eval import transcription
from mir_eval import util


def validate(ref_intervals, ref_pitches, ref_sources, est_intervals,
             est_pitches, est_sources):
    """
    TODO - if opening mir_eval pull request, update documentation in similar manner to
           https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
    """
    transcription.validate(ref_intervals, ref_pitches, est_intervals,
                           est_pitches)
    # Check that sources have the same length as intervals/pitches
    if not ref_sources.shape[0] == ref_pitches.shape[0]:
        raise ValueError('Reference sources must have the same length as '
                         'pitches and intervals.')
    if not est_sources.shape[0] == est_pitches.shape[0]:
        raise ValueError('Estimated sources must have the same length as '
                         'pitches and intervals.')


def match_notes(
        ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches,
        est_sources, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False):
    """
    TODO - if opening mir_eval pull request, update documentation in similar manner to
           https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
    """

    matching = []

    # for testing:
    #shuffle_idcs = np.arange(len(ref_pitches))
    #np.random.shuffle(shuffle_idcs)
    #est_pitches, est_intervals, est_sources = ref_pitches[shuffle_idcs], ref_intervals[shuffle_idcs], ref_sources[shuffle_idcs]

    # Obtain a collection of global indices for estimated and reference notes
    glb_ref_idcs, glb_est_idcs = np.arange(len(ref_pitches)), np.arange(len(est_pitches))

    for src in np.unique(ref_sources):
        src_ref_idcs = (ref_sources == src)
        src_est_idcs = (est_sources == src)

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


def precision_recall_f1_overlap(
        ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches,
        est_sources, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False,
        beta=1.0):
    """
    TODO - if opening mir_eval pull request, update documentation in similar manner to
           https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
    """
    validate(ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches, est_sources)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0., 0.

    matching = match_notes(
        ref_intervals, ref_pitches, ref_sources, est_intervals, est_pitches,
        est_sources, onset_tolerance, pitch_tolerance, offset_ratio,
        offset_min_tolerance, strict)

    precision = float(len(matching))/len(est_pitches)
    recall = float(len(matching))/len(ref_pitches)
    f_measure = util.f_measure(precision, recall, beta=beta)

    avg_overlap_ratio = transcription.average_overlap_ratio(
        ref_intervals, est_intervals, matching)

    return precision, recall, f_measure, avg_overlap_ratio
