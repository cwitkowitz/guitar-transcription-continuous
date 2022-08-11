'''
TODO - if opening mir_eval pull request, update documentation in similar manner to
       https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
'''

import numpy as np
from mir_eval.multipitch import validate, \
                                resample_multipitch, \
                                frequencies_to_midi, \
                                midi_to_chroma, \
                                compute_num_freqs, \
                                compute_num_true_positives, \
                                compute_accuracy, \
                                compute_err_score
from mir_eval import util
import warnings


def pre_validate(ref_time, ref_freqs, ref_sources,
                 est_time, est_freqs, est_sources):
    """
    TODO - if opening mir_eval pull request, update documentation in similar manner to
           https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
    """
    # Check that sources have the same length as number of frames
    if not ref_sources.shape[0] == ref_time.shape[0]:
        raise ValueError('Reference sources must have the same length as '
                         'pitch observations.')
    if not est_sources.shape[0] == est_time.shape[0]:
        raise ValueError('Estimated sources must have the same length as '
                         'pitch observations.')
    if len(np.setdiff1d(ref_sources, est_sources)):
        warnings.warn('Some reference sources have no corresponding '
                      'observations in estimates. This could cause '
                      'unwanted behavior if reference and estimates '
                      'have different sampling periods for other sources.')
    if len(np.setdiff1d(est_sources, ref_sources)):
        warnings.warn('Some estimate sources have no corresponding '
                      'observations in reference. This could cause '
                      'unwanted behavior if reference and estimates '
                      'have different sampling periods.')


def multipitch_metrics(ref_time, ref_freqs, ref_sources,
                       est_time, est_freqs, est_sources, **kwargs):
    """
    TODO - if opening mir_eval pull request, update documentation in similar manner to
           https://github.com/cwitkowitz/mir_eval/blob/master/mir_eval/transcription_velocity.py
    """
    pre_validate(ref_time, ref_freqs, ref_sources,
                 est_time, est_freqs, est_sources)

    n_ref, n_est = np.empty(0), np.empty(0)

    true_positives, true_positives_chroma = np.empty(0), np.empty(0)

    for src in np.unique(np.concatenate((ref_sources, est_sources))):
        src_ref_idcs = (ref_sources == src)
        src_est_idcs = (est_sources == src)

        _ref_time, _est_time = ref_time[src_ref_idcs], est_time[src_est_idcs]

        _ref_freqs = [ref_freqs[i] for i in np.where(src_ref_idcs)[0]]
        _est_freqs = [est_freqs[i] for i in np.where(src_est_idcs)[0]]

        validate(_ref_time, _ref_freqs, _est_time, _est_freqs)

        # resample est_freqs if est_times is different from ref_times
        if _est_time.size != _ref_time.size or not np.allclose(_est_time, _ref_time):
            warnings.warn("Estimate times not equal to reference times for source {}. "
                          "Resampling to common time base.".format(src))
            _est_freqs = resample_multipitch(_est_time, _est_freqs, _ref_time)

        # convert frequencies from Hz to continuous midi note number
        ref_freqs_midi = frequencies_to_midi(_ref_freqs)
        est_freqs_midi = frequencies_to_midi(_est_freqs)

        # count number of occurrences across
        n_ref = np.append(n_ref, compute_num_freqs(ref_freqs_midi))
        n_est = np.append(n_est, compute_num_freqs(est_freqs_midi))

        # compute chroma wrapped midi number
        ref_freqs_chroma = midi_to_chroma(ref_freqs_midi)
        est_freqs_chroma = midi_to_chroma(est_freqs_midi)

        # compute the number of true positives
        true_positives = np.append(true_positives, util.filter_kwargs(
            compute_num_true_positives, ref_freqs_midi, est_freqs_midi, **kwargs))

        # compute the number of true positives ignoring octave mistakes
        true_positives_chroma = np.append(
            true_positives_chroma, util.filter_kwargs(
            compute_num_true_positives, ref_freqs_chroma,
            est_freqs_chroma, chroma=True, **kwargs))

    # compute accuracy metrics
    precision, recall, accuracy = compute_accuracy(
        true_positives, n_ref, n_est)

    # compute error metrics
    e_sub, e_miss, e_fa, e_tot = compute_err_score(
        true_positives, n_ref, n_est)

    # compute accuracy metrics ignoring octave mistakes
    precision_chroma, recall_chroma, accuracy_chroma = compute_accuracy(
        true_positives_chroma, n_ref, n_est)

    # compute error metrics ignoring octave mistakes
    e_sub_chroma, e_miss_chroma, e_fa_chroma, e_tot_chroma = compute_err_score(
        true_positives_chroma, n_ref, n_est)

    return (precision, recall, accuracy, e_sub, e_miss, e_fa, e_tot,
            precision_chroma, recall_chroma, accuracy_chroma, e_sub_chroma,
            e_miss_chroma, e_fa_chroma, e_tot_chroma)
