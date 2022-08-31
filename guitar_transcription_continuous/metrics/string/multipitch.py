# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from mir_eval.multipitch import validate, \
                                resample_multipitch, \
                                frequencies_to_midi, \
                                midi_to_chroma, \
                                compute_num_freqs, \
                                compute_num_true_positives, \
                                compute_accuracy, \
                                compute_err_score
from mir_eval import util

import numpy as np
import warnings


def validate_sources(times_ref, sources_ref, times_est, sources_est):
    """
    Determine if valid sources have been provided, given
    corresponding observation times, for both reference and estimates.

    Parameters
    ----------
    times_ref : ndarray (N)
      Time for each reference observation
    sources_ref : ndarray (N)
      Source for each reference observation
    times_est : ndarray (L)
      Time for each estimate observation
    sources_est : ndarray (L)
      Source for each estimate observation
    """

    # Check that sources have length equal to the number of observations
    if not sources_ref.shape[0] == times_ref.shape[0]:
        raise ValueError('Reference sources must have the same length as pitch observations.')
    if not sources_est.shape[0] == times_est.shape[0]:
        raise ValueError('Estimated sources must have the same length as pitch observations.')


def multipitch_metrics(ref_time, ref_freqs, ref_sources, est_time, est_freqs, est_sources, **kwargs):
    """
    Evaluate frame-wise continuous multipitch predictions
    when taking into account the source of the pitches.

    Parameters
    ----------
    ref_sources : ndarray (N)
      Source for each reference observation
    est_sources : ndarray (L)
      Source for each estimate observation
    See mir_eval.multipitch.metrics for others...

    Returns
    ----------
    See mir_eval.multipitch.metrics...
    """

    # Make sure the provided sources are valid
    validate_sources(ref_time, ref_sources, est_time, est_sources)

    # Create empty arrays to hold counts
    n_ref = np.empty(0)
    n_est = np.empty(0)
    true_positives = np.empty(0)
    true_positives_chroma = np.empty(0)

    # Determine which sources are present in the reference/estimate
    all_sources = np.unique(np.concatenate((ref_sources, est_sources)))

    for src in all_sources:
        # Determine which indices in reference/estimate correspond to the source
        src_ref_idcs, src_est_idcs = (ref_sources == src), (est_sources == src)

        # Obtain the reference/estimate time arrays for the source
        _ref_time, _est_time = ref_time[src_ref_idcs], est_time[src_est_idcs]

        # Obtain the reference/estimate pitch lists for the source
        _ref_freqs = [ref_freqs[i] for i in np.where(src_ref_idcs)[0]]
        _est_freqs = [est_freqs[i] for i in np.where(src_est_idcs)[0]]

        # Make sure the pitch list data for this source is valid
        validate(_ref_time, _ref_freqs, _est_time, _est_freqs)

        # Check if there are sources in estimates that are not present in reference
        if not _ref_time.size:
            raise ValueError(f'No reference data for source \'{src}\'. Resampling '
                             f'will nullify estimates. Empty observations should '
                             f'be provided as reference data for this source.')

        # Resample estimated pitch list if estimate times differ from reference
        if _est_time.size != _ref_time.size or not np.allclose(_est_time, _ref_time):
            warnings.warn(f'Estimate times not equal to reference times for '
                          f'source \'{src}\'. Resampling to common time base.')
            _est_freqs = resample_multipitch(_est_time, _est_freqs, _ref_time)

        # Convert frequencies from Hz to continuous midi number
        _ref_freqs_midi = frequencies_to_midi(_ref_freqs)
        _est_freqs_midi = frequencies_to_midi(_est_freqs)

        # Count number of estimate and reference pitches for the source
        n_ref = np.append(n_ref, compute_num_freqs(_ref_freqs_midi))
        n_est = np.append(n_est, compute_num_freqs(_est_freqs_midi))

        # Compute chroma wrapped continuous midi number
        _ref_freqs_chroma = midi_to_chroma(_ref_freqs_midi)
        _est_freqs_chroma = midi_to_chroma(_est_freqs_midi)

        # Compute the number of true positives for the source
        true_positives = np.append(true_positives, util.filter_kwargs(
            compute_num_true_positives, _ref_freqs_midi, _est_freqs_midi, **kwargs))

        # Compute the number of true positives for the source while ignoring octave mistakes
        true_positives_chroma = np.append(true_positives_chroma,
                                          util.filter_kwargs(compute_num_true_positives, _ref_freqs_chroma,
                                                             _est_freqs_chroma, chroma=True, **kwargs))

    # Compute accuracy metrics across all sources
    precision, recall, accuracy = compute_accuracy(true_positives, n_ref, n_est)

    # Compute error metrics across all sources
    e_sub, e_miss, e_fa, e_tot = compute_err_score(true_positives, n_ref, n_est)

    # Compute accuracy metrics across all sources while ignoring octave mistakes
    precision_chroma, recall_chroma, accuracy_chroma = \
        compute_accuracy(true_positives_chroma, n_ref, n_est)

    # Compute error metrics across all sources while ignoring octave mistakes
    e_sub_chroma, e_miss_chroma, e_fa_chroma, e_tot_chroma = \
        compute_err_score(true_positives_chroma, n_ref, n_est)

    return precision, recall, accuracy, e_sub, e_miss, e_fa, e_tot, precision_chroma, \
           recall_chroma, accuracy_chroma, e_sub_chroma, e_miss_chroma, e_fa_chroma, e_tot_chroma
