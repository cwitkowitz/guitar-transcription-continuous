# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
from mir_eval.multipitch import resample_multipitch

import numpy as np

__all__ = [
    'resample_stacked_pitch_list',
    'unroll_sources_stacked_pitch_list',
    'get_sources_stacked_notes'
]


def resample_stacked_pitch_list(stacked_pitch_list, global_times=None):
    """
    Resample each pitch list in a stack to a common collection of times. If a
    collection of times is not provided, it is derived from all unique times
    within the stack.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    global_times : ndarray (N) (Optional)
      Array of global times for resampling
      N - number of global time samples

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    if global_times is None:
        # Determine the global collection of times represented in the stack
        global_times, _ = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

    # Loop through the pitch list in each slice
    for key, pitch_list in stacked_pitch_list.items():
        # Resample the pitch list to align with the global times
        stacked_pitch_list[key] = global_times, resample_multipitch(*pitch_list, global_times)

    return stacked_pitch_list


def unroll_sources_stacked_pitch_list(stacked_pitch_list):
    """
    Unroll a stacked pitch list, differentiating pitch
    observations from each slice with a new source attribute.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs

    Returns
    ----------
    times : ndarray (N)
      Time in seconds associated with each frame
      N - number of time samples (frames)
    pitch_list : list of ndarray (N x [...])
      Array of pitches active across each frame
    sources : ndarray (N)
      Array of key indices corresponding to each source
    """

    # Initialize empty collections for pitch list contents
    times, pitch_list, sources = np.empty(0), [], np.empty(0).astype(tools.INT)

    # Loop through the contents of each slice of the stacked pitch list
    for slc, (_times, _pitch_list) in enumerate(stacked_pitch_list.values()):
        # Concatenate the pitch list contents for this slice
        times, pitch_list = np.append(times, _times), pitch_list + _pitch_list
        # Repeat each key index (source) for each observation (frame)
        sources = np.append(sources, [slc] * len(_times))

    return times, pitch_list, sources


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
      Array of key indices corresponding to each source
      N - total number of notes across the stack
    """

    # Obtain a list of the keys for each slice
    source_keys = list(stacked_notes.keys())
    # Repeat each key index (source) for each note associated with the source
    sources = np.concatenate([[slc] * len(stacked_notes[key][0])
                              for slc, key in enumerate(source_keys)]).astype(tools.INT)

    return sources
