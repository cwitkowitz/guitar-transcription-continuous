# My imports
from .tracking import ContourTracker

import amt_tools.tools as tools

# Regular imports
import numpy as np
import warnings

__all__ = [
    'get_note_contour_grouping_by_cluster',
    'streams_to_continuous_multi_pitch',
    'stacked_streams_to_stacked_continuous_multi_pitch',
    'stacked_relative_multi_pitch_to_relative_multi_pitch',
    'continuous_multi_pitch_to_pitch_list',
    'stacked_continuous_multi_pitch_to_stacked_pitch_list'
]


def get_note_contour_grouping_by_cluster(notes, pitch_list, semitone_radius=0.5, stream_tolerance=1.0,
                                         minimum_contour_duration=None, attempt_corrections=False,
                                         suppress_warnings=True):
    """
    Associate pitch contours in a pitch list with a collection of notes, based off
    of clusters of pitch observations and their intersection-over-union (IoU) with
    the notes and proximity w.r.t pitch.

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to (non-overlapping) notes
      (K - number of notes)
    pitch_list : tuple (_times, _pitch_list)
      pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to (non-overlapping) notes
      _times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    semitone_radius : float
      Amount of deviation from nominal pitch supported
    stream_tolerance : float
      Pitch difference tolerated across adjacent frames of a single contour
    minimum_contour_duration : float (Optional)
      Minimum amount of time in milliseconds a contour should span to be considered
    attempt_corrections : bool
      Whether to avoid grouping contours and notes with pitch mismatch and attempt to remedy these situations
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    ----------
    grouping : (note, [contours]) pairs : dict
      Dictionary containing (note_idx -> [PitchContour, ...]) pairs
    """

    # Unpack the note attributes
    pitches, intervals = notes

    # Unpack the pitch list attributes
    _times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    # Initialize a new pitch contour tracker
    tracker = ContourTracker()
    # Track the contours within the provided pitch list
    tracker.parse_pitch_list(pitch_list, tolerance=stream_tolerance)

    # Compute the duration of each inferred contour
    contour_durations = np.diff(tracker.get_contour_intervals(_times)).squeeze(-1)

    if minimum_contour_duration is not None:
        # Determine which contours have duration above the minimum required
        valid_contours = contour_durations > minimum_contour_duration * 1E-3
        # Remove contours with intervals smaller than specified threshold
        tracker.filter_contours(valid_contours)

    # Initialize an array for the assignment of each contour to a note
    assignment = np.array([-1] * tracker.get_num_contours(), dtype=tools.INT)

    if len(pitches):
        # Loop through each pitch contour's time interval
        for i, ((start, end), avg) in enumerate(zip(tracker.get_contour_intervals(_times),
                                                    tracker.get_contour_averages(0.25, 0.5))):
            # Identify points of interest for each contour/note pair
            left_bound, left_inter = np.minimum(start, intervals[:, 0]), np.maximum(start, intervals[:, 0])
            right_inter, right_bound = np.minimum(end, intervals[:, 1]), np.maximum(end, intervals[:, 1])

            # Compute the length in time of the intersections
            iou = right_inter - left_inter
            # Set intersection of non-overlapping pairs to zero
            iou[left_inter >= right_inter] = 0
            # Divide by the union to produce the IOU of each contour/note pair
            iou[left_inter < right_inter] /= (right_bound - left_bound)[left_inter < right_inter]

            # Compute pitch proximity scores using exponential distribution with lambda=1
            pitch_proximities = np.exp(-np.abs(pitches - avg))

            # Point-wise multiply the IOU and pitch proximities to obtain matching scores
            matching_scores = iou * pitch_proximities

            # Determine the note with the highest score and the value of the score
            max_score, note_idx = np.max(matching_scores), np.argmax(matching_scores)

            if max_score > 0:
                # Assign the chosen note to the contour
                assignment[i] = note_idx
            else:
                if not suppress_warnings:
                    # Pitch contour cannot be paired with a note, throw a warning
                    warnings.warn('Inferred pitch contour does not ' +
                                  'overlap with any notes.', category=RuntimeWarning)

    # Ignore pitch contours without an assignment
    tracker.filter_contours(assignment != -1)
    assignment = assignment[assignment != -1]

    # Determine which contours where assigned a note index
    assigned_notes = np.unique(assignment)

    if len(np.setdiff1d(np.arange(len(pitches)), assigned_notes)) and not suppress_warnings:
        # Some notes are not assigned to any pitch contours
        # TODO - occurs quite frequently when notes of same pitch are
        #        played consecutively with no silent frames in between.
        #        Currently, addressing this case isn't really important,
        #        since for now there is no explicit onset estimation and
        #        the result is only multipitch activity. However, this
        #        could otherwise be addressed by identify notes of the
        #        same pitch with boundaries within the same frame, and
        #        breaking apart inferred contours at the intersection of
        #        these frames.
        warnings.warn('Some notes are not accounted for ' +
                      'by any pitch contours.', category=RuntimeWarning)

    if len(assigned_notes) < len(assignment) and not suppress_warnings:
        # More than one pitch contour occurs maximally within a single note
        warnings.warn('Multiple pitch contours assigned ' +
                      'to the same note.', category=RuntimeWarning)

    # Compute nominal pitch values for the contours
    contour_region_averages = tracker.get_contour_averages(0.25, 0.5)

    # TODO - select 10th percentile vs. mean as identified note pitch

    # Compute the difference in magnitude between the average pitch of
    # the contour and the average pitch of the note for each grouping
    magnitude_differences = np.abs(pitches[assignment] - contour_region_averages)

    for i in np.where(magnitude_differences > semitone_radius)[0]:
        if not suppress_warnings:
            # Average pitches of a contour and note match is too large
            warnings.warn('Average pitch of grouped contour and note differ ' +
                          'beyond specified semitone width.', category=RuntimeWarning)
        if attempt_corrections:
            # Create new note entry for the poorly matching contour
            pitches = np.append(pitches, contour_region_averages[i])
            # Change the assignment of the contour
            assignment[i] = len(pitches) - 1
            # Update the list of assigned notes
            assigned_notes = np.unique(assignment)

    # Initialize a dictionary to hold (note, [contours]) pairs
    grouping = dict()

    # Loop through all notes (including additions from corrections)
    for n in range(len(pitches)):
        # Pair the note's index with a list of the contours assigned to the note
        grouping[n] = [c for (i, c) in enumerate(tracker.contours) if assignment[i] == n]

    return grouping


def streams_to_continuous_multi_pitch(notes, pitch_list, profile, times=None, suppress_warnings=True, **kwargs):
    """
    Obtain discretized and relative multi pitch information for pitch
    contours in a pitch list associated with a collection of notes.

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to (non-overlapping) notes
      (K - number of notes)
    pitch_list : tuple (_times, _pitch_list)
      pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to (non-overlapping) notes
      _times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    times : ndarray (L) (Optional)
      Array of alternate times for optional resampling of pitch list
      L - number of time samples (frames)
    suppress_warnings : bool
      Whether to ignore warning messages
    **kwargs : N/A
      Arguments for grouping notes and contours

    Returns
    ----------
    relative_multi_pitch : ndarray (F x T)
      Array of deviations from multiple discrete pitches
      F - number of discrete pitches
      T - number of frames
    adjusted_multi_pitch : ndarray (F x T)
      Discrete pitch activation map aligned with pitch contours
      F - number of discrete pitches
      T - number of frames
    """

    # Unpack the note attributes, removing notes with out-of-bounds nominal pitch
    pitches, intervals = tools.filter_notes(*notes, profile, suppress_warnings=suppress_warnings)

    # Unpack the pitch list attributes
    _times, _pitch_list = pitch_list

    # Determine the dimensionality for the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(_pitch_list)

    # Initialize two empty multi pitch arrays for relative and adjusted multi pitch data
    relative_multi_pitch = np.zeros((num_pitches, num_frames))
    adjusted_multi_pitch = np.zeros((num_pitches, num_frames))

    # Round note pitches to nearest semitone and subtract the lowest
    # supported note of the instrument to obtain pitch indices
    pitch_idcs = np.round(pitches - profile.low).astype(tools.INT)

    # Obtain a note-contour grouping by cluster
    grouping = get_note_contour_grouping_by_cluster((pitches, intervals), pitch_list,
                                                    suppress_warnings=suppress_warnings,
                                                    **kwargs)

    # Loop through each note
    for i in range(len(pitches)):
        # Loop through all contours associated with the note
        for contour in grouping[i]:
            # Obtain the contour interval
            onset, offset = contour.get_interval()
            # Populate the multi pitch array with activations for the note
            adjusted_multi_pitch[pitch_idcs[i], onset : offset + 1] = 1

            # Compute the deviation between the pitch observations and the nominal value
            deviations = contour.pitch_observations - round(pitches[i])

            # Populate the multi pitch array with relative deviations for the note
            relative_multi_pitch[pitch_idcs[i], onset: offset + 1] = deviations

    if times is not None:
        # If times given, obtain indices to resample the multi pitch arrays
        resample_idcs = tools.get_resample_idcs(_times, times)

        if resample_idcs is None:
            # Initialize empty arrays with the expected number of frames
            relative_multi_pitch = np.zeros((num_pitches, len(times)))
            adjusted_multi_pitch = np.zeros((num_pitches, len(times)))
        else:
            # Reduce the multi pitch arrays to the resample indices
            # TODO - elegant solution, but could result in notes with duration
            #        shorter than a frame being erased if undersampled
            relative_multi_pitch = relative_multi_pitch[..., resample_idcs]
            adjusted_multi_pitch = adjusted_multi_pitch[..., resample_idcs]

    return relative_multi_pitch, adjusted_multi_pitch


def stacked_streams_to_stacked_continuous_multi_pitch(stacked_notes, stacked_pitch_list, profile, **kwargs):
    """
    Convert associated stacked notes and stacked pitch contours into
    a stack of discretized and relative multi pitch activations.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    stacked_pitch_list : dict
      Dictionary containing (slice -> (_times, pitch_list)) pairs
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    **kwargs : N/A
      Arguments for grouping notes and contours

    Returns
    ----------
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Array of deviations from multiple discrete pitches
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    stacked_adjusted_multi_pitch : ndarray (S x F x T)
      Discrete pitch activation map aligned with pitch contours
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize empty lists to hold the stacked multi pitch arrays
    stacked_relative_multi_pitch = list()
    stacked_adjusted_multi_pitch = list()

    # Obtain the in-order keys for each stack
    stacked_notes_keys = list(stacked_notes.keys())
    stacked_pitch_list_keys = list(stacked_pitch_list.keys())

    # Loop through the slices of the collections
    for i in range(len(stacked_notes_keys)):
        # Extract the key for the current slice in each collection
        key_n, key_pl = stacked_notes_keys[i], stacked_pitch_list_keys[i]
        # Obtain the continuous stream multi pitch arrays for the notes in this slice
        relative_multi_pitch, \
            adjusted_multi_pitch = streams_to_continuous_multi_pitch(stacked_notes[key_n],
                                                                     stacked_pitch_list[key_pl],
                                                                     profile, **kwargs)
        # Add the multi pitch arrays to their respective stacks
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(relative_multi_pitch))
        stacked_adjusted_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(adjusted_multi_pitch))

    # Collapse the lists into arrays
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)
    stacked_adjusted_multi_pitch = np.concatenate(stacked_adjusted_multi_pitch)

    return stacked_relative_multi_pitch, stacked_adjusted_multi_pitch


def stacked_relative_multi_pitch_to_relative_multi_pitch(stacked_relative_multi_pitch,
                                                         stacked_adjusted_multi_pitch=None):
    """
    Collapse a stacked relative multi pitch array along the slice dimension.

    Parameters
    ----------
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Array of deviations from multiple discrete pitches
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    stacked_adjusted_multi_pitch : ndarray (S x F x T) (optional)
      Discrete pitch activation map aligned with pitch contours
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    relative_multi_pitch : ndarray (F x T)
      Relative pitch deviations anchored to discrete pitches
      F - number of discrete pitches
      T - number of frames
    """

    if stacked_adjusted_multi_pitch is None:
        # Default the multi pitch activations to non-zero pitch deviations
        stacked_adjusted_multi_pitch = stacked_relative_multi_pitch != 0

    # Sum the deviations across the pitches of each slice
    relative_multi_pitch = np.sum(stacked_relative_multi_pitch, axis=-3)
    # Determine the amount of active notes at each pitch in each frame
    adjusted_multi_pitch_count = np.sum(stacked_adjusted_multi_pitch, axis=-3)

    # Take the average semitone deviation
    relative_multi_pitch[adjusted_multi_pitch_count > 0] /= \
        adjusted_multi_pitch_count[adjusted_multi_pitch_count > 0]

    return relative_multi_pitch


def continuous_multi_pitch_to_pitch_list(discrete_multi_pitch, relative_multi_pitch, profile):
    """
    Convert discrete and relative multi pitch arrays into a pitch list.

    Parameters
    ----------
    discrete_multi_pitch : ndarray (F x T)
      Discrete pitch activation map
      F - number of discrete pitches
      T - number of frames
    relative_multi_pitch : ndarray (F x T)
      Relative pitch deviations anchored to discrete pitches
      F - number of discrete pitches
      T - number of frames
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    pitch_list : list of ndarray (T x [...])
      Frame-level observations detailing active pitches
      T - number of frames
    """

    # Determine the number of frames in the multi pitch array
    num_frames = discrete_multi_pitch.shape[-1]

    # Initialize empty pitch arrays for each time entry
    pitch_list = [np.empty(0)] * num_frames

    # Determine which frames contain pitch activity
    non_silent_frames = np.where(np.sum(discrete_multi_pitch, axis=-2) > 0)[-1]

    # Loop through the frames containing pitch activity
    for i in list(non_silent_frames):
        # Determine the MIDI pitches active in the frame
        pitch_idcs = np.where(discrete_multi_pitch[..., i])[-1]
        # Compute the continuous pitches for the frame and add to the list
        pitch_list[i] = profile.low + pitch_idcs + relative_multi_pitch[pitch_idcs, i]

    return pitch_list


def stacked_continuous_multi_pitch_to_stacked_pitch_list(stacked_discrete_multi_pitch,
                                                         stacked_relative_multi_pitch, times, profile):
    """
    Convert a stack of discrete and relative multi pitch arrays into a stack of pitch lists.

    Parameters
    ----------
    stacked_discrete_multi_pitch : ndarray (S x F x T)
      Discrete pitch activation map
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Relative pitch deviations anchored to discrete pitches
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    times : ndarray (T)
      Time in seconds of beginning of each frame
      T - number of time samples (frames)
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Determine the number of slices in the stacked multi pitch array
    stack_size = stacked_discrete_multi_pitch.shape[-3]

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the multi pitch arrays pertaining to this slice
        slice_discrete, slice_relative = stacked_discrete_multi_pitch[slc], stacked_relative_multi_pitch[slc]

        # Convert the multi pitch array to a pitch list
        _slice_pitch_list = continuous_multi_pitch_to_pitch_list(slice_discrete, slice_relative, profile)

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(tools.pitch_list_to_stacked_pitch_list(times, _slice_pitch_list, slc))

    return stacked_pitch_list
