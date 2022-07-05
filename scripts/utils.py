# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import warnings


"""
def fill_empties(pitch_list):
    "/""
    Replace empty pitch observations across frames with null (zero) observations.
    Generally, a pitch list should not contain null observations, but it is useful
    to have them in some situations.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames
    "/""

    # Add a null frequency to empty observations
    pitch_list = [p if len(p) else np.array([0.]) for p in pitch_list]

    return pitch_list
"""


def detect_overlap_notes(intervals):
    """
    Determine if a set of intervals contains any overlap.

    Parameters
    ----------
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs
      N - number of notes

    Returns
    ----------
    overlap : bool
      Whether any intervals overlap
    """

    # Make sure the intervals are sorted by onset (abusing this function slightly)
    intervals = tools.sort_batched_notes(intervals, by=0)
    # Check if any onsets occur before the offset of a previous interval
    overlap = np.sum(np.diff(intervals.flatten()) < 0) > 0

    return overlap


def get_active_pitch_count(pitch_list):
    """
    Count the number of active pitches in each frame of a pitch list.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    active_pitch_count : ndarray
      Number of active pitches in each frame
    """

    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)
    # Determine the amount of non-zero frequencies in each frame
    active_pitch_count = np.array([len(p) for p in pitch_list])

    return active_pitch_count


def detect_overlap_pitch_list(pitch_list):
    """
    Determine if a pitch list representation contains overlapping pitch contours.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames

    Returns
    ----------
    overlap : bool
      Whether there are overlapping pitch contours
    """

    # Check if at any time there is more than one observation
    overlap = np.sum(get_active_pitch_count(pitch_list) > 1) > 0

    return overlap


def infer_monophonic_pitch_list_groups(pitch_list, tolerance=None):
    """
    Infer the boundaries of pitch contours by the number of active pitches.

    TODO - for polyphonic data, would need simple PitchContourTracker with state
           in order to match offsets with most likely note (according to what
           pitches still remain after the offset)

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active MIDI pitches
      N - number of frames
    tolerance : float (Optional)
      Pitch difference (in semitones) allowed within adjacent frames of a contour

    Returns
    ----------
    intervals : ndarray (K x 2)
      Array of onset-offset index pairs corresponding to pitch contours
    """

    if detect_overlap_pitch_list(pitch_list):
        # Don't go any further if there is overlap in the pitch contours
        raise ValueError('Only monophonic pitch contours are supported here.')

    # Determine the amount of active pitches in each frame
    active_pitch_count = get_active_pitch_count(pitch_list)

    # Determine where the active pitch count increases by 1
    onset_idcs = np.where(np.diff(np.append(0, active_pitch_count)) == 1)[0]
    # Determine where the active pitch count decreases by 1
    offset_idcs = np.where(np.diff(np.append(active_pitch_count, 0)) == -1)[0]

    # Construct the interval indices for the monophonic streams
    intervals = np.array([onset_idcs, offset_idcs]).T

    """
    # TODO - edge case where offset/onset of two contours occur on adjacent frames
    if tolerance is not None:
        # Replace empty entries with null observations
        pitch_list = fill_empties(pitch_list)

        # Collapse the monophonic pitch contours into a 1D array
        pitch_observations = np.array([np.max(p) for p in pitch_list])

        # Determine the difference in pitch of adjacent frames
        adjacent_differences = np.append(0, np.abs(np.diff(pitch_observations)))
    """

    # TODO - should I actually slice up the pitch list here?

    return intervals


def streams_to_relative_multi_pitch_by_interval(notes, pitch_list, profile, semitone_width=0.5):
    """
    Represent note streams as anchored pitch deviations within a multi pitch array, along
    with an accompanying multi pitch array adjusted in accordance with the pitch list (so
    0 deviation activations can be clearly interpreted). This function is intended for use
    with strictly monophonic data, i.e. with no overlap in the note intervals AND no
    overlapping pitch contours in the pitch list. However, it does support polyphonic data
    and should function robustly under most circumstances, as long the note and pitch contour
    data provided is tightly aligned with no same-pitch notes played in unison or overlapping
    pitch contours with significant deviation from the nominal pitches of their note sources.

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to (non-overlapping) notes
      (K - number of notes)
    pitch_list : tuple (times, pitch_list)
      pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to (non-overlapping) notes
      times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    semitone_width : float
      Amount of deviation from nominal pitch supported

    Returns
    ----------
    relative_multi_pitch : ndarray (F x T)
      Relative pitch deviations anchored to discrete pitches
      F - number of discrete pitches
      T - number of frames
    adjusted_multi_pitch : ndarray (F x T)
      Discrete pitch activation map corresponding to pitch contours
      F - number of discrete pitches
      T - number of frames
    """

    # Unpack the note attributes, removing notes with out-of-bounds nominal pitch
    # TODO - leave suppress warnings true
    pitches, intervals = tools.filter_notes(*notes, profile, False)

    # Unpack the pitch list attributes
    times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    # Check if there is any overlap within the streams
    if detect_overlap_notes(intervals) or detect_overlap_pitch_list(pitch_list):
        warnings.warn('Overlapping streams were provided. Will attempt ' +
                      'to infer note-pitch groupings.', category=RuntimeWarning)

    # Determine the dimensionality for the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(pitch_list)

    # Initialize two empty multi pitch array for relative and adjusted multi pitch data
    relative_multi_pitch = np.zeros((num_pitches, num_frames))
    adjusted_multi_pitch = np.zeros((num_pitches, num_frames))

    # Round note pitches to nearest semitone and subtract the lowest
    # supported note of the instrument to obtain pitch indices
    pitch_idcs = np.round(pitches - profile.low).astype(tools.INT)

    # Duplicate the array of times for each note and stack along a new axis
    times = np.concatenate([[times]] * max(1, len(pitch_idcs)), axis=0)

    # Determine the frame where each note begins and ends
    onset_idcs = np.argmin((times <= intervals[..., :1]), axis=1) - 1
    offset_idcs = np.argmin((times < intervals[..., 1:]), axis=1) - 1

    # Clip all offsets at last frame - they will end up at -1 from
    # previous operation if they occurred beyond last frame time
    offset_idcs[offset_idcs == -1] = num_frames - 1

    # Loop through each note
    for i in range(len(pitch_idcs)):
        # Keep track of adjusted note boundaries without modifying original values
        adjusted_onset, adjusted_offset = onset_idcs[i], offset_idcs[i]

        # Adjust the onset index to the first frame with non-empty pitch observations
        while not len(pitch_list[adjusted_onset]) and adjusted_onset < adjusted_offset:
            adjusted_onset += 1

        # Adjust the offset index to the last frame with non-empty pitch observations
        while not len(pitch_list[adjusted_offset]) and adjusted_offset > adjusted_onset:
            adjusted_offset -= 1

        # Check that there are non-empty pitch observations
        if adjusted_onset != adjusted_offset and len(pitch_list[adjusted_onset]):
            # Extract the (cropped) pitch observations within the note interval
            pitch_observations = pitch_list[adjusted_onset : adjusted_offset + 1]
        else:
            # There are no non-empty pitch observations, throw a warning
            warnings.warn('No pitch observations occur within the note interval. ' +
                          'Inserting average pitch of note instead.', category=RuntimeWarning)
            # Reset the interval to the original note boundaries
            adjusted_onset, adjusted_offset = onset_idcs[i], offset_idcs[i]
            # Populate the frames with the average pitch of the note
            pitch_observations = [np.ndarray([pitches[i]])] * (adjusted_offset + 1 - adjusted_onset)

        # Populate the multi pitch array with adjusted activations for the note
        adjusted_multi_pitch[pitch_idcs[i], adjusted_onset: adjusted_offset + 1] = 1

        # Check if there are any empty observations remaining
        if np.sum([len(p) == 0 for p in pitch_observations]) > 0:
            # There are some gaps in the observations, throw a warning
            warnings.warn('Missing pitch observations within note interval. ' +
                          'Will attempt to interpolate gaps.', category=RuntimeWarning)

        # Convert the cropped pitch list to an array of monophonic pitches, choosing
        # the pitch closest to the nominal value of the note if a frame is polyphonic
        pitch_observations = np.array([p[np.argmin(np.abs(p - pitches[i]))]
                                       if len(p) else 0. for p in pitch_observations])

        # Interpolate between gaps in pitch observations
        pitch_observations = tools.interpolate_gaps(pitch_observations)

        # Determine the nominal pitch of the note
        nominal_pitch = round(pitches[i])

        # Clip pitch observations such they are within supported semitone boundaries
        pitch_observations = np.clip(pitch_observations,
                                     a_min=nominal_pitch - semitone_width,
                                     a_max=nominal_pitch + semitone_width)

        # Compute the deviation between the pitch observations and the nominal value
        deviations = pitch_observations - nominal_pitch

        # Populate the multi pitch array with relative deviations for the note
        relative_multi_pitch[pitch_idcs[i], adjusted_onset: adjusted_offset + 1] = deviations

    return relative_multi_pitch, adjusted_multi_pitch


def streams_to_relative_multi_pitch_by_cluster(notes, pitch_list, profile, semitone_width=0.5):
    # TODO - how to deal with overlapping pitches? - need to cluster somehow
    # TODO - go through pitch list, find clumps of observations with no gaps
    # TODO - how to represent these clumps efficiently?
    # TODO - assign each clump to a note unless there is literally zero overlap
    # TODO - interpolate between any gaps in the clumps assigned to a note
    # TODO - follow same methodology for multi pitch arrays as below

    # Unpack the note attributes
    pitches, intervals = notes

    # Unpack the pitch list attributes
    times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    """
    # Check if there is any overlapping in the streams
    if detect_overlap_notes(intervals) or detect_overlap_pitch_list(pitch_list):
        warnings.warn('Overlapping streams were provided. Will attempt ' +
                      'to infer note-pitch groupings.', category=RuntimeWarning)

    # TODO - implement everything below
    #times, pitch_list = tools.cat_pitch_list(times, pitch_list, times.copy(), pitch_list.copy())

    # Determine the dimensionality for the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(pitch_list)

    # Initialize two empty multi pitch array for relative and adjusted multi pitch data
    relative_multi_pitch = np.zeros((num_pitches, num_frames))
    adjusted_multi_pitch = np.zeros((num_pitches, num_frames))
    """

    # Obtain intervals for pitch observation clusters (contours)
    contour_intervals = infer_monophonic_pitch_list_groups(pitch_list)
    # Extract the corresponding times of the inferred contours
    contour_times = times[contour_intervals.flatten()].reshape(contour_intervals.shape)
    # Compute the duration of each inferred contour
    #contour_durations = np.diff(contour_times).squeeze(-1)

    # Determine the total number of clusters (contours)
    num_contours = contour_intervals.shape[0]

    if num_contours < len(pitches):
        # There are more notes than contours, throw a warning
        warnings.warn('Not enough pitch contours were detected ' +
                      'to account for all notes.', category=RuntimeWarning)

    # Initialize a list for the assignment of each contour to a note
    assignment = -1 * np.ones(num_contours).astype(tools.INT)

    # Loop through each pitch contour
    for i, (start, end) in enumerate(contour_times):
        # Identify points of interest for each contour/note pair
        left_bound, left_inter = np.minimum(start, intervals[:, 0]), np.maximum(start, intervals[:, 0])
        right_inter, right_bound = np.minimum(end, intervals[:, 1]), np.maximum(end, intervals[:, 1])

        # Compute the length in time of the intersections
        iou = right_inter - left_inter
        # Set intersection of non-overlapping pairs to zero
        iou[left_inter >= right_inter] = 0
        # Divide by the union to produce the IOU of each contour/note pair
        iou[left_inter < right_inter] /= (right_bound - left_bound)[left_inter < right_inter]

        # Determine the note with the highest IOU and the value of the IOU
        max_iou, note_idx = np.max(iou), np.argmax(iou)

        if max_iou > 0:
            # Assign the chosen note to the contour
            assignment[i] = note_idx
        else:
            # Pitch contour cannot be paired with a note, throw a warning
            warnings.warn('Inferred pitch contour does not ' +
                          'overlap with any notes.', category=RuntimeWarning)

    # TODO - combine duplicated assignments with overall min/max time?

    # Replace note intervals with contour intervals, ignoring contours without an assignment
    notes = pitches[assignment[assignment != -1]], contour_times[assignment != -1]

    # Parse the inferred contour intervals to obtain the multi pitch arrays
    relative_multi_pitch, \
        adjusted_multi_pitch = streams_to_relative_multi_pitch_by_interval(notes, pitch_list,
                                                                           profile, semitone_width)

    return relative_multi_pitch, adjusted_multi_pitch


def stacked_streams_to_stacked_relative_multi_pitch(stacked_notes, stacked_pitch_list, profile, semitone_width=0.5):
    """
    Convert associated stacked notes and stacked pitch contours into
    a stack of discretized and relative multi pitch activations.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    semitone_width : float
      Amount of deviation from nominal pitch supported

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
        notes_key, pitch_list_key = stacked_notes_keys[i], stacked_pitch_list_keys[i]
        # Obtain the note stream multi pitch arrays for the notes in this slice
        relative_multi_pitch, \
            adjusted_multi_pitch = streams_to_relative_multi_pitch_by_cluster(stacked_notes[notes_key],
                                                                              stacked_pitch_list[pitch_list_key],
                                                                              profile, semitone_width)
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


##################################################
# CURRENTLY UNUSED                               #
##################################################


def pitch_list_to_relative_multi_pitch(pitch_list, profile):
    """
    Convert a MIDI pitch list into a relative multi pitch array.

    Parameters
    ----------
    pitch_list : list of ndarray (N x [...])
      Frame-level observations detailing active pitches
      N - number of frames
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    relative_multi_pitch : ndarray (F x T)
      Relative pitch deviations anchored to discrete pitches
      F - number of discrete pitches
      T - number of frames
    """

    # Throw away out-of-bounds pitche observations
    pitch_list = tools.filter_pitch_list(pitch_list, profile)

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(pitch_list)

    # Initialize an empty multi pitch array
    relative_multi_pitch = np.zeros((num_pitches, num_frames))

    # Loop through each frame
    for i in range(len(pitch_list)):
        # Calculate the semitone difference w.r.t. the lowest note
        pitch_idcs = np.round(pitch_list[i] - profile.low).astype(tools.UINT)
        # Compute the semitone deviation of each pitch
        deviation = pitch_list[i] - np.round(pitch_list[i])
        # Populate the multi pitch array with deviations
        relative_multi_pitch[pitch_idcs, i] = deviation

    return relative_multi_pitch


def stacked_pitch_list_to_stacked_relative_multi_pitch(stacked_pitch_list, profile):
    """
    Convert a stacked MIDI pitch list into a stack of relative multi pitch arrays.

    Parameters
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup

    Returns
    ----------
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Array of deviations from multiple discrete pitches
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    # Initialize an empty list to hold the multi pitch arrays
    stacked_relative_multi_pitch = list()

    # Loop through the slices of notes
    for slc in stacked_pitch_list.keys():
        # Get the pitches and intervals from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Obtain a relative pitch deviation map for the pitch list and add to the list
        relative_multi_pitch = pitch_list_to_relative_multi_pitch(pitch_list, profile)
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(relative_multi_pitch))

    # Collapse the list into an array
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)

    return stacked_relative_multi_pitch
