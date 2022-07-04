# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import warnings


def streams_to_relative_multi_pitch(notes, pitch_list, profile, semitone_width=0.5):
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
        Array of MIDI pitches corresponding to (non-overlapping) notes
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
    # Make sure the notes are sorted by onset
    pitches, intervals = tools.sort_notes(pitches, intervals, by=0)

    # Unpack the pitch list attributes
    times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    # Flag to avoid throwing redundant warnings
    warning_threw = False

    # Check if there are any overlapping note intervals
    if np.sum(np.diff(intervals.flatten()) < 0) > 0:
        warnings.warn('Overlapping notes were provided. Will attempt ' +
                      'to infer note-pitch groupings.', category=RuntimeWarning)
        # Set the flag
        warning_threw = True

    # Check if any frames contain overlapping pitch observations
    if np.sum([len(p) > 1 for p in pitch_list]) > 0 and not warning_threw:
        warnings.warn('Ambiguous (overlapping) pitch contours were provided. ' +
                      'Will attempt to infer note-pitch groupings.', category=RuntimeWarning)

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
            adjusted_multi_pitch = streams_to_relative_multi_pitch(stacked_notes[notes_key],
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
      Array of MIDI pitches corresponding to notes
      N - number of pitch observations (frames)
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
