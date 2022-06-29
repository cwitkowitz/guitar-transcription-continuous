# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import warnings


def monophonic_streams_to_relative_multi_pitch(notes, pitch_list, semitone_width=0.5):
    """
    TODO - description
    TODO - stress here that intervals should not overlap, e.g. monophonic notes

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to notes
      (K - number of notes)
    pitch_list : tuple (times, pitch_list)
      pitch_list : list of ndarray (N x [...])
        Array of MIDI pitches corresponding to (non-overlapping) notes
      times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    semitone_width : float
      Amount of deviation from nominal pitch supported

    Returns
    ----------
    relative_multi_pitch : ndarray (F x T)
      Relative pitch deviations anchored to discrete pitches
      F - number of discrete pitches
      T - number of frames
    """

    # Unpack the note attributes
    pitches, intervals = notes
    # Unpack the pitch list attributes
    times, pitch_list = pitch_list

    relative_multi_pitch = None

    # TODO - assume intervals do not overlap, e.g. monophonic notes
    #        compute diff on flattened (sorted) intervals and ensure all positive (throw warning else)
    # TODO - determine which frames correspond to each note
    # TODO - drop frames with pitches (including 0) outside of the boundaries
    #        imposed by the note's nominal midi pitch and the semitone width
    # TODO - compute the deviation between the pitch of the remaining frames and the nominal value
    # TODO - organize the deviations within a multi pitch array

    return relative_multi_pitch


def stacked_streams_to_stacked_relative_multi_pitch(stacked_notes, stacked_pitch_list, semitone_width=0.5):
    """
    TODO - description

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    semitone_width : float
      Amount of deviation from nominal pitch supported

    Returns
    ----------
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Array of deviations from multiple discrete pitches
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    """

    stacked_relative_multi_pitch = None

    # TODO - convert stacked notes to stacked multipitch
    # TODO - feed notes and pitch list corresponding to one slice into above function
    # TODO - stack the resulting multi pitch arrays for each slice

    return stacked_relative_multi_pitch


def stacked_relative_multi_pitch_to_relative_multi_pitch(stacked_multi_pitch, stacked_relative_multi_pitch):
    """
    TODO - description
    TODO - not sure what they best way to do this is yet, commenting out what I started with

    Parameters
    ----------
    stacked_multi_pitch : ndarray (S x F x T)
      Array of multiple discrete pitch activation maps
      S - number of slices in stack
      F - number of discrete pitches
      T - number of frames
    stacked_relative_multi_pitch : ndarray (S x F x T)
      Array of deviations from multiple discrete pitches
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

    relative_multi_pitch = None

    # Sum the deviations across the pitches of each slice
    #relative_multi_pitch_sum = np.sum(stacked_relative_multi_pitch, axis=-3)
    # Determine the amount of active notes at each pitch in each frame
    #relative_multi_pitch_count = np.sum(stacked_multi_pitch != 0, axis=-3)
    # Make the floor of this count 1 to avoid divide by zero
    #relative_multi_pitch_count[relative_multi_pitch_count == 0] = 1

    # Take the average semitone deviation
    #relative_multi_pitch = relative_multi_pitch_sum / relative_multi_pitch_count

    return relative_multi_pitch


def pitch_list_to_relative_multi_pitch(pitch_list, profile):
    """
    TODO - might not need this but it could still be useful
    """

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(pitch_list)

    # Initialize an empty multi pitch array
    relative_multi_pitch = np.zeros((num_pitches, num_frames))

    # Loop through each frame
    for i in range(len(pitch_list)):
        # Extract the pitch list associated with the frame
        valid_pitches = pitch_list[i]
        # Throw away out-of-bounds pitches
        valid_pitches = valid_pitches[np.round(valid_pitches) >= profile.low]
        valid_pitches = valid_pitches[np.round(valid_pitches) <= profile.high]

        if len(valid_pitches) != len(pitch_list[i]):
            # Print a warning message if continuous pitches were ignored
            warnings.warn('Attempted to represent pitches in multi-pitch array '
                          'which exceed boundaries. These will be ignored.', category=RuntimeWarning)

        # Calculate the semitone difference w.r.t. the lowest note
        pitch_idcs = np.round(valid_pitches - profile.low).astype(tools.UINT)
        # Compute the semitone deviation of each pitch
        deviation = valid_pitches - np.round(valid_pitches)
        # Populate the multi pitch array with deviations
        relative_multi_pitch[pitch_idcs, i] = deviation

    return relative_multi_pitch


def stacked_pitch_list_to_stacked_relative_multi_pitch(stacked_pitch_list, profile):
    """
    TODO - might not need this but it could still be useful
    """

    # Initialize an empty list to hold the multi pitch arrays
    stacked_relative_multi_pitch = list()

    # Loop through the slices of notes
    for slc in stacked_pitch_list.keys():
        # Get the pitches and intervals from the slice
        times, pitch_list = stacked_pitch_list[slc]
        relative_multi_pitch = pitch_list_to_relative_multi_pitch(pitch_list, profile)
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(relative_multi_pitch))

    # Collapse the list into an array
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)

    return stacked_relative_multi_pitch
