# My imports
from .tracking import PitchContour

import amt_tools.tools as tools

# Regular imports
from mir_eval.multipitch import resample_multipitch

import numpy as np
import warnings
import librosa

__all__ = [
    'get_note_contour_grouping_by_index',
    'extract_continuous_multi_pitch_jams',
    'extract_stacked_continuous_multi_pitch_jams',
    'get_note_contour_grouping_by_interval',
    'get_rotarized_relative_multi_pitch',
    'pitch_list_to_relative_multi_pitch',
    'stacked_pitch_list_to_stacked_relative_multi_pitch'
]


def get_note_contour_grouping_by_index(jam, times, index=None, suppress_warnings=True):
    """
    Parse JAMS data to obtain a grouping between note indices and pitch
    observations based off of the "index" field of each observation.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    times : ndarray (N)
      Times in seconds for computing frame indices
      N - number of time samples
    index : int or None (Optional)
      Specific index to isolate
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    ----------
    pitches : ndarray (K)
      Array of pitches corresponding to dictionary entries
    grouping : (note, [contours]) pairs : dict
      Dictionary containing (note_idx -> [PitchContour]) pairs
    """

    # Extract all of the pitch and note annotations
    pitch_data_slices = jam.annotations[tools.JAMS_PITCH_HZ]
    note_data_slices = jam.annotations[tools.JAMS_NOTE_MIDI]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold (note, [contours]) pairs
    grouping = dict()

    # Initialize a note index offset to avoid overlap across slices
    index_offset = 0

    # Initialize an array/list to keep track of note pitch/ordering
    note_pitches = np.empty(0)
    note_onset_times = list()

    # Obtain a list of all slice indices
    slices = list(range(stack_size))

    # Check if a specific slice index was chosen
    if index is not None and index in slices:
        # Ignore all other slices
        slices = [index]

    # Loop through the selected slices
    for slc in slices:
        # Extract the pitch observations of this slice
        slice_pitches = pitch_data_slices[slc]

        # Initialize arrays to hold pitch contour information for the slice
        slice_times = np.empty(0)
        slice_freqs = np.empty(0)
        slice_sources = np.empty(0, dtype=int)

        # Loop through the pitch observations
        for pitch in slice_pitches:
            # Extract the time, pitch and note index
            time = pitch.time
            freq = pitch.value['frequency']
            note_idx = pitch.value['index'] + index_offset

            if freq != 0 and pitch.value['voiced']:
                # Convert frequency from Hertz to MIDI
                freq = librosa.hz_to_midi(freq)
            else:
                # Represent unvoiced frequencies as zero
                freq = 0

            # Add pitch observation information to the respective arrays
            slice_times = np.append(slice_times, time)
            slice_freqs = np.append(slice_freqs, freq)
            slice_sources = np.append(slice_sources, note_idx)

        # Loop through all note sources
        for source in np.unique(slice_sources):
            # Obtain the time/pitch observations for the note
            source_times = slice_times[slice_sources == source]
            source_freqs = slice_freqs[slice_sources == source]

            # Make sure the observations correspond to a uniform time grid
            source_times, source_freqs = tools.time_series_to_uniform(times=source_times,
                                                                      values=source_freqs,
                                                                      duration=jam.file_metadata.duration)

            # Resample the pitch observations to the provided time grid
            source_freqs = resample_multipitch(source_times, source_freqs, times)

            # Represent the pitch observations for the note source as a 1D array
            pitch_observations = np.array([p if isinstance(p, float) else 0. for p in source_freqs])

            # Determine which observation index corresponds to the note onset
            onset_idx = np.where(pitch_observations)[0][0]

            # Remove leading/trailing empty pitch observations
            pitch_observations = np.trim_zeros(pitch_observations)

            # Check if there are any empty observations
            if tools.contains_empties_pitch_list(pitch_observations) and not suppress_warnings:
                # There are some gaps in the observations, throw a warning
                warnings.warn('Missing pitch observations within contour. ' +
                              'Will attempt to interpolate gaps.', category=RuntimeWarning)

            # Make sure any missing observations are filled in
            pitch_observations = tools.interpolate_gaps(pitch_observations)

            # Create a new entry for the note and the extracted pitch list
            grouping[source] = [PitchContour(pitch_observations, onset_idx)]

        # Extract the pitches and onset times of the notes within this slice
        slice_note_pitches = [note.value for note in note_data_slices[slc]]
        slice_note_onsets = [note.time for note in note_data_slices[slc]]

        # Update the array of note pitches
        note_pitches = np.append(note_pitches, np.array(slice_note_pitches))
        # Update the list of onset times
        note_onset_times += slice_note_onsets

        # Increment the note index offset by the amount of notes added
        index_offset += len(slice_note_onsets)

    # Determine the ordering of the notes
    note_order = np.argsort(note_onset_times)

    # Sort the onset times of notes across all slices
    note_sorting_idcs = np.argsort(note_order)

    # Update the dictionary keys to reflect the sorting
    grouping = dict(sorted(zip(note_sorting_idcs, grouping.values())))

    # Update the order of the note pitches based on the sorting
    note_pitches = note_pitches[note_order]

    return note_pitches, grouping


def extract_continuous_multi_pitch_jams(jam, times, profile, index=None, **kwargs):
    """
    Extract discretized and relative multi pitch activations
    from JAMS data containing grouped note and pitch annotations.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    times : ndarray (N)
      Frame times in seconds
      N - number of time samples
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    index : int or None (Optional)
      Specific index to isolate
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

    # Determine the dimensionality for the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(times)

    # Initialize two empty multi pitch arrays for relative and adjusted multi pitch data
    relative_multi_pitch = np.zeros((num_pitches, num_frames))
    adjusted_multi_pitch = np.zeros((num_pitches, num_frames))

    # Obtain a note-contour grouping by index
    pitches, grouping = get_note_contour_grouping_by_index(jam, times, index, **kwargs)

    # Round note pitches to nearest semitone and subtract the lowest
    # supported note of the instrument to obtain pitch indices
    pitch_idcs = np.round(pitches - profile.low).astype(tools.INT)

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

    return relative_multi_pitch, adjusted_multi_pitch


def extract_stacked_continuous_multi_pitch_jams(jam, times, profile, **kwargs):
    """
    Extract a stack of discretized and relative multi pitch activations
    from JAMS data containing grouped note and pitch annotations.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    times : ndarray (N)
      Frame times in seconds
      N - number of time samples
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

    # Loop through the slices of the collections
    for i in range(len(jam.annotations[tools.JAMS_NOTE_MIDI])):
        # Obtain the continuous multi pitch arrays for the notes in this slice
        relative_multi_pitch, \
            adjusted_multi_pitch = extract_continuous_multi_pitch_jams(jam, times, profile, i, **kwargs)

        # Add the multi pitch arrays to their respective stacks
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(relative_multi_pitch))
        stacked_adjusted_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(adjusted_multi_pitch))

    # Collapse the lists into arrays
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)
    stacked_adjusted_multi_pitch = np.concatenate(stacked_adjusted_multi_pitch)

    return stacked_relative_multi_pitch, stacked_adjusted_multi_pitch


def get_note_contour_grouping_by_interval(notes, pitch_list, suppress_warnings=True):
    """
    Associate pitch contours in a pitch list with a collection of notes, based off of
    the observations which occur within the note intervals. This function does support
    polyphonic data and should function robustly under most circumstances, as long the
    note and pitch contour data provided is tightly aligned with no same-pitch notes
    played in unison or overlapping pitch contours with significant deviation from the
    nominal pitches of their note sources.

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to (non-overlapping) notes
      (K - number of notes)
    pitch_list : tuple (_times, pitch_list)
      pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to (non-overlapping) notes
      _times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    suppress_warnings : bool
      Whether to ignore warning messages

    Returns
    ----------
    grouping : (note, [contours]) pairs : dict
      Dictionary containing (note_idx -> [PitchContour]) pairs
    """

    # Unpack the note attributes
    pitches, intervals = notes

    # Unpack the pitch list attributes
    _times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    # Check if there is any overlap within the streams
    if (tools.detect_overlap_notes(intervals) or
        tools.detect_overlap_pitch_list(pitch_list)) and not suppress_warnings:
        warnings.warn('Overlapping streams were provided. Will attempt ' +
                      'to infer note-pitch groupings.', category=RuntimeWarning)

    # Determine the dimensionality for the multi pitch array
    num_frames = len(_times)

    # Initialize a dictionary to hold (note, [contours]) pairs
    grouping = dict()

    # Make sure the pitch list is not empty
    if num_frames:
        # Estimate the duration of the track (for bounding note offsets)
        _times = np.append(_times, _times[-1] + tools.estimate_hop_length(_times))

        # Remove notes with out-of-bounds intervals
        pitches, intervals = tools.filter_notes(pitches, intervals,
                                                min_time=np.min(_times),
                                                max_time=np.max(_times))

        # Determine how many notes were provided
        num_notes = len(pitches)

        # Duplicate the array of times for each note and stack along a new axis
        _times_broadcast = np.concatenate([[_times]] * max(1, num_notes), axis=0)

        # Determine the frame where each note begins and ends
        onset_idcs = np.argmin((_times_broadcast <= intervals[..., :1]), axis=1) - 1
        offset_idcs = np.argmin((_times_broadcast <= intervals[..., 1:]), axis=1) - 1

        # Clip all onsets/offsets at first/last frame - these will end up
        # at -1 from previous operation if they occurred beyond boundaries
        onset_idcs[onset_idcs == -1], offset_idcs[offset_idcs == -1] = 0, num_frames - 1

        # Loop through each note
        for i in range(num_notes):
            # Keep track of adjusted note boundaries without modifying original values
            adjusted_onset, adjusted_offset = onset_idcs[i], offset_idcs[i]

            # Adjust the onset index to the first frame with non-empty pitch observations
            while not len(pitch_list[adjusted_onset]) and adjusted_onset <= adjusted_offset:
                adjusted_onset += 1

            # Adjust the offset index to the last frame with non-empty pitch observations
            while not len(pitch_list[adjusted_offset]) and adjusted_offset >= adjusted_onset:
                adjusted_offset -= 1

            # Check that there are non-empty pitch observations
            if adjusted_onset <= adjusted_offset:
                # Extract the (cropped) pitch observations within the note interval
                pitch_observations = pitch_list[adjusted_onset : adjusted_offset + 1]
            else:
                if not suppress_warnings:
                    # TODO - occurs quite frequently for notes with small
                    #        duration if the pitch list is undersampled.
                    # There are no non-empty pitch observations, throw a warning
                    warnings.warn('No pitch observations occur within the note interval. ' +
                                  'Inserting average pitch of note instead.', category=RuntimeWarning)
                # Reset the interval to the original note boundaries
                adjusted_onset, adjusted_offset = onset_idcs[i], offset_idcs[i]
                # Populate the frames with the average pitch of the note
                pitch_observations = [np.array([pitches[i]])] * (adjusted_offset + 1 - adjusted_onset)

            # Check if there are any empty observations remaining
            if tools.contains_empties_pitch_list(pitch_observations) and not suppress_warnings:
                # There are some gaps in the observations, throw a warning
                warnings.warn('Missing pitch observations within note interval. ' +
                              'Will attempt to interpolate gaps.', category=RuntimeWarning)

            # Convert the cropped pitch list to an array of monophonic pitches, choosing
            # the pitch closest to the nominal value of the note if a frame is polyphonic
            pitch_observations = np.array([p[np.argmin(np.abs(p - pitches[i]))]
                                           if len(p) else 0. for p in pitch_observations])

            # Interpolate between gaps in pitch observations
            pitch_observations = tools.interpolate_gaps(pitch_observations)

            # Create a new entry for the note and the extracted pitch list
            grouping[i] = [PitchContour(pitch_observations, adjusted_onset)]

    return grouping


def get_rotarized_relative_multi_pitch(relative_multi_pitch, adjusted_multi_pitch=None):
    """
    Compute pitch deviations at all anchoring points, assuming
    monophony, for each frame of a relative multi pitch array.

    Parameters
    ----------
    relative_multi_pitch : ndarray (... x F x T)
      Array of anchored pitch deviations
      F - number of discrete pitches
      T - number of frames
    adjusted_multi_pitch : ndarray (... x F x T) (optional)
      Discrete pitch activation map aligned with pitch contours
      F - number of discrete pitches
      T - number of frames

    Returns
    ----------
    relative_multi_pitch : ndarray (... x F x T)
      Array of rotarized pitch deviations
      F - number of discrete pitches
      T - number of frames
    """

    if adjusted_multi_pitch is None:
        # Default the multi pitch activations to non-zero pitch deviations
        adjusted_multi_pitch = relative_multi_pitch != 0

    # Determine the number of pitches supported
    num_pitches = relative_multi_pitch.shape[-2]

    # Determine where there are active pitches
    active_idcs = np.sum(adjusted_multi_pitch, axis=-2) > 0

    # Obtain the multi pitch activity at the active frames
    active_frames = np.swapaxes(adjusted_multi_pitch, -1, -2)[active_idcs]

    # Determine where the pitch activity is anchored (assuming monophony)
    anchors = np.argmax(active_frames, axis=-1)

    # Temporarily switch the pitch/frame axes
    relative_multi_pitch = np.swapaxes(relative_multi_pitch, -1, -2)

    # Extract the frame-level deviations
    frame_deviations = relative_multi_pitch[active_idcs]

    # Obtain the deviations at the anchoring points
    deviations = frame_deviations[np.arange(len(anchors)), anchors]

    # Obtain pitch offsets for all frames w.r.t. the anchoring points
    pitch_offsets = np.subtract.outer(anchors, np.arange(num_pitches))

    # Compute the rotarized deviations for all frames with pitch activity
    rotarized_deviations = np.expand_dims(deviations, axis=-1) + pitch_offsets

    # Insert the rotarized deviations
    relative_multi_pitch[np.where(active_idcs)] = rotarized_deviations

    # Switch the pitch/frame axes back
    relative_multi_pitch = np.swapaxes(relative_multi_pitch, -1, -2)

    return relative_multi_pitch


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
    This function assumes that all pitch lists are relative to the same timing grid.

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
        _, pitch_list = stacked_pitch_list[slc]
        # Obtain a relative pitch deviation map for the pitch list and add to the list
        relative_multi_pitch = pitch_list_to_relative_multi_pitch(pitch_list, profile)
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(relative_multi_pitch))

    # Collapse the list into an array
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)

    return stacked_relative_multi_pitch
