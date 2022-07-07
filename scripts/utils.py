# My imports
import amt_tools.tools as tools

# Regular imports
from mir_eval.multipitch import resample_multipitch

import numpy as np
import warnings


def detect_overlap_notes(intervals, decimals=3):
    """
    Determine if a set of intervals contains any overlap.

    Parameters
    ----------
    intervals : ndarray (N x 2)
      Array of onset-offset time pairs
      N - number of notes
    decimals : int (Optional - millisecond by default)
      Decimal resolution for timing comparison

    Returns
    ----------
    overlap : bool
      Whether any intervals overlap
    """

    # Make sure the intervals are sorted by onset (abusing this function slightly)
    intervals = tools.sort_batched_notes(intervals, by=0)
    # Check if any onsets occur before the offset of a previous interval
    overlap = np.sum(np.round(np.diff(intervals).flatten(), decimals) < 0) > 0

    return overlap


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
    overlap = np.sum(tools.get_active_pitch_count(pitch_list) > 1) > 0

    return overlap


class PitchContour(object):
    """
    Simple class representing a collection of pitch
    observations corresponding to a single pitch contour.
    """

    def __init__(self, onset_idx, onset_pitch):
        """
        Initialize a pitch contour object.

        Parameters
        ----------
        onset_idx : int
          Index within a pitch list where the contour begins
        onset_pitch : float
          Observed pitch at the onset frame of the contour
        """

        # Initialize the contour's interval indices
        self.onset_idx = onset_idx
        self.offset_idx = onset_idx

        # Initialize an array to hold the pitch observations across the contour
        self.pitch_observations = np.array([onset_pitch])

    def append_observation(self, pitch):
        """
        Extend the tracked pitch contour with a new pitch observation.

        Parameters
        ----------
        pitch : float
          Observed pitch at the next frame of the contour
        """

        # Add the new pitch to the collection of observations
        self.pitch_observations = np.append(self.pitch_observations, pitch)

        # Increment the offset index
        self.offset_idx += 1

    def get_last_observation(self):
        """
        Helper function to access the most recent pitch observation.

        Returns
        ----------
        last_observation : float
          Observed pitch at the most recent frame
        """

        # Extract the most recent observation from the tracked pitches
        last_observation = self.pitch_observations[-1]

        return last_observation


class ContourTracker(object):
    """
    Class to maintain state while parsing pitch lists to track pitch contours.
    """

    def __init__(self):
        """
        Initialize a contour tracker object.
        """

        # Initialize empty lists to hold all contours
        # as well as the indices of active contours
        self.contours = list()
        self.active_idcs = list()

    def track_new_contour(self, onset_idx, onset_pitch):
        """
        Start tracking a new pitch contour.

        Parameters
        ----------
        onset_idx : int
          Index within a pitch list where the contour begins
        onset_pitch : float
          Observed pitch at the onset frame of the contour
        """

        # Initialize a new contour for the pitch
        self.contours += [PitchContour(onset_idx, onset_pitch)]
        # Mark the index of the contour as active
        self.active_idcs += [len(self.contours) - 1]

    def get_active_contours(self):
        """
        Obtain references to all active pitch contours.

        Returns
        ----------
        active_pitches : list of PitchContour
          Currently active pitch contours
        """

        # Group the contour objects of active contours in a list
        active_contours = [self.contours[idx] for idx in self.active_idcs]

        return active_contours

    def get_active_pitches(self):
        """
        Obtain the most recent pitch observed for all active pitch contours.

        Returns
        ----------
        active_pitches : ndarray
          Last pitches observed for each contour
        """

        # Obtain the most recent pitch observation for all active contours
        active_pitches = np.array([c.get_last_observation() for c in self.get_active_contours()])

        return active_pitches

    def parse_pitch_list(self, pitch_list, tolerance=None):
        """
        Parse a pitch list and track the activity of all pitch contours, which are inferred via
        separation by empty frames or deviations in pitch above a specified tolerance. The function
        should work well for polyphonic pitch contours unless contours cross or get very close to
        one another. In the polyphonic case, a matching strategy is employed to determine when
        pitch contours become active and inactive.

        Parameters
        ----------
        pitch_list : list of ndarray (N x [...])
          Frame-level observations detailing active MIDI pitches
          N - number of frames
        tolerance : float (Optional)
          Pitch difference tolerated across adjacent frames of a single contour
        """

        # Loop through all observations in the pitch list
        for i, observations in enumerate(pitch_list):
            # Obtain the active pitches for comparison
            active_pitches = self.get_active_pitches()

            # Determine the number of active pitches and observations
            num_ap, num_ob = len(active_pitches), len(observations)

            # Initialize arrays to keep track of assignments
            assignment_ap = np.array([-1] * num_ap)
            assignment_ob = np.array([-1] * num_ob)

            # Repeat the active pitches for each observation
            expanded_pitches = np.concatenate([[active_pitches]] * max(1, num_ob), axis=0)
            # Compute the magnitude difference of each pitch w.r.t. each observation
            magnitude_difference = np.abs(expanded_pitches - np.expand_dims(observations, axis=-1))

            if num_ap and num_ob:
                # Iterate in case first choice is not granted
                while np.sum(assignment_ob != -1) < min(num_ap, num_ob):
                    # Don't consider columns and rows that have already been matched
                    magnitude_difference[:, assignment_ap != -1] = np.inf
                    magnitude_difference[assignment_ob != -1] = np.inf

                    # Determine the pairs with minimum difference from the active pitches for both views
                    best_mapping_ap = np.argmin(magnitude_difference, axis=0)
                    best_mapping_ob = np.argmin(magnitude_difference, axis=1)

                    # Determine which indices represent true matches for both views
                    matches_ap = best_mapping_ob[best_mapping_ap] == np.arange(len(best_mapping_ap))
                    matches_ob = best_mapping_ap[best_mapping_ob] == np.arange(len(best_mapping_ob))

                    # Don't reassign anything that has already been assigned
                    matches_ap[assignment_ap != -1] = False
                    matches_ob[assignment_ob != -1] = False

                    # Assign matches to their respective indices for both views
                    assignment_ap[matches_ap] = best_mapping_ap[matches_ap]
                    assignment_ob[matches_ob] = best_mapping_ob[matches_ob]

            # Create a copy of the active indices to iterate through
            active_idcs_copy = self.active_idcs.copy()

            # Loop through active contours
            for k in range(num_ap):
                # Obtain the index of the contour
                idx = active_idcs_copy[k]
                # Check if the contour has no match
                if assignment_ap[k] == -1:
                    # Mark the contour as being inactive
                    self.active_idcs.remove(idx)
                else:
                    # Determine which pitch was matched to this contour
                    matched_pitch = observations[assignment_ap[k]]
                    # Make sure the matched pitch is within the specified tolerance
                    if tolerance is not None and \
                            np.abs(self.contours[idx].get_last_observation() - matched_pitch) <= tolerance:
                        # Append the matched pitch to the contour
                        self.contours[idx].append_observation(matched_pitch)
                    else:
                        # Mark the contour as being inactive
                        self.active_idcs.remove(idx)
                        # Start tracking a new contour instead
                        self.track_new_contour(i, matched_pitch)

            # Loop through pitch observations with no matches
            for pitch in observations[assignment_ob == -1]:
                # Start tracking a new contour
                self.track_new_contour(i, pitch)

    def get_contour_intervals(self):
        """
        Helper function to get the intervals of all tracked pitch contours.

        Returns
        ----------
        intervals : ndarray (N x 2)
          Array of onset-offset index pairs corresponding to pitch contours
        """

        # Group the onset and offset indices of all tracked contours
        intervals = np.array([[c.onset_idx, c.offset_idx] for c in self.contours])

        return intervals

    def get_contour_means(self):
        """
        Helper function to get the average pitch of all tracked pitch contours.

        Returns
        ----------
        means : ndarray
          Array of average pitches corresponding to pitch contours
        """

        # Compute the average pitch for all tracked contours
        means = np.array([np.mean(c.pitch_observations) for c in self.contours])

        return means


def streams_to_continuous_multi_pitch_by_interval(notes, pitch_list, profile, semitone_width=0.5, times=None):
    """
    Represent note streams as anchored pitch deviations within a multi pitch array, along
    with an accompanying multi pitch array adjusted in accordance with the pitch list (so
    0 deviation activations can be clearly interpreted). This function is intended for use
    with strictly monophonic data, i.e. with no overlap in the note intervals AND no
    overlapping pitch contours in the pitch list. However, it does support polyphonic data
    and should function robustly under most circumstances, as long the note and pitch contour
    data provided is tightly aligned with no same-pitch notes played in unison or overlapping
    pitch contours with significant deviation from the nominal pitches of their note sources.

    TODO - validate with polyphonic data

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
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    semitone_width : float
      Amount of deviation from nominal pitch supported
    times : ndarray (L) (Optional)
      Array of alternate times for optional resampling of pitch list
      L - number of time samples (frames)

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
    _times, pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    pitch_list = tools.clean_pitch_list(pitch_list)

    if times is not None:
        # If times given, resample the pitch list here
        pitch_list = resample_multipitch(_times, pitch_list, times)
    else:
        # Use the extracted times
        times = _times

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
        while not len(pitch_list[adjusted_onset]) and adjusted_onset <= adjusted_offset:
            adjusted_onset += 1

        # Adjust the offset index to the last frame with non-empty pitch observations
        while not len(pitch_list[adjusted_offset]) and adjusted_offset >= adjusted_onset:
            adjusted_offset -= 1

        # Check that there are non-empty pitch observations
        if adjusted_onset <= adjusted_offset and len(pitch_list[adjusted_onset]):
            # Extract the (cropped) pitch observations within the note interval
            pitch_observations = pitch_list[adjusted_onset : adjusted_offset + 1]
        else:
            # There are no non-empty pitch observations, throw a warning
            warnings.warn('No pitch observations occur within the note interval. ' +
                          'Inserting average pitch of note instead.', category=RuntimeWarning)
            # Reset the interval to the original note boundaries
            adjusted_onset, adjusted_offset = onset_idcs[i], offset_idcs[i]
            # Populate the frames with the average pitch of the note
            pitch_observations = [np.array([pitches[i]])] * (adjusted_offset + 1 - adjusted_onset)

        # Populate the multi pitch array with adjusted activations for the note
        adjusted_multi_pitch[pitch_idcs[i], adjusted_onset: adjusted_offset + 1] = 1

        # Check if there are any empty observations remaining
        if tools.contains_empties_pitch_list(pitch_observations):
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


def streams_to_continuous_multi_pitch_by_cluster(notes, pitch_list, profile, semitone_width=0.5,
                                                 times=None, minimum_contour_duration=None,
                                                 combine_associated_contours=True):
    """
    Associate pitch contours in a pitch list with a collection of notes, then obtain
    discretized and relative multi pitch information after adjusting the provided
    note intervals based off of the intervals of the pitch contours associated with
    each note.

    TODO - validate with polyphonic data

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to (non-overlapping) notes
      (K - number of notes)
    pitch_list : tuple (_times, _pitch_list)
      _pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to (non-overlapping) notes
      _times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    profile : InstrumentProfile (instrument.py)
      Instrument profile detailing experimental setup
    semitone_width : float
      Amount of deviation from nominal pitch supported
    times : ndarray (L) (Optional)
      Array of alternate times for optional resampling of pitch list
      L - number of time samples (frames)
    minimum_contour_duration : float (Optional)
      Minimum amount of time in milliseconds a contour should span to be considered
    combine_associated_contours : bool
      Whether to construct a single interval from all contours assigned to the same note

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

    # Unpack the note attributes
    pitches, intervals = notes

    # Unpack the pitch list attributes
    _times, _pitch_list = pitch_list
    # Make sure there are no null observations in the pitch list
    _pitch_list = tools.clean_pitch_list(_pitch_list)

    # Initialize a new pitch contour tracker
    tracker = ContourTracker(tolerance=0.5)
    # Track the contours within the provided pitch list
    tracker.parse_pitch_list(_pitch_list)
    # Obtain intervals and averages for pitch observation clusters (contours)
    contour_intervals, contour_means = tracker.get_contour_intervals(), tracker.get_contour_means()
    # Extract the corresponding times of the inferred contours
    contour_times = _times[contour_intervals.flatten()].reshape(contour_intervals.shape)

    # Compute the duration of each inferred contour
    contour_durations = np.diff(contour_times).squeeze(-1)

    if minimum_contour_duration is not None:
        # Remove contours with intervals smalled than specified threshold
        contour_times = contour_times[contour_durations > minimum_contour_duration * 1E-3]

    # Determine the total number of clusters (contours)
    num_contours = contour_times.shape[0]

    # Initialize an array for the assignment of each contour to a note
    assignment = np.array([-1] * len(num_contours))

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

    # Count the total number of notes provided
    num_notes = len(pitches)
    # Determine which contours where assigned a note index
    assigned_notes = np.unique(assignment)

    if len(np.setdiff1d(np.arange(num_notes), assigned_notes)):
        # Some notes are not assigned to any pitch contours
        warnings.warn('Some notes are not accounted for ' +
                      'by any pitch contours.', category=RuntimeWarning)

    if len(assigned_notes) < len(assignment):
        # More than one pitch contour occurs maximally within a single note
        warnings.warn('Multiple pitch contours assigned ' +
                      'to the same note.', category=RuntimeWarning)

    # TODO - could also check for pitch mismatch here and add a new
    #        entry for a contour (parameter attempt_corrections=False)
    # TODO - compute average pitch for all contours
    # TODO - compare this value with the pitch of the assigned note
    # TODO - throw warning if the difference is above a threshold (semitone_width?)
    # TODO - assign computed average pitch instead (create new note entry, insert the new length of the notes as the assignment)

    if combine_associated_contours:
        # Initialize empty arrays for combined contours
        _pitches, _intervals = np.empty(0), np.empty((0, 2))
        # Loop through all notes with assigned contours
        for note_idx in assigned_notes:
            # Add the pitch of the note assigned to the contours
            _pitches = np.append(_pitches, pitches[note_idx])
            # Obtain the intervals of the contours assigned to the note
            _contour_times = contour_times[assignment == note_idx]
            # Determine the earliest onset and latest offset among all contours
            min_t, max_t = np.min(_contour_times[:, 0]), np.max(_contour_times[:, 1])
            # Combine contours with the same note assignment
            _intervals = np.append(_intervals, np.array([[min_t, max_t]]), axis=0)
    else:
        # Replace note intervals with contour intervals, ignoring contours without an assignment
        _pitches, _intervals = pitches[assignment[assignment != -1]], contour_times[assignment != -1]

    # Represent contours as note groups
    contours = _pitches, _intervals

    # Parse the inferred contour intervals to obtain the multi pitch arrays
    relative_multi_pitch, \
        adjusted_multi_pitch = streams_to_continuous_multi_pitch_by_interval(notes=contours,
                                                                             pitch_list=pitch_list,
                                                                             profile=profile,
                                                                             semitone_width=semitone_width,
                                                                             times=times)

    return relative_multi_pitch, adjusted_multi_pitch


def stacked_streams_to_stacked_continuous_multi_pitch(stacked_notes, stacked_pitch_list, profile,
                                                      semitone_width=0.5, times=None):
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
    semitone_width : float
      Amount of deviation from nominal pitch supported
    times : ndarray (L) (Optional)
      Array of alternate times for optional resampling of pitch lists
      L - number of time samples (frames)

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
        # Obtain the note stream multi pitch arrays for the notes in this slice
        relative_multi_pitch, \
            adjusted_multi_pitch = streams_to_continuous_multi_pitch_by_cluster(stacked_notes[key_n],
                                                                                stacked_pitch_list[key_pl],
                                                                                profile=profile,
                                                                                semitone_width=semitone_width,
                                                                                times=times,
                                                                                minimum_contour_duration=100,
                                                                                combine_associated_contours=True)
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


def fill_empties(pitch_list):
    """
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
    """

    # Add a null frequency to empty observations
    pitch_list = [p if len(p) else np.array([0.]) for p in pitch_list]

    return pitch_list


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
