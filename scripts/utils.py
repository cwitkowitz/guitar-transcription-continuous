# My imports
import amt_tools.tools as tools

# Regular imports
from math import floor, ceil

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

    def __init__(self, pitch, onset_idx):
        """
        Initialize a pitch contour object.

        Parameters
        ----------
        pitch : float or ndarray
          Observed pitch at the onset frame of the contour or array of pitch observations
        onset_idx : int
          Index within a pitch list where the contour begins
        """

        # Initialize the contour's interval indices
        self.onset_idx = onset_idx
        self.offset_idx = onset_idx

        if isinstance(pitch, np.ndarray):
            # Use the provided array as the pitch observations
            self.pitch_observations = pitch
            # Update the offset index to reflect the size of the array
            self.offset_idx += (len(pitch) - 1)
        else:
            # Initialize an array to hold the pitch observations across the contour
            self.pitch_observations = np.array([pitch])

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

    def get_interval(self, times=None):
        """
        Helper function to obtain the currently tracked interval for the contour.

        Parameters
        ----------
        times : ndarray (Optional)
          Array of times to index

        Returns
        ----------
        interval : ndarray (2)
          Onset and offset corresponding to pitch contour intervals or times
        """

        # Construct an array to hold the interval indices
        interval = np.array([self.onset_idx, self.offset_idx])

        if times is not None:
            # Index the array of times with the interval
            interval = times[interval]

        return interval

    def get_times(self, _times):
        """
        Helper function to obtain the times of the contour observations.

        Parameters
        ----------
        _times : ndarray
          Array of times to index

        Returns
        ----------
        times : ndarray (N)
          Array of times corresponding to pitch observations
          N - number of observations across the contour
        """

        # Index the provided times with the interval of the contour
        times = _times[np.arange(*self.get_interval() + [0, 1])]

        return times

    def get_pitch_list(self, _times=None):
        """
        Helper function to represent the contour data as a pitch list.

        Parameters
        ----------
        _times : ndarray or None (optional)
          Array of times to index if tuple form is desired

        Returns
        ----------
        times : ndarray (N) (if _times is not None)
          Array of times corresponding to pitch observations
          N - number of observations across the contour
        pitch_list : list of ndarray (N x [ndarray (1)])
          Frame-level observations detailing active pitches
          N - number of observations across the contour
        """

        # Represent the observations as a pitch list
        pitch_list = [np.array([p]) for p in self.pitch_observations]

        if tools.contains_empties_pitch_list(pitch_list):
            # TODO - tools.clean_pitch_list()? is it even necessary?
            print()

        if _times is not None:
            # Obtain times corresponding to the observations
            times = self.get_times(_times)
            # Represent the data as a tuple
            pitch_list = times, pitch_list

        return pitch_list

    @staticmethod
    def discard_null_observations(pitch_list):
        """
        Discard null pitch observations within frames.

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

        # Discard empty observations
        pitch_list = np.array([p for p in pitch_list if p])

        return pitch_list

    def get_region_average(self, percentage_start=0., percentage_stop=1.):
        """
        Compute the average pitch with a duration-dependent region of the contour.

        Parameters
        ----------
        percentage_start : float (0, 1)
          Percent of duration where region begins
        percentage_stop : float (0, 1)
          Percent of duration where region ends

        Returns
        ----------
        average : float
          Average pitch across the specified region
        """

        # Determine the number of pitch observations
        num_observations = len(self.pitch_observations)

        # Determine which indices correspond to the specified region
        idx_start = floor(percentage_start * num_observations)
        idx_stop = ceil(percentage_stop * num_observations)

        # Slice the specified region of the observations
        region_observations = self.pitch_observations[idx_start : idx_stop]

        # Make sure null observations do not influence the statistics
        #region_observations = self.discard_null_observations(region_observations)

        # Compute the mean of the remaining observations
        average = np.mean(region_observations)

        return average

    def get_pitch_percentile(self, percentile=0.):
        """
        Obtain the value at the specified percentile of pitch across the contour.

        Parameters
        ----------
        percentile : float (0, 1)
          Pitch percentile to obtain

        Returns
        ----------
        value : float
          Value at specified percentile
        """

        # Make sure null observations do not influence the statistics
        #observations = self.discard_null_observations(self.pitch_observations)
        observations = self.pitch_observations

        # Obtain the specified percentile of the remaining observations
        value = np.percentile(observations, 100 * percentile) if len(observations) else np.nan

        return value


class ContourTracker(object):
    """
    Class to maintain state while parsing pitch lists to track pitch contours.

    TODO - this is an excellent place to resample a pitch list that should avoid
           the pitfalls of undersampling; after tracking contours, can linearly
           interpolate them to obtain pitch observations at arbitrary times. Then,
           can take a collection of arbitrary times and attempt to place pitches
           that occur within frame times rather than pitches that occur at frame
           times!!
    """

    def __init__(self):
        """
        Initialize a contour tracker object.
        """

        # Initialize fields for the tracker
        self.contours = None
        self.active_idcs = None

        # Reset tracker state
        self.reset_tracker()

    def reset_tracker(self):
        """
        Reset tracker state.
        """

        # Initialize empty lists to hold all contours
        # as well as the indices of active contours
        self.contours = list()
        self.active_idcs = list()

    def get_num_contours(self):
        """
        Helper function to determine the number of contours currently tracked.

        Returns
        ----------
        num_contours : int
          Amount of tracked contours
        """

        num_contours = len(self.contours)

        return num_contours

    def filter_contours(self, idcs):
        """
        Discard all contours except those at the specified indices.

        Parameters
        ----------
        idcs : ndarray of bools
          Array of bools corresponding to contours to keep
        """

        # Remove contours marked False
        self.contours = [self.contours[i] for (i, keep) in enumerate(idcs) if keep]
        # Remove active indices corresponding to contours marked False
        self.active_idcs = [i for i in np.where(idcs)[0] if i in self.active_idcs]

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
        self.contours += [PitchContour(onset_pitch, onset_idx)]
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

    def get_contour_intervals(self, times=None):
        """
        Helper function to get the intervals of all tracked pitch contours.

        Parameters
        ----------
        times : ndarray (Optional)
          Array of times to index

        Returns
        ----------
        intervals : ndarray (N x 2)
          Array of onset-offset index or time pairs corresponding to pitch contours
        """

        # Group the onset and offset indices or times of all tracked contours
        intervals = np.array([c.get_interval(times) for c in self.contours]).reshape(-1, 2)

        return intervals

    def get_contour_averages(self, percentage_start=0., percentage_stop=1.):
        """
        Helper function to compute the average pitch with a
        duration-dependent region for all tracked pitch contours.

        Parameters
        ----------
        percentage_start : float (0, 1)
          Percent of duration where region begins
        percentage_stop : float (0, 1)
          Percent of duration where region ends

        Returns
        ----------
        averages : ndarray
          Array of average pitch across the specified region of each pitch contour
        """

        # Compute the average pitch for all tracked contours
        averages = np.array([c.get_region_average(percentage_start,
                                                  percentage_stop) for c in self.contours])

        return averages

    def get_contour_percentiles(self, percentile=0.10):
        """
        Helper function to obtain the specified percentile
        of pitch associated for all tracked pitch contours.

        Parameters
        ----------
        percentile : float (0, 1)
          Pitch percentile to obtain for each contour

        Returns
        ----------
        values : ndarray
          Values at specified percentile of each pitch contour
        """

        # Obtain the specified pitch percentile of all tracked contours
        values = np.array([c.get_pitch_percentile(percentile) for c in self.contours])

        return values


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
    if (detect_overlap_notes(intervals) or detect_overlap_pitch_list(pitch_list)) and not suppress_warnings:
        warnings.warn('Overlapping streams were provided. Will attempt ' +
                      'to infer note-pitch groupings.', category=RuntimeWarning)

    # Determine the dimensionality for the multi pitch array
    num_frames = len(_times)

    # Initialize a dictionary to hold (note, [contours]) pairs
    grouping = dict()

    # Make sure the pitch list is not empty
    if num_frames:
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


def get_note_contour_grouping_by_cluster(notes, pitch_list, semitone_width=0.5, stream_tolerance=1.0,
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
    semitone_width : float
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

        # Compute pitch proximity scores using exponential distribution
        pitch_proximities = np.exp(-1.5 * np.abs(pitches - avg))

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

    for i in np.where(magnitude_differences > semitone_width)[0]:
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


def streams_to_continuous_multi_pitch(notes, pitch_list, profile, times=None,
                                      semitone_width=0.5, suppress_warnings=True,
                                      **kwargs):
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
    semitone_width : float
      Amount of deviation from nominal pitch supported
    suppress_warnings : bool
      Whether to ignore warning messages
    **kwargs : N/A
      Arguments for grouping notes and contours

    Returns
    ----------
    grouping : (note, [contours]) pairs : dict
      Dictionary containing (note_idx -> [PitchContour]) pairs
    """

    # Unpack the note attributes, removing notes with out-of-bounds nominal pitch
    pitches, intervals = tools.filter_notes(*notes, profile, suppress_warnings)

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

            # Clip the deviations at the supported semitone width
            deviations = np.clip(deviations, a_min=-semitone_width, a_max=semitone_width)

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
      Arguments for converting to stream

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
        slice_pitch_list_ = continuous_multi_pitch_to_pitch_list(slice_discrete, slice_relative, profile)

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(tools.pitch_list_to_stacked_pitch_list(times, slice_pitch_list_, slc))

    return stacked_pitch_list


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
