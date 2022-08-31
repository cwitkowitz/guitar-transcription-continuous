# My imports
import amt_tools.tools as tools

# Regular imports
from math import floor, ceil

import numpy as np


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

        # Make sure there are no null observations in the pitch list
        pitch_list = tools.clean_pitch_list(pitch_list)

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
        region_observations = self.discard_null_observations(region_observations)

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
        observations = self.discard_null_observations(self.pitch_observations)

        # Obtain the specified percentile of the remaining observations
        value = np.percentile(observations, 100 * percentile) if len(observations) else np.nan

        return value


class ContourTracker(object):
    """
    Class to maintain state while parsing pitch lists to track pitch contours.
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
