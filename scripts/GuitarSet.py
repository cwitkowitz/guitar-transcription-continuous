# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet, TranscriptionDataset

import amt_tools.tools as tools
import constants

# Regular imports
import numpy as np
import warnings


"""
def notes_to_relative_multi_pitch(pitches, intervals, times, profile, include_offsets=True):
    "/""
    TODO
    "/""

    # Determine the dimensionality of the multi pitch array
    num_pitches = profile.get_range_len()
    num_frames = len(times)

    # Initialize an empty multi pitch array
    relative_multi_pitch = np.zeros((num_pitches, num_frames))

    # Round to nearest semitone and subtract the lowest
    # note of the instrument to obtain relative pitches
    pitch_idcs = np.round(pitches - profile.low).astype(tools.UINT)

    # Subtract the absolute pitch from the rounded pitch
    relative_pitches = pitches - np.round(pitches)

    # Determine if and where there is underflow or overflow
    valid_idcs = np.logical_and((pitch_idcs >= 0), (pitch_idcs < num_pitches))

    # Remove any invalid (out-of-bounds) pitches
    relative_pitches, pitch_idcs, intervals = relative_pitches[valid_idcs], pitch_idcs[valid_idcs], intervals[valid_idcs]

    # Duplicate the array of times for each note and stack along a new axis
    times = np.concatenate([[times]] * max(1, len(pitch_idcs)), axis=0)

    # Determine the frame where each note begins and ends
    onsets = np.argmin((times <= intervals[..., :1]), axis=1) - 1
    offsets = np.argmin((times < intervals[..., 1:]), axis=1) - 1

    # Clip all offsets at last frame - they will end up at -1 from
    # previous operation if they occurred beyond last frame time
    offsets[offsets == -1] = num_frames - 1

    # Loop through each note
    for i in range(len(pitch_idcs)):
        # Populate the multi pitch array with activations for the note
        relative_multi_pitch[pitch_idcs[i], onsets[i] : offsets[i] + int(include_offsets)] = relative_pitches[i]

    return relative_multi_pitch


def stacked_notes_to_stacked_relative_multi_pitch(stacked_notes, times, profile, include_offsets=True):
    "/""
    TODO
    "/""

    # Initialize an empty list to hold the multi pitch arrays
    stacked_relative_multi_pitch = list()

    # Loop through the slices of notes
    for slc in stacked_notes.keys():
        # Get the pitches and intervals from the slice
        pitches, intervals = stacked_notes[slc]
        # Convert to multi pitch and add to the list
        slice_relative_multi_pitch = notes_to_relative_multi_pitch(pitches, intervals, times, profile, include_offsets)
        stacked_relative_multi_pitch.append(tools.multi_pitch_to_stacked_multi_pitch(slice_relative_multi_pitch))

    # Collapse the list into an array
    stacked_relative_multi_pitch = np.concatenate(stacked_relative_multi_pitch)

    return stacked_relative_multi_pitch
"""


def pitch_list_to_relative_multi_pitch(pitch_list, profile):
    """
    TODO
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
    TODO
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


class GuitarSet(GuitarSet):
    """
    Simple wrapper to additionally include notes in ground-truth.
    """

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          Name of the track to load

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = TranscriptionDataset.load(self, track)

        # If the track data is being instantiated, it will not have the 'audio' key
        # TODO - most of this is redundant, but I'm not sure if I want to save twice
        if tools.KEY_AUDIO not in data.keys():
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            data[tools.KEY_AUDIO], data[tools.KEY_FS] = tools.load_normalize_audio(wav_path, self.sample_rate)

            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the notes by string from the JAMS file
            #stacked_notes = tools.load_stacked_notes_jams(jams_path)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(data[tools.KEY_AUDIO])

            # Convert the string-wise notes into a stacked multi pitch array
            #stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            #import mirdata
            #dataset_mirdata = mirdata.initialize('guitarset', self.base_dir)
            #pitch_anno_mirdata = dataset_mirdata.load_pitch_contour(jams_path, 3)

            # TODO - comment
            stacked_pitch_list = tools.stacked_pitch_list_to_midi(tools.load_stacked_pitch_list_jams(jams_path, times))

            # TODO - comment
            # TODO - seems to shift onset by 1 frame sometimes (likely due to resampling)
            stacked_multi_pitch = tools.stacked_pitch_list_to_stacked_multi_pitch(stacked_pitch_list, self.profile)

            # Convert the stacked multi pitch array into a single representation
            data[tools.KEY_MULTIPITCH] = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Convert the stacked multi pitch array into tablature
            #data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Obtain the notes from the multi pitch array
            pitches, intervals = tools.multi_pitch_to_notes(data[tools.KEY_MULTIPITCH], times, self.profile)
            # Batch the notes
            batched_notes = tools.notes_to_batched_notes(pitches, intervals)
            # Add the notes to the dictionary
            data[tools.KEY_NOTES] = batched_notes

            # TODO - this is used to analyze frame-level signal power - remove
            # Obtain the onsets for each string from the stacked notes
            #stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)
            # Convert the onsets to tablature format
            #data[tools.KEY_ONSETS] = tools.stacked_multi_pitch_to_tablature(stacked_onsets, self.profile)

            # TODO - comment
            stacked_relative_multi_pitch = stacked_pitch_list_to_stacked_relative_multi_pitch(stacked_pitch_list, self.profile)

            # TODO - comment
            relative_multi_pitch_sum = np.sum(stacked_relative_multi_pitch, axis=-3)
            relative_multi_pitch_count = np.sum(stacked_relative_multi_pitch != 0, axis=-3)
            relative_multi_pitch_count[relative_multi_pitch_count == 0] = 1

            relative_multi_pitch = relative_multi_pitch_sum / relative_multi_pitch_count

            # TODO - comment
            data[constants.KEY_MULTIPITCH_REL] = relative_multi_pitch

            # TODO - actually, want to get standard/relative multipitch from another function
            #        which takes both notes and pitch list so we can associate pitches we notes,
            #        which is necessary if, for example, the pitch contour extends beyond the
            #        standard semitone region of a note - def stacked_note_streams_to_pitch_activations()

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                keys = (tools.KEY_FS, tools.KEY_AUDIO,
                        #tools.KEY_TABLATURE,
                        tools.KEY_MULTIPITCH,
                        constants.KEY_MULTIPITCH_REL, tools.KEY_NOTES)
                tools.save_pack_npz(gt_path, keys, data[tools.KEY_FS], data[tools.KEY_AUDIO],
                                    #data[tools.KEY_TABLATURE],
                                    data[tools.KEY_MULTIPITCH],
                                    data[constants.KEY_MULTIPITCH_REL], data[tools.KEY_NOTES])

        return data
