# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet, TranscriptionDataset

import amt_tools.tools as tools


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
            stacked_notes = tools.load_stacked_notes_jams(jams_path)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(data[tools.KEY_AUDIO])

            # Convert the string-wise notes into a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into a single representation
            data[tools.KEY_MULTIPITCH] = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Convert the stacked multi pitch array into tablature
            data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Obtain the notes from the multi pitch array
            pitches, intervals = tools.multi_pitch_to_notes(data[tools.KEY_MULTIPITCH], times, self.profile)
            # Batch the notes
            batched_notes = tools.notes_to_batched_notes(pitches, intervals)
            # Add the notes to the dictionary
            data[tools.KEY_NOTES] = batched_notes

            # TODO - remove
            # Obtain the onsets for each string from the stacked notes
            #stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)
            # Convert the onsets to tablature format
            #data[tools.KEY_ONSETS] = tools.stacked_multi_pitch_to_tablature(stacked_onsets, self.profile)

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                keys = (tools.KEY_FS, tools.KEY_AUDIO, tools.KEY_TABLATURE, tools.KEY_MULTIPITCH, tools.KEY_NOTES)
                tools.save_pack_npz(gt_path, keys, data[tools.KEY_FS], data[tools.KEY_AUDIO],
                                    data[tools.KEY_TABLATURE], data[tools.KEY_MULTIPITCH], data[tools.KEY_NOTES])

        return data
