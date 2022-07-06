# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet, TranscriptionDataset

import amt_tools.tools as tools
import constants
import utils


class GuitarSet(GuitarSet):
    """
    Simple wrapper to additionally include notes and relative pitch deviation in ground-truth.
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
        if tools.KEY_AUDIO not in data.keys():
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            data[tools.KEY_AUDIO], data[tools.KEY_FS] = tools.load_normalize_audio(wav_path, self.sample_rate)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(data[tools.KEY_AUDIO])

            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the notes by string from the JAMS file
            stacked_notes = tools.load_stacked_notes_jams(jams_path)

            # Represent the string-wise notes as a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Collapse the stacked multi pitch array into a single representation
            data[tools.KEY_MULTIPITCH] = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Convert the stacked multi pitch array into tablature
            #data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Load the string-wise pitch annotations from the JAMS file
            # TODO - add this to the ground-truth for evaluation of continuous pitch predictions?
            stacked_pitch_list = tools.load_stacked_pitch_list_jams(jams_path)
            stacked_pitch_list = tools.stacked_pitch_list_to_midi(stacked_pitch_list)

            # Obtain the relative pitch deviation of the contours anchored by string/fret
            stacked_relative_multi_pitch, stacked_adjusted_multi_pitch = \
                utils.stacked_streams_to_stacked_relative_multi_pitch(stacked_notes,
                                                                      stacked_pitch_list,
                                                                      self.profile,
                                                                      semitone_width=1.5)

            # Collapse the stacked relative multi pitch array into a single representation
            data[constants.KEY_MULTIPITCH_REL] = \
                utils.stacked_relative_multi_pitch_to_relative_multi_pitch(stacked_relative_multi_pitch,
                                                                           stacked_adjusted_multi_pitch)

            # Collapse the stacked notes representation into a single collection
            pitches, intervals = tools.stacked_notes_to_notes(stacked_notes)
            # Batch the notes
            batched_notes = tools.notes_to_batched_notes(pitches, intervals)
            # Add the notes to the dictionary
            data[tools.KEY_NOTES] = batched_notes

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                keys = (tools.KEY_FS, tools.KEY_AUDIO, #tools.KEY_TABLATURE,
                        tools.KEY_MULTIPITCH, constants.KEY_MULTIPITCH_REL, tools.KEY_NOTES)
                tools.save_pack_npz(gt_path, keys, data[tools.KEY_FS], data[tools.KEY_AUDIO],
                                    #data[tools.KEY_TABLATURE],
                                    data[tools.KEY_MULTIPITCH],
                                    data[constants.KEY_MULTIPITCH_REL], data[tools.KEY_NOTES])

        return data
