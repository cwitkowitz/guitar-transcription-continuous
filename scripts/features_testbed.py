# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import STFT, WaveformWrapper
from amt_tools.datasets import TranscriptionDataset
from yousician_private import SingleNotes
from power import SignalPower

import amt_tools.tools as tools

# Regular imports
from tqdm import tqdm

import numpy as np
import torch
import os


class BasicTestWrapper(SingleNotes):
    """
    Simple wrapper to serve as a testbed for features.
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
            data[tools.KEY_AUDIO], data[tools.KEY_FS] = tools.load_normalize_audio(wav_path,
                                                                                   fs=self.sample_rate,
                                                                                   norm=self.audio_norm)

            # Parse the track name to determine the string and fret of sample
            string, fret = self.get_string_fret(track)

            # Create an annotation for the track
            stacked_notes = self.get_stacked_notes_annotation(string, fret)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(data[tools.KEY_AUDIO], at_start=True)
            # Add the times to the data dictionary
            data[tools.KEY_TIMES] = times

            # Convert the string-wise notes into a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into tablature
            data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Obtain the onsets for each string from the stacked notes
            stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)
            # Convert the onsets to tablature format
            data[tools.KEY_ONSETS] = tools.stacked_multi_pitch_to_tablature(stacked_onsets, self.profile)

        return data


# Processing parameters
sample_rate = 22050
hop_length = 512
win_length = 512

# Compute features as frame-level signal power
data_proc = SignalPower(sample_rate=sample_rate,
                        hop_length=hop_length,
                        decibels=True,
                        win_length=win_length,
                        center=False)

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Instantiate the synthetic data with no normalization
sample_data = BasicTestWrapper(base_dir=None,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=None,
                               audio_norm=-1,
                               data_proc=data_proc,
                               profile=profile,
                               store_data=False,
                               save_data=False)

# Loop through each sample in the collection
for note in tqdm(sample_data):
    times    = note[tools.KEY_TIMES]
    features = note[tools.KEY_FEATS]

    onsets = np.sum(note[tools.KEY_ONSETS] != -1, axis=-2)
    tabs = note[tools.KEY_TABLATURE]

    print()
