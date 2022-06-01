# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import WaveformWrapper

# Regular imports
from librosa.core import amplitude_to_db
from librosa.util import frame

import numpy as np


class SignalPower(WaveformWrapper):
    """
    Computes signal power at the frame-level.
    """
    def __init__(self, sample_rate=44100, hop_length=512, decibels=True, win_length=None, center=True):
        """
        Initialize parameters for computing signal power.

        Parameters
        ----------
        See WaveformWrapper class...
        """

        super().__init__(sample_rate=sample_rate,
                         hop_length=hop_length,
                         decibels=decibels,
                         win_length=win_length,
                         center=center)

    def _pad_audio(self, audio):
        """
        Pad audio such that trailing audio can be used.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio : ndarray
          Audio padded such that it will not throw away non-zero samples
        """

        # Handle the centered frame case
        if self.center and not audio.shape[-1] == 0:
            # TODO - make this part of waveform wrapper and
            #        have no padding in child transforms?
            # Compute the padding which would occur in (librosa) STFT
            padding = [tuple([int(self.win_length // 2)] * 2)]
            # Pad the signal on both sides
            audio = np.pad(audio, padding, mode='constant')
        else:
            audio = super()._pad_audio(audio)

        return audio

    def process_audio(self, audio):
        """
        Get the signal power for each frame of a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        powers : ndarray
          Frame-level signal powers
        """

        # Pad the audio
        audio = self._pad_audio(audio)
        # Obtain the audio samples associated with each frame
        audio_frames = frame(audio,
                             frame_length=self.win_length,
                             hop_length=self.hop_length)
        # Compute frame-level signal powers
        powers = np.sum(audio_frames ** 2, axis=-2) / self.win_length

        if self.decibels:
            # Convert to Decibels using the maximum
            # power (among this signal) as the reference
            powers = amplitude_to_db(powers, ref=np.max)

        return powers

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        # Simply one value (power) per frame
        feature_size = 1

        return feature_size
