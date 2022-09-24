# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet, TranscriptionDataset

import guitar_transcription_continuous.utils as utils

import amt_tools.tools as tools

# Regular imports
from copy import deepcopy

import numpy as np
import warnings
import jams
import muda
import os


class GuitarSetPlus(GuitarSet):
    """
    Simple wrapper to additionally include notes and continuous pitch information in ground-truth.
    """

    # TODO - can I remove clutter by making other dataset signatures like this?
    def __init__(self, semitone_radius=0.5, rotarize_deviations=False, augment=False,
                 silence_activations=False, evaluation_extras=False, **kwargs):
        """
        Initialize the dataset variant.

        Parameters
        ----------
        See GuitarSet class for others...

        semitone_radius : float
          Scaling factor for relative pitch estimates
        rotarize_deviations : bool
          Whether non-active relative pitch values should
          be non-zero, revolving around the active pitch
        augment : bool
          Whether to apply pitch shifting data augmentation
        silence_activations : bool
          Whether silent strings are explicitly modeled as activations
        evaluation_extras : bool
          Whether to compute/load extraneous data (for evaluation)
        """

        self.semitone_radius = semitone_radius
        self.rotarize_deviations = rotarize_deviations
        self.augment = augment
        self.silence_activations = silence_activations
        self.evaluation_extras = evaluation_extras

        # Determine if the base directory argument was provided
        base_dir = kwargs.pop('base_dir', None)

        # Select a default base directory path if none was provided
        if base_dir is None:
            # Use the same naming scheme as regular GuitarSet
            base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, GuitarSet.dataset_name())

        # Update the argument in the collection
        kwargs.update({'base_dir' : base_dir})

        if self.augment:
            # Make sure there is no fixed data storage in RAM, since
            # ground-truth and features will change every iteration
            kwargs.update({'store_data': False})

        super().__init__(**kwargs)

        if not self.evaluation_extras and self.save_data:
            warnings.warn('Evaluation extras will be excluded from saved data.', category=RuntimeWarning)

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

        # Default the amount of pitch shifting to 0 semitones
        semitone_shift = 0

        if self.augment:
            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the original jams data (ignoring validation for speedup)
            jams_data = jams.load(jams_path, validate=False)

            # Load the original notes by string from the JAMS data
            stacked_notes = tools.extract_stacked_notes_jams(jams_data)

            # Sample a valid semitone shift according to the pre-existing notes in the track
            semitone_shift = sample_valid_pitch_shift(stacked_notes, self.profile, 5, None, self.rng)

        # Update the track name to reflect any augmentation
        track_ = track + f'_{semitone_shift}' if semitone_shift else track

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = TranscriptionDataset.load(self, track_)

        # If the track data is being instantiated, it will not have the 'audio' key
        if not tools.query_dict(data, tools.KEY_AUDIO):
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            audio, fs = tools.load_normalize_audio(wav_path,
                                                   fs=self.sample_rate,
                                                   norm=self.audio_norm)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(audio)

            if not self.augment:
                # Construct the path to the track's JAMS data
                jams_path = self.get_jams_path(track)

                # Load the original jams data
                jams_data = jams.load(jams_path)
            else:
                # Add the audio to the jams data
                jams_data = muda.jam_pack(jams_data, _audio=dict(y=audio, sr=fs))

                # Apply a pitch shift transformation to the audio and annotations
                jams_data = next(muda.deformers.PitchShift(n_semitones=semitone_shift).transform(jams_data))

                # Extract the augmented audio from the JAMS data
                fs = jams_data.sandbox.muda._audio['sr']
                audio = jams_data.sandbox.muda._audio['y']

            # Load the notes by string from the JAMS data
            stacked_notes = tools.extract_stacked_notes_jams(jams_data)

            # Represent the string-wise notes as stacked multi pitch arrays
            stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)
            stacked_offsets = tools.stacked_notes_to_stacked_offsets(stacked_notes, times, self.profile)
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into tablature
            tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Load the string-wise pitch annotations from the JAMS data
            stacked_pitch_list = tools.extract_stacked_pitch_list_jams(jams_data)
            stacked_pitch_list = tools.stacked_pitch_list_to_midi(stacked_pitch_list)

            # Obtain the relative pitch deviation of the contours anchored by string/fret
            stacked_relative_multi_pitch, stacked_adjusted_multi_pitch = \
                utils.stacked_streams_to_stacked_continuous_multi_pitch(stacked_notes,
                                                                        stacked_pitch_list,
                                                                        self.profile,
                                                                        times=times,
                                                                        semitone_radius=self.semitone_radius,
                                                                        stream_tolerance=0.4, # semitones
                                                                        minimum_contour_duration=18, # milliseconds
                                                                        attempt_corrections=True,
                                                                        suppress_warnings=True)

            if self.rotarize_deviations:
                # Obtain a rotary representation of the relative pitch deviations
                stacked_relative_multi_pitch = utils.get_rotarized_relative_multi_pitch(stacked_relative_multi_pitch,
                                                                                        stacked_adjusted_multi_pitch)

            # Clip the deviations at the supported semitone width
            stacked_relative_multi_pitch = np.clip(stacked_relative_multi_pitch,
                                                   a_min=-self.semitone_radius,
                                                   a_max=self.semitone_radius)

            # Obtain a logistic representation of the multi pitch ground-truth adjusted to the pitch contours
            adjusted_multi_pitch = \
                tools.stacked_multi_pitch_to_logistic(stacked_adjusted_multi_pitch,
                                                      self.profile, self.silence_activations)

            # Obtain a logistic representation of the corresponding relative pitch deviations
            relative_multi_pitch = \
                tools.stacked_multi_pitch_to_logistic(stacked_relative_multi_pitch, self.profile, False)

            # Add all relevant ground-truth to the dictionary
            data.update({tools.KEY_FS : fs,
                         tools.KEY_AUDIO : audio,
                         tools.KEY_TABLATURE: tablature,
                         utils.KEY_TABLATURE_ADJ : adjusted_multi_pitch,
                         utils.KEY_TABLATURE_REL : relative_multi_pitch,
                         tools.KEY_ONSETS : stacked_onsets})

            if self.evaluation_extras:
                # Add evaluation extras to the dictionary
                data.update({tools.KEY_NOTES : stacked_notes,
                             tools.KEY_OFFSETS : stacked_offsets,
                             tools.KEY_MULTIPITCH : stacked_multi_pitch,
                             tools.KEY_PITCHLIST : stacked_pitch_list})

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track_)

                # Create a copy of the data
                data_to_save = deepcopy(data)

                if self.evaluation_extras:
                    # Package the stacked notes into save-friendly format
                    data_to_save.update({tools.KEY_NOTES : tools.pack_stacked_representation(stacked_notes)})
                    # Package the stacked pitch list into save-friendly format
                    data_to_save.update({tools.KEY_PITCHLIST : tools.pack_stacked_representation(stacked_pitch_list)})

                # Save the data as a NumPy zip file
                tools.save_dict_npz(gt_path, data_to_save)

        return data


def sample_valid_pitch_shift(stacked_notes, profile, steepness=5, max_pitch_shift=None, rng=None):
    """
    Sample a random pitch shift that does not violate the instrument profile.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    profile : GuitarProfile (instrument.py)
      Instrument profile detailing experimental setup
    steepness : float
      Scaling factor (exponential) for relative weights before normalization
    max_pitch_shift : int or None (Optional)
      Maximum allowable pitch shift in either direction
    rng : NumPy RandomState
      Random number generator to use for augmentation

    Returns
    ----------
    semitone_shift : int
      Sampled semitone shift within valid range
    """

    if rng is None:
        # Default the random state
        rng = np.random

    # Get the MIDI pitches of the open strings
    midi_tuning = profile.get_midi_tuning()

    # Determine the minimum and maximum pitch played on each string
    min_pitches, max_pitches = tools.find_pitch_bounds_stacked_notes(stacked_notes)

    # Default the number of unused frets
    unused_frets_left = profile.get_num_frets() * np.ones(midi_tuning.shape)
    unused_frets_right = profile.get_num_frets() * np.ones(midi_tuning.shape)

    # Determine how many frets are unused along both directions of fretboard
    unused_frets_left[min_pitches != 0] = (min_pitches - midi_tuning)[min_pitches != 0]
    unused_frets_right[max_pitches != 0] = (midi_tuning - max_pitches)[max_pitches != 0] + profile.get_num_frets()

    # Determine the maximum capo shift in both directions
    max_shift_down = max(0, np.min(unused_frets_left))
    max_shift_up = max(0, np.min(unused_frets_right))

    if max_pitch_shift is not None:
        # Clip the boundaries at the maximum allowable shift
        max_shift_down = min(max_shift_down, max_pitch_shift)
        max_shift_up = min(max_shift_up, max_pitch_shift)

    # Construct an array of valid choices for the semitone shift
    valid_range = np.arange(-max_shift_down, max_shift_up + 1)

    # Obtain relative weights for all choices based on distance from zero
    relative_weights = (max(max_shift_down, max_shift_up) - np.abs(valid_range) + 1) ** steepness

    # Normalize the weights to obtain probabilities
    norm_weights = relative_weights / np.sum(relative_weights)

    # Sample a semitone shift within the valid range using the computed probabilities
    semitone_shift = int(rng.choice(valid_range, p=norm_weights))

    return semitone_shift
