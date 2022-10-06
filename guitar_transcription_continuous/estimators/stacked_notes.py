# My imports
from amt_tools.transcribe import StackedNoteTranscriber as _StackedNoteTranscriber

import amt_tools.tools as tools

# Regular imports
from typing import Optional, List, Tuple
from math import ceil

import numpy as np
import scipy


def get_infered_onsets(onsets: np.array, frames: np.array, n_diff: int = 2) -> np.array:
    """Infer onsets from large changes in frame amplitudes.
    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.
    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences

    return max_onsets_diff


def output_to_notes_polyphonic(
    frames: np.array,
    onsets: np.array,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    profile,
    #max_freq: Optional[float],
    #min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events
    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        melodia_trick : Whether to use the melodia trick to better detect notes.
        energy_tol: Drop notes below this energy.
    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1
    """

    n_frames = frames.shape[0]

    #onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < profile.high - profile.low:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # add the note
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + profile.low,
                amplitude,
            )
        )

    if melodia_trick:

        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            # forward pass
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:

                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < profile.high - profile.low:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i += 1

            i_end = i - 1 - k  # go back to frame above threshold

            # backward pass
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:

                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < profile.high - profile.low:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            assert i_start >= 0, "{}".format(i_start)
            assert i_end < n_frames

            if i_end - i_start <= min_note_len:
                # note is too short, skip it
                continue

            # add the note
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append(
                (
                    i_start,
                    i_end,
                    freq_idx + profile.low,
                    amplitude,
                )
            )

    return note_events


class StackedNoteTranscriber(_StackedNoteTranscriber):
    """
    Wrapper to use the note decoding method from the Basic Pitch paper.
    """

    def estimate(self, raw_output):
        """
        Estimate notes for each slice of a stacked multi pitch activation map.

        Parameters
        ----------
        raw_output : dict
          Dictionary containing raw output relevant to estimation

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        """

        # Obtain the multi pitch activation maps to transcribe
        stacked_multi_pitch = tools.unpack_dict(raw_output, self.multi_pitch_key)

        # Determine the number of slices in the stacked multi pitch array
        stack_size = stacked_multi_pitch.shape[-3]

        # Obtain the frame times associated with the activation maps
        times = tools.unpack_dict(raw_output, tools.KEY_TIMES)

        # Obtain the onsets from the raw output
        stacked_onsets = tools.unpack_dict(raw_output, self.onsets_key)

        # Initialize a dictionary to hold the notes
        stacked_notes = dict()

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Obtain all of the transcription information for this slice
            multi_pitch, onsets = stacked_multi_pitch[slc], stacked_onsets[slc]

            if self.minimum_duration is None:
                # Accept all notes
                min_note_len = 1
            else:
                # Notes must be at least enough frames to cover the specified minimum duration
                min_note_len = ceil(self.minimum_duration / tools.estimate_hop_length(times))

            # Transcribe this slice of activations
            note_events = output_to_notes_polyphonic(frames=multi_pitch.T,
                                                     onsets=onsets.T,
                                                     onset_thresh=0.5,
                                                     frame_thresh=0.5,
                                                     min_note_len=min_note_len,
                                                     infer_onsets=True,
                                                     profile=self.profile)

            # Collapse the list of event tuples into a single array
            batched_notes = np.array([event for event in note_events]).reshape((-1, 4))[:, :-1]

            # Convert the batched notes to loose frame groups
            pitches, frame_intervals = tools.batched_notes_to_notes(batched_notes)

            # Index the times with the frame intervals
            intervals = times[frame_intervals.astype(tools.INT)]

            # Sort the notes by onset
            pitches, intervals = tools.sort_notes(pitches, intervals)

            # Add the pitch-interval pairs to the stacked notes dictionary under the slice key
            stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, slc))

        return stacked_notes
