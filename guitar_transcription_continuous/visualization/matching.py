# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

import amt_tools.tools as tools

import utils

# Regular imports
import matplotlib.pyplot as plt
import numpy as np
import jams
import os


def plot_association(pitch, interval, times, contours, color='k', fig=None):
    """
    Plot a collection of pitch contours associated with a single note.

    Parameters
    ----------
    pitch : ndarray (1)
      Pitch corresponding to a single note
    interval : ndarray (1 x 2)
      Onset-offset time pair corresponding to the note
    times : ndarray (N)
      Time in seconds of beginning of each frame
      N - number of time samples (frames)
    contours : list of PitchContour
      Contour objects associated with the provided note
    color : string
      Color for the association
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the association
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = tools.initialize_figure(interactive=False)

    # Obtain an independent multi pitch array for each note
    note_multi_pitch = tools.notes_to_multi_pitch(pitch, interval, times, profile)

    # Plot the multi pitch array for the note
    fig = tools.plot_pianoroll(note_multi_pitch, times, profile, overlay=True, color=color, alpha=0.25, fig=fig)

    # Plot the note (rounded to nearest pitch) as a rectangular outline
    fig = tools.plot_notes(np.round(pitch), interval, x_bounds=fig.gca().get_xlim(), color=color, fig=fig)

    # Loop through all contours associated with the note as a result of the grouping algorithm
    for contour in contours:
        # Plot the pitch contour data associated with the note
        fig = tools.plot_pitch_list(*contour.get_pitch_list(times), point_size=25,
                                    x_bounds=fig.gca().get_xlim(), overlay=True,
                                    color=color, alpha=0.5, fig=fig)

    return fig


def plot_note_contour_associations(notes, pitch_list, grouping, primary_color_loop=None,
                                   secondary_color_loop=None, fig=None):
    """
    Plot estimated groupings between notes and pitch contours.

    Parameters
    ----------
    notes : tuple (pitches, intervals)
      pitches : ndarray (K)
        Array of pitches corresponding to notes in MIDI format
      intervals : ndarray (K x 2)
        Array of onset-offset time pairs corresponding to notes
      (K - number of notes)
    pitch_list : tuple (times, pitch_list)
      pitch_list : list of ndarray (N x [...])
        Collection of MIDI pitches corresponding to notes
      times : ndarray (N)
        Time in seconds of beginning of each frame
      (N - number of pitch observations (frames))
    grouping : (note, [contours]) pairs : dict
      Dictionary containing (note_idx -> [PitchContour]) pairs
    primary_color_loop : list of str
      Colors to use in sequence when plotting notes
    secondary_color_loop : list of str
      Colors to use in sequence when plotting corrections
    fig : matplotlib Figure object
      Preexisting figure to use for plotting

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the association
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = tools.initialize_figure(interactive=False)

    if primary_color_loop is None:
        # Default colors to loop through for ground-truth notes
        primary_color_loop = ['red', 'green', 'blue']

    if secondary_color_loop is None:
        # Default colors to loop through for note corrections
        secondary_color_loop = ['magenta', 'yellow', 'cyan']

    # Unpack the note data
    pitches, intervals = notes

    # Unpack the pitch list data
    times, pitch_list = pitch_list

    # Loop through all ground-truth notes
    for n in range(len(pitches)):
        # Obtain a notes representation of the single note
        pitch, interval = pitches[n: n + 1], intervals[n: n + 1]
        # Choose the next color for the note
        color = primary_color_loop[n % len(primary_color_loop)]
        # Plot the note with any associated pitch contours
        fig = plot_association(pitch, interval, times, grouping[n], color, fig)

    # Loop through any note corrections
    for k in np.setdiff1d(list(grouping.keys()), np.arange(len(pitches))):
        # Compute the mean of the contour in the beginning-mid region as the pitch
        pitch = np.mean([c.get_region_average(0.25, 0.5) for c in grouping[k]], keepdims=True)
        # Obtain the intervals for all contours associated with the correction
        _intervals = np.array([c.get_interval(times) for c in grouping[k]])
        # Construct one overarching interval for the correction
        interval = np.array([[np.min(_intervals[:, 0]), np.max(_intervals[:, 1])]])
        # Choose the next color for the correction
        color = secondary_color_loop[k % len(secondary_color_loop)]
        # Plot the correction with any associated pitch contours
        fig = plot_association(pitch, interval, times, grouping[k], color, fig)

    # Plot the pitch contour data for the whole track
    fig = tools.plot_pitch_list(times=times, pitch_list=pitch_list,
                                point_size=10, x_bounds=fig.gca().get_xlim(),
                                overlay=True, color='k', alpha=0.5, fig=fig)
    # Make grid lines visible on the plot at the frame times
    fig.gca().vlines(times, ymin=profile.low - 0.5, ymax=profile.high + 0.5, color='k', alpha=0.05)

    return fig


if __name__ == '__main__':
    # Processing parameters
    sample_rate = 22050
    hop_length = 512

    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Create the cqt data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=192,
                    bins_per_octave=24)

    # Initialize the dataset to visualize
    tablature_data = GuitarSet(base_dir=None,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               data_proc=data_proc,
                               profile=profile,
                               save_data=False,
                               store_data=False)

    # Construct a path to the base directory for saving visualizations
    save_dir = os.path.join('../..', 'generated', 'visualization', 'matching')
    os.makedirs(save_dir, exist_ok=True)

    # Use this line to only visualize data for specific tracks
    # tablature_data.tracks = ['<track_name>']

    kwargs = {'semitone_width' : 1.0, # semitones
              'stream_tolerance' : 0.4, # semitones
              'minimum_contour_duration' : 18, # milliseconds
              'attempt_corrections' : True,
              'suppress_warnings' : False}

    # Loop through each track in the tablature data
    for track in tablature_data:
        # Obtain the name of the track
        track_name = track[tools.KEY_TRACK]

        # Create a save directory for all data from this track
        track_dir = os.path.join(save_dir, track_name)

        # Make sure the save directory exists
        os.makedirs(track_dir, exist_ok=True)

        print(f'Processing track {track_name}...')

        # Construct the path to the track's JAMS data
        jams_path = tablature_data.get_jams_path(track_name)

        # Load the track's JAMS data
        jams_data = jams.load(jams_path)

        # Load the notes by string from the JAMS file
        stacked_notes = tools.extract_stacked_notes_jams(jams_data)

        # Load the string-wise pitch annotations from the JAMS file
        stacked_pitch_list = tools.extract_stacked_pitch_list_jams(jams_data)
        stacked_pitch_list = tools.stacked_pitch_list_to_midi(stacked_pitch_list)

        # Load the non-uniform string-wise pitch annotations from the JAMS file
        _stacked_pitch_list = tools.extract_stacked_pitch_list_jams(jams_data, uniform=False)
        _stacked_pitch_list = tools.stacked_pitch_list_to_midi(_stacked_pitch_list)

        # Create path to save the plot for all strings
        save_path = os.path.join(track_dir, f'all.jpg')

        # Collapse the notes from all slices into a single representation
        all_notes = tools.stacked_notes_to_notes(stacked_notes)
        # Collapse contour data from all slices into a single representation
        all_contours = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

        # Obtain the contour grouping for all notes at once
        grouping = utils.get_note_contour_grouping_by_cluster(all_notes,
                                                              all_contours,
                                                              **kwargs)

        # Initialize a new figure for the associations
        fig = tools.initialize_figure(interactive=False, figsize=(20, 5))
        # Plot all the associations drawn from the data
        fig = plot_note_contour_associations(notes=all_notes,
                                             pitch_list=all_contours,
                                             grouping=grouping,
                                             primary_color_loop=['r', 'orange', 'y', 'g', 'b', 'indigo', 'violet'],
                                             secondary_color_loop=['k'],
                                             fig=fig)

        # Collapse non-uniform contours from all slices into a single representation
        _all_contours = tools.stacked_pitch_list_to_pitch_list(_stacked_pitch_list)

        # Plot the non-uniform pitch contour data for the whole track
        fig = tools.plot_pitch_list(*_all_contours, point_size=10, x_bounds=fig.gca().get_xlim(),
                                    overlay=True, color='k', marker='x', alpha=0.25, fig=fig)

        # Save the figure
        fig.savefig(save_path, dpi=500)
        # Close the figure
        plt.close(fig)
