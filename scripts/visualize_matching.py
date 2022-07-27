# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

import amt_tools.tools as tools

import utils

# Regular imports
import matplotlib.pyplot as plt
import numpy as np
import os

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

# All cached data/features kept here
gset_cache = os.path.join('..', 'generated', 'data')

# Initialize the dataset to visualize
tablature_data = GuitarSet(base_dir=None,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           data_proc=data_proc,
                           profile=profile,
                           save_data=False,
                           store_data=False,
                           save_loc=gset_cache)

# Construct a path to the base directory for saving visualizations
save_dir = os.path.join('..', 'generated', 'visualization', 'matching')
os.makedirs(save_dir, exist_ok=True)

# Use this line to only visualize data for specific tracks
# tablature_data.tracks = ['<track_name>']

kwargs = {'semitone_width' : 1.5, # semitones
          'stream_tolerance' : 0.55, # semitones
          'minimum_contour_duration' : 6, # milliseconds
          'attempt_corrections' : True,
          'combine_associated_contours' : False,
          'suppress_warnings' : False}

# Define a list of colors to loop through when plotting event-level data
color_loop = ['red', 'green', 'blue']

# Loop through each track in the tablature data
for track in tablature_data:
    # Obtain the name of the track
    track_name = track[tools.KEY_TRACK]

    # Create a save directory for all data from this track
    track_dir = os.path.join(save_dir, track_name)

    # Make sure the save directory exists
    os.makedirs(track_dir, exist_ok=True)

    print(f'Processing track {track_name}...')

    # Construct the path to the track's audio
    wav_path = tablature_data.get_wav_path(track_name)
    # Load and normalize the audio along with the sampling rate
    audio, fs = tools.load_normalize_audio(wav_path, sample_rate)

    # We need the frame times for the tablature
    #times = tablature_data.data_proc.get_times(audio)

    # Construct the path to the track's JAMS data
    jams_path = tablature_data.get_jams_path(track_name)

    # Load the notes by string from the JAMS file
    stacked_notes = tools.load_stacked_notes_jams(jams_path)

    # Load the string-wise pitch annotations from the JAMS file
    stacked_pitch_list = tools.load_stacked_pitch_list_jams(jams_path)
    stacked_pitch_list = tools.stacked_pitch_list_to_midi(stacked_pitch_list)

    # Obtain the in-order keys for each stack
    stacked_notes_keys = list(stacked_notes.keys())
    stacked_pitch_list_keys = list(stacked_pitch_list.keys())

    # Loop through the slices of the collections
    for i in range(len(stacked_notes_keys)):
        # Extract the key for the current slice in each collection
        key_n, key_pl = stacked_notes_keys[i], stacked_pitch_list_keys[i]

        # Create paths to save the plots for this string
        save_path_rg = os.path.join(track_dir, f'regular-{key_pl}.jpg')
        save_path_ds = os.path.join(track_dir, f'downsampled-{key_pl}.jpg')

        # Obtain the note stream multi pitch arrays for the notes in this slice
        #relative_multi_pitch, \
        #    adjusted_multi_pitch = utils.streams_to_continuous_multi_pitch_by_cluster(stacked_notes[key_n],
        #                                                                              stacked_pitch_list[key_pl],
        #                                                                              profile, **kwargs)

        # Extract the notes from the slice
        pitches, intervals = stacked_notes[key_n]

        # Extract the pitch list from the slice
        _times, _pitch_list = stacked_pitch_list[key_pl]

        # TODO - turn following plotting code into a function so it can be reused for downsampled data

        # Initialize a figure to hold all the plots
        fig = tools.initialize_figure(interactive=False, figsize=(20, 5))

        # Loop through all ground-truth notes
        for n in range(len(pitches)):
            # Obtain a notes representation of the single note
            pitch, interval = pitches[n: n + 1], intervals[n: n + 1]
            # Choose the next color for the note
            color = color_loop[n % len(color_loop)]
            # Obtain an independent multi pitch array for each note
            note_multi_pitch = tools.notes_to_multi_pitch(pitch, interval, _times, profile)
            # Plot the multi pitch array for the note
            fig = tools.plot_pianoroll(note_multi_pitch, _times, profile, overlay=True, color=color, alpha=0.25, fig=fig)
            # Plot the note (rounded to nearest pitch) as a rectangular outline
            fig = tools.plot_notes(np.round(pitch), interval, x_bounds=fig.gca().get_xlim(), color=color, fig=fig)

        # Plot all pitch contour data
        # TODO - would be interesting to look at pitch contour data before placing it on a uniform time grid
        fig = tools.plot_pitch_list(_times, _pitch_list, point_size=7,
                                    x_bounds=fig.gca().get_xlim(), color='k', alpha=0.25, fig=fig)
        # Make grid lines visible on the plot at the frame times
        fig.gca().vlines(_times, ymin=profile.low - 0.5, ymax=profile.high + 0.5, color='k', alpha=0.05)

        # Save the figure
        fig.savefig(save_path_rg)#, dpi=500)
        # Close the figure
        plt.close(fig)

    # TODO - need the following for regular plot
    #        - notes plotted w/ absolute time (rectangles w/ transparent fill)
    #        - adjusted multipitch at frame-level (whilst retaining note associations)
    #        - pitch contours associated with notes

    # Determine the total length of the track in seconds
    #total_duration = tools.load_duration_jams(jams_path)

    # TODO - in order to set this up without re-writing code, I'd have
    #        to make the functions in utils.py much more modular. But
    #        it might be good to do that anyway, so that it would be
    #        possible to keep track of note/contour relationships to,
    #        e.g., retain onset information

    # TODO - visualize matching

    # Obtain the times according to the feature extraction parameters
    #downsampled_times = data_proc.get_times(track[tools.KEY_AUDIO])

    # TODO - visualize downsampled version
