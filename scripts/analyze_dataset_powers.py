# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from yousician_private import SyntheticGuitar_V2
from amt_tools.datasets import GuitarSet
from power import SignalPower

import amt_tools.tools as tools

# Regular imports
from librosa.core import amplitude_to_db
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import random
import os

# TODO - make wrapper which additionally adds left-aligned times (at_start=True) to data

# Number of tracks to use randomly (None for all)
N = 200
# Which type of ground-truth to analyze ('onsets' | 'tablature')
frame_type = tools.KEY_ONSETS

# Processing parameters
sample_rate = 22050
hop_length = 512

# Save plot one level up
save_dir = os.path.join('..', 'generated')
os.makedirs(save_dir, exist_ok=True)

# Compute features as frame-level signal power
data_proc = SignalPower(sample_rate=sample_rate,
                        hop_length=hop_length,
                        decibels=False, # Convert to dB later with global max
                        win_length=None,
                        center=True)

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Instantiate the synthetic data with no normalization
tablature_data = SyntheticGuitar_V2(base_dir=None,
                                    splits=['train', 'val'],
                                    hop_length=hop_length,
                                    sample_rate=sample_rate,
                                    num_frames=None,
                                    audio_norm=None,
                                    data_proc=data_proc,
                                    profile=profile,
                                    store_data=False,
                                    save_data=False)

if N is not None:
    # Shuffle the track ordering
    random.shuffle(tablature_data.tracks)
    # Trim the dataset to N random tracks
    tablature_data.tracks = tablature_data.tracks[:N]

# Create a dictionary with an empty array for each level of polyphony
powers = dict.fromkeys(range(7), np.array([]))

# Loop through each track in the tablature data
for track in tqdm(tablature_data):
    # Determine the number of active strings (from ground-truth) in each frame
    num_active_strings = np.sum(track[frame_type] != -1, axis=-2)
    # Loop through each level of polyphony
    for polyphony in range(7):
        # Obtain the powers for each frame in the track matching the polyphony
        new_powers = track[tools.KEY_FEATS][num_active_strings == polyphony]
        # Add these to the tracked powers for the polyphony level
        powers[polyphony] = np.append(powers[polyphony], new_powers)

# Determine the maximum power level across all polyphony levels
global_max = np.max([np.max(vals) for vals in powers.values() if vals.size > 0])

# Initialize a figure with a subplot for each polyphony level
fig, ax = plt.subplots(7, 1, figsize=(14, 21))
# Open the figure and leave it open
plt.show(block=False)

# Create a series of bins (Decibel levels) for the histograms
bins = np.arange(-80, 1)

# Loop through each level of polyphony
for polyphony in range(7):
    # Make sure there exist some observations
    if powers[polyphony].size > 0:
        # Convert the powers to Decibels using the global max as a reference
        db_powers = amplitude_to_db(powers[polyphony], ref=global_max)
        # Clip power levels as the lowest bin boundary
        db_powers[db_powers < bins[0]] = bins[0]

        # Plot the frame-level powers as a histogram
        ax[polyphony].hist(db_powers, bins, color='black', alpha=0.85)
        # Label the polyphony level and statistics as the title for the subplot
        ax[polyphony].set_title(f'Polyphony: {polyphony} | Mean Power : {np.mean(db_powers)} ± {np.std(db_powers)} dB')

# Add an x-label to the lowest subplot
plt.setp(ax[-1], xlabel='Decibels')
# Add a y-label to each subplot
plt.setp(ax[:], ylabel='Count')

# Adjust sizes so subplot titles do not overlap with axes
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Construct the file name based off of processing parameters
file_name = f'{tablature_data.dataset_name()}'
if N is not None:
    file_name += f'-{N}'
file_name += f'-{frame_type}.jpg'

# Save the plot as a figure
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
