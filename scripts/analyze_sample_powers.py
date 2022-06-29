# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from yousician_private import SingleNotes
from power import SignalPower, WaveformWrapper

import amt_tools.tools as tools

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os

# Processing parameters
sample_rate = 22050
hop_length = 512

# Save plot one level up
save_dir = os.path.join('..', 'generated')
os.makedirs(save_dir, exist_ok=True)

# Compute features as frame-level signal power
data_proc = SignalPower(sample_rate=sample_rate,
                        hop_length=hop_length,
                        decibels=True,
                        win_length=None,
                        center=False)
"""data_proc = WaveformWrapper(sample_rate=sample_rate,
                            hop_length=hop_length,
                            decibels=True,
                            win_length=None,
                            center=True)"""

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Instantiate the synthetic data with no normalization
sample_data = SingleNotes(base_dir=None,
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          num_frames=None,
                          audio_norm=-1,
                          data_proc=data_proc,
                          profile=profile,
                          store_data=False,
                          save_data=False)

# Create a dictionary with an empty array for both inactive/active frame collections
db_powers = dict.fromkeys(range(2), np.array([]))

sample_data.tracks = ['S_0xxxxx_correct/BlueYetiPro/EpiphoneLesPaulBlackBeautyCleanAmpHard_sthumb']

# Loop through each sample in the collection
for note in tqdm(sample_data):
    # Determine the frame(s) where the note becomes active
    onset_frames = np.sum(note[tools.KEY_ONSETS] != -1, axis=-2)

    # Obtain the powers for each "inactive" frame in the track
    new_powers = note[tools.KEY_FEATS][onset_frames == 0]
    # Add these to the tracked powers for "inactive" frames
    db_powers[0] = np.append(db_powers[0], new_powers)

    # Obtain the powers for each "active" frame in the track
    new_powers = note[tools.KEY_FEATS][onset_frames > 0]
    # Add these to the tracked powers for "active" frames
    db_powers[1] = np.append(db_powers[1], new_powers)

    # TODO - do onset/duration flagging here

# Initialize a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(14, 6))

# Create a series of bins (Decibel levels) for the histograms
bins = np.arange(-80, 1)

# Loop through each level of polyphony
for polyphony in range(2):
    # Make sure there exist some observations
    if db_powers[polyphony].size > 0:
        # Extract the powers from the dictionary
        _db_powers = db_powers[polyphony]
        # Clip power levels as the lowest bin boundary
        _db_powers[_db_powers < bins[0]] = bins[0]

        # Switch the ordering of polyphony so onset frames are plotted on top
        idx = int(not polyphony)

        # Plot the frame-level powers as a histogram
        ax[idx].hist(_db_powers, bins, color='black', alpha=0.85)

        if polyphony == 0:
            # Label the plot as representing "inactive" frames
            ax[idx].set_title(f'Other Frames')
        else:
            # Label the plot as representing "active" frames
            ax[idx].set_title(f'Onset Frames')

# Add an x-label to the lowest subplot
plt.setp(ax[-1], xlabel='Decibels')
# Add a y-label to each subplot
plt.setp(ax[:], ylabel='Count')

# Open the plot
plt.show()
