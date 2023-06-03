# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.visualization import plot_note_contour_associations
from guitar_transcription_continuous.utils import get_note_contour_grouping_by_cluster, \
                                                  get_note_contour_grouping_by_index
from guitar_transcription_continuous.estimators import TablatureStreamer
from amt_tools.features import CQT, HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOnsetsWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
from amt_tools.inference import run_offline

import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
from matplotlib import rcParams

import numpy as np
import librosa
import torch
import jams
import os


track = '05_Rock1-130-A_solo'

tabcnn_path = os.path.join(tools.HOME, 'Downloads', 'TabCNN', 'models', 'fold-5', 'model-7200.pt')
fretnet_path = os.path.join(tools.HOME, 'Downloads', 'FretNet', 'models', 'fold-5', 'model-2100.pt')

# Define path to audio and ground-truth
audio_path = os.path.join(tools.HOME, 'Desktop', 'Datasets', 'GuitarSet', 'audio_mono-mic', f'{track}_mic.wav')
jams_path = os.path.join(tools.HOME, 'Desktop', 'Datasets', 'GuitarSet', 'annotation', f'{track}.jams')

# Feature extraction parameters
sample_rate = 22050
hop_length = 512

# Choose the GPU on which to perform evaluation
gpu_id = 0

# Plotting parameters
colors = ['#191919', '#00d296', 'gray']
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 20
time_bounds = [7.25, 17.75]
figsize = (10, 8)
save_figure = True
cluster_grouping = False

# Construct a path to the base directory for saving visualizations
save_dir = os.path.join('..', 'generated', 'visualization', 'comparison')
os.makedirs(save_dir, exist_ok=True)

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load in the audio and normalize it
audio, _ = tools.load_normalize_audio(audio_path, sample_rate)

# Create a CQT feature extraction module
# spanning 8 octaves w/ 2 bins per semitone
cqt_proc = CQT(sample_rate=sample_rate, hop_length=hop_length, n_bins=192, bins_per_octave=24)

# Create an HCQT feature extraction module comprising
# the first five harmonics and a sub-harmonic, where each
# harmonic transform spans 4 octaves w/ 3 bins per semitone
hcqt_proc = HCQT(sample_rate=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.note_to_hz('E2'),
                 harmonics=[0.5, 1, 2, 3, 4, 5],
                 n_bins=144, bins_per_octave=36)

for i, (name, path, data_proc) in enumerate([('TabCNN', tabcnn_path, cqt_proc),
                                             ('FretNet', fretnet_path, hcqt_proc)]):
    # Load the chosen model checkpoint
    model = torch.load(path, map_location=device)
    model.change_device(device)
    model.eval()

    # Compute the features
    features = {tools.KEY_FEATS : data_proc.process_audio(audio),
                tools.KEY_TIMES : data_proc.get_times(audio)}

    # Initialize the estimation pipeline
    estimator = ComboEstimator([
        # Discrete tablature -> stacked multi pitch array
        TablatureWrapper(profile=model.profile),
        # Stacked multi pitch array -> stacked offsets array
        StackedOffsetsWrapper(profile=model.profile),
        # Stacked multi pitch array -> stacked notes
        StackedNoteTranscriber(profile=model.profile),
        # Continuous tablature arrays & notes -> stacked grouping
        TablatureStreamer(profile=model.profile,
                          multi_pitch_key=tools.KEY_TABLATURE,
                          multi_pitch_rel_key=utils.KEY_TABLATURE_REL)])

    if i == 0 or not model.estimate_onsets:
        # Infer the onsets directly from the multi pitch data
        estimator.estimators.insert(1,
            # Stacked multi pitch array -> stacked onsets array
            StackedOnsetsWrapper(profile=model.profile))

    # Perform inference offline
    predictions = run_offline(features, model, estimator)

    # Extract the stacked notes and grouping from the predictions
    stacked_notes = predictions[tools.KEY_NOTES]
    stacked_grouping = predictions[utils.KEY_GROUPING]

    grouping = {}
    for slc in stacked_grouping.keys():
        # Collapse the stacked grouping
        grouping.update(stacked_grouping[slc])

    # Obtain the indices of the unsorted collapsed notes
    _, unsorted_intervals = tools.stacked_notes_to_notes(stacked_notes, None)

    # Determine the ordering of the notes by onset
    note_order = np.argsort(unsorted_intervals[:, 0])

    # Sort the onset times of notes across all slices
    note_sorting_idcs = np.argsort(note_order)

    # Update the dictionary keys to reflect the sorting
    grouping = dict(sorted(zip(note_sorting_idcs, grouping.values())))

    # Re-collpase the notes and sort them by onset
    all_notes = tools.stacked_notes_to_notes(stacked_notes)

    # Initialize a new figure for the associations
    fig = tools.initialize_figure(interactive=False, figsize=figsize)

    # Plot all the associations drawn from the data
    fig = plot_note_contour_associations(notes=all_notes,
                                         times=features[tools.KEY_TIMES],
                                         profile=model.profile,
                                         grouping=grouping,
                                         primary_color_loop=colors,
                                         fig=fig)

    if time_bounds is not None:
        # Set x bounds for the predictions
        fig.gca().set_xlim(time_bounds)

    if save_figure:
        # Save the plot as a PDF
        fig.savefig(os.path.join(save_dir, f'{name}.pdf'), bbox_inches='tight')

##############################
# Ground-Truth               #
##############################

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Open up the JAMS data
jams_data = jams.load(jams_path)

# Load the notes by string from the JAMS data
stacked_notes = tools.extract_stacked_notes_jams(jams_data)
# Collapse the string dimension of the notes
all_notes = tools.stacked_notes_to_notes(stacked_notes)

# Load the string-wise pitch annotations from the JAMS data
stacked_pitch_list = tools.extract_stacked_pitch_list_jams(jams_data)
stacked_pitch_list = tools.stacked_pitch_list_to_midi(stacked_pitch_list)
times, pitch_list = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list)

if cluster_grouping:
    # Obtain the ground-truth grouping using the cluster-based methodology
    grouping = get_note_contour_grouping_by_cluster(all_notes,
                                                    (times, pitch_list),
                                                    semitone_radius=1.0,
                                                    stream_tolerance=0.4, # semitones
                                                    minimum_contour_duration=18, # milliseconds
                                                    attempt_corrections=True,
                                                    suppress_warnings=True)
else:
    # Obtain the ground-truth grouping directly from GuitarSet
    _, grouping = get_note_contour_grouping_by_index(jams_data, times)

# Initialize a new figure for the associations
fig = tools.initialize_figure(interactive=False, figsize=figsize)
# Plot all the associations drawn from the data
fig = plot_note_contour_associations(notes=all_notes,
                                     times=times,
                                     profile=profile,
                                     grouping=grouping,
                                     primary_color_loop=colors,
                                     fig=fig)
if time_bounds is not None:
    # Set x bounds on the ground-truth plot
    fig.gca().set_xlim(time_bounds)

if save_figure:
    # Save the plot as a PDF
    fig.savefig(os.path.join(save_dir, f'ground-truth.pdf'), bbox_inches='tight')
