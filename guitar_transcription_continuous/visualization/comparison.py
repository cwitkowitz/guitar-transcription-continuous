# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.visualization import plot_note_contour_associations
from guitar_transcription_continuous.utils import get_note_contour_grouping_by_cluster, \
                                                  get_note_contour_grouping_by_index
from guitar_transcription_continuous.estimators import TablatureStreamer
from amt_tools.features import CQT, HCQT, WaveformWrapper

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOnsetsWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
from amt_tools.inference import run_offline

import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
import librosa
import torch
import jams
import os


track = '00_BN3-119-G_comp'

tabcnn_path = '../../generated/experiments/baselines/TabCNN_GuitarSetPlus_CQT_1/models/fold-0/model-700.pt'
fretnet_path = '../../generated/experiments/second_insights/FretNet_GuitarSetPlus_HCQT_21/models/fold-0/model-2000.pt'

# Define path to audio and ground-truth
audio_path = f'/home/rockstar/Desktop/Datasets/GuitarSet/audio_mono-mic/{track}_mic.wav'
jams_path = f'/home/rockstar/Desktop/Datasets/GuitarSet/annotation/{track}.jams'

# Feature extraction parameters
sample_rate = 22050
hop_length = 512

# Choose the GPU on which to perform evaluation
gpu_id = 0

# Plotting parameters
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 20
time_bounds = None#[5, 10]
figsize = (10, 8)
save_figure = True

# Construct a path to the base directory for saving visualizations
save_dir = os.path.join('../..', 'generated', 'visualization', 'comparison')
os.makedirs(save_dir, exist_ok=True)

# Initialize a device pointer for loading the models
#device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = torch.device(f'cpu')

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
    #model.change_device(gpu_id)
    model.change_device(device)
    model.eval()

    # Extract the guitar profile
    profile = model.profile

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

    # Initialize a new figure for the associations
    fig = tools.initialize_figure(interactive=False, figsize=figsize)

    # Loop through each grouping in the stack
    for key in stacked_grouping.keys():
        # Extract the note estimates and grouping for the current slice
        notes, grouping = stacked_notes[key], stacked_grouping[key]

        # Plot all the associations drawn from the data
        fig = plot_note_contour_associations(notes=notes,
                                             times=features[tools.KEY_TIMES],
                                             profile=profile,
                                             grouping=grouping,
                                             primary_color_loop=['#191919', '#00d296', 'grey'],
                                             fig=fig)

    if time_bounds is not None:
        # Set x bounds on the ground-truth plot
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

# Obtain frame times
times = WaveformWrapper(sample_rate=sample_rate, hop_length=hop_length).get_times(audio)

# Obtain the ground-truth grouping
_, grouping = get_note_contour_grouping_by_index(jams_data, times)

# Initialize a new figure for the associations
fig = tools.initialize_figure(interactive=False, figsize=figsize)
# Plot all the associations drawn from the data
fig = plot_note_contour_associations(notes=all_notes,
                                        times=times,
                                        profile=profile,
                                        grouping=grouping,
                                        primary_color_loop=['#191919', '#00d296', 'grey'],
                                        fig=fig)
if time_bounds is not None:
    # Set x bounds on the ground-truth plot
    fig.gca().set_xlim(time_bounds)

if save_figure:
    # Save the plot as a PDF
    fig.savefig(os.path.join(save_dir, f'ground-truth.pdf'), bbox_inches='tight')
