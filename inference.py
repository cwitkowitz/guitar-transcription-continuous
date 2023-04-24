# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.estimators import StackedPitchListTablatureWrapper
from amt_tools.features import HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
from amt_tools.inference import run_offline

import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
import matplotlib.pyplot as plt
import matplotlib
import librosa
import torch
import os


matplotlib.use('TkAgg')

# Define path to model and audio to transcribe
model_path = os.path.join(tools.HOME, 'Downloads', 'FretNet', 'models', 'fold-0', 'model-2000.pt')
audio_path = os.path.join(tools.HOME, 'Desktop', 'Datasets', 'GuitarSet', 'audio_mono-mic', '00_Jazz2-187-F#_solo_mic.wav')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512
# Flag to re-acquire ground-truth data and re-calculate features
reset_data = False
# Choose the GPU on which to perform evaluation
gpu_id = 0

# Initialize a device pointer for loading the model
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load the model
model = torch.load(model_path, map_location=device)
model.change_device(gpu_id)
model.eval()

# Extract the guitar profile
profile = model.profile

##############################
# Predictions                #
##############################

# Load in the audio and normalize it
audio, _ = tools.load_normalize_audio(audio_path, sample_rate)

# Create an HCQT feature extraction module comprising
# the first five harmonics and a sub-harmonic, where each
# harmonic transform spans 4 octaves w/ 3 bins per semitone
data_proc = HCQT(sample_rate=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.note_to_hz('E2'),
                 harmonics=[0.5, 1, 2, 3, 4, 5],
                 n_bins=144, bins_per_octave=36)

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
    # Continuous tablature arrays -> stacked pitch list
    StackedPitchListTablatureWrapper(profile=model.profile,
                                     multi_pitch_key=tools.KEY_TABLATURE,
                                     multi_pitch_rel_key=utils.KEY_TABLATURE_REL)])

# Perform inference offline
predictions = run_offline(features, model, estimator)

# Extract the estimated notes
stacked_notes_est = predictions[tools.KEY_NOTES]

##############################
# Plotting                   #
##############################

# Convert the estimated notes to frets
stacked_frets_est = tools.stacked_notes_to_frets(stacked_notes_est)

# Plot estimated tablature and add an appropriate title
fig_est = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_est = tools.plot_guitar_tablature(stacked_frets_est, fig=fig_est)
fig_est.suptitle('Inference')

# Display the plot
plt.show(block=True)
