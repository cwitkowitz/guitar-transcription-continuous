# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet
from guitar_transcription_continuous.estimators import StackedPitchListTablatureWrapper, \
                                                       StackedNoteTranscriber
from guitar_transcription_continuous.evaluators import *
from amt_tools.features import CQT, HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOnsetsWrapper, \
                                 StackedOffsetsWrapper
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate, \
                               append_results, \
                               average_results

import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
import datetime
import warnings
import librosa
import torch
import json
import os

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join(tools.HOME, 'Desktop', 'guitar-transcription-continuous',
                              'generated', 'experiments', 'FretNet_GuitarSetPlus_HCQT_X')

# Define the model checkpoints to use for six-fold cross-validation
checkpoints = [-1, -1, -1, -1, -1, -1]
# Tag to mark the metric used to choose checkpoints
identifier = 'string-note-on'

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512
# Flag to re-acquire ground-truth data and re-calculate features
reset_data = False
# Choose the GPU on which to perform evaluation
gpu_id = 0

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Create an HCQT feature extraction module comprising
# the first five harmonics and a sub-harmonic, where each
# harmonic transform spans 4 octaves w/ 3 bins per semitone
data_proc = HCQT(sample_rate=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.note_to_hz('E2'),
                 harmonics=[0.5, 1, 2, 3, 4, 5],
                 n_bins=144, bins_per_octave=36)

# Define tolerances to use when evaluating pitch lists
tols = [1/2, 1/4, 1/8, 1/16] # semitones

# Determine which directory under the experiment corresponds to the latest run
output_dir = sorted([dir for dir in os.listdir(experiment_dir) if dir.isdigit()])[-1]
# Specify the path to an output file to log results
output_path = os.path.join(experiment_dir, output_dir, f'six-fold-{identifier}.json')

# Initialize an empty dictionary to hold the average results across folds
results = dict()

# Loop through each fold
for k in range(6):
    # Obtain the path to the directory containing model checkpoints for the fold
    checkpoints_dir = os.path.join(experiment_dir, 'models', f'fold-{k}')

    if not os.path.exists(checkpoints_dir):
        # Move to the next fold if the fold directory doesn't exist
        continue

    # Parse the files in the directory to obtain all model checkpoint files
    model_paths = [path for path in os.listdir(checkpoints_dir) if 'model' in path]

    if not len(model_paths):
        # Move to the next fold if there are no model checkpoints
        continue

    # Sort the checkpoints by iteration
    model_paths = sorted(model_paths, key=tools.file_sort)
    # Determine the file name of the chosen checkpoint
    target_name = f'model-{checkpoints[k]}.pt'

    if target_name in model_paths:
        # If the checkpoint exits, build a path to it
        model_path = os.path.join(checkpoints_dir, target_name)
    else:
        if checkpoints[k] != -1:
            # If the checkpoint doesn't exist and the latest checkpoint wasn't chosen, throw a warning
            warnings.warn(f'Could not find file {target_name} under checkpoints directory for ' +
                          f'fold {k}. Choosing latest checkpoint instead.', category=RuntimeWarning)
        # Build a path to the latest checkpoint
        model_path = os.path.join(checkpoints_dir, model_paths[-1])

    # Determine which checkpoint ended up being selected
    checkpoint = int(os.path.basename(model_path).replace('model-', '').replace('.pt', ''))

    # Load the model onto the chosen device
    model = torch.load(model_path, map_location=device)
    model.change_device(gpu_id)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([
        # Discrete tablature -> stacked multi pitch array
        TablatureWrapper(profile=model.profile),
        # Stacked multi pitch array -> stacked offsets array
        StackedOffsetsWrapper(profile=model.profile),
        # Stacked multi pitch array -> stacked notes
        StackedNoteTranscriber(profile=model.profile,
                               minimum_duration=0.12),
        # Continuous tablature arrays -> stacked pitch list
        StackedPitchListTablatureWrapper(profile=model.profile,
                                         multi_pitch_key=tools.KEY_TABLATURE,
                                         multi_pitch_rel_key=utils.KEY_TABLATURE_REL)])

    if not model.estimate_onsets:
        # Infer the onsets directly from the multi pitch data
        validation_estimator.estimators.insert(1,
            # Stacked multi pitch array -> stacked onsets array
            StackedOnsetsWrapper(profile=model.profile))

    # Initialize the evaluation pipeline - ( Loss,
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           # Onsets/Offsets,
                                           OnsetsEvaluator(),
                                           OffsetsEvaluator(),
                                           # Multi Pitch,
                                           MultipitchEvaluator(),
                                           # String-Level Onsets/Offsets,
                                           TablatureOnsetEvaluator(profile=model.profile,
                                                                   results_key=f'string-{tools.KEY_ONSETS}'),
                                           TablatureOffsetEvaluator(profile=model.profile,
                                                                    results_key=f'string-{tools.KEY_OFFSETS}'),
                                           # Tablature (String-Level Multi Pitch),
                                           TablatureEvaluator(profile=model.profile),
                                           SoftmaxAccuracy(),
                                           # Notes
                                           NoteEvaluator(results_key=tools.KEY_NOTE_ON),
                                           NoteEvaluator(offset_ratio=0.2,
                                                         results_key=tools.KEY_NOTE_OFF),
                                           # String-Level Notes
                                           TablatureNoteEvaluator(results_key=f'string-{tools.KEY_NOTE_ON}'),
                                           TablatureNoteEvaluator(offset_ratio=0.2,
                                                                  results_key=f'string-{tools.KEY_NOTE_OFF}'),
                                           # Continuous Pitch
                                           PitchListEvaluator(pitch_tolerances=tols),
                                           # String-Level Continuous Pitch )
                                           TablaturePitchListEvaluator(pitch_tolerances=tols,
                                                                       results_key=f'string-{tools.KEY_PITCHLIST}')])

    # Allocate the testing split for the fold
    test_splits = [GuitarSet.available_splits().pop(k)]

    # Define expected path for calculated features and ground-truth
    gset_cache = os.path.join('..', 'generated', 'data', 'val')

    # Create a dataset corresponding to the testing partition
    gset_test = GuitarSet(base_dir=None,
                          splits=test_splits,
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          num_frames=None,
                          data_proc=data_proc,
                          profile=model.profile,
                          reset_data=reset_data,
                          store_data=False,
                          save_loc=gset_cache,
                          semitone_radius=model.semitone_radius,
                          silence_activations=model.tablature_layer.silence_activations,
                          evaluation_extras=True)

    print(f'Evaluating fold {k} at checkpoint {checkpoint}...')

    # Compute the average results for the fold
    fold_results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

    with open(output_path, 'a') as json_file:
        # Add some other fields to the data before writing
        json_data = {'fold' : k,
                     'checkpoint' : checkpoint,
                     'time' : str(datetime.datetime.now()),
                     'results' : fold_results}
        # Write the fold results to the output file
        json.dump(json_data, json_file, sort_keys=True, indent=2)

    # Add the results to the tracked fold results
    results = append_results(results, fold_results)

with open(output_path, 'a') as json_file:
    # Add some other fields to the data before writing
    json_data = {'overall': {'time': str(datetime.datetime.now()),
                             'results': average_results(results)}}
    # Write the overall results to the output file
    json.dump(json_data, json_file, sort_keys=True, indent=2)
