# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_continuous.models import TabCNN, TabCNNLogisticContinuous, FretNet
from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet
from guitar_transcription_continuous.estimators import StackedPitchListTablatureWrapper
from guitar_transcription_continuous.evaluators import *
from amt_tools.features import CQT, HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOnsetsWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate, \
                               append_results, \
                               average_results
from amt_tools.train import train

import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import librosa
import torch
import time
import os

torch.multiprocessing.set_start_method('spawn', force=True)

EX_NAME = '_'.join([FretNet.model_name(),
                    GuitarSet.dataset_name(),
                    HCQT.features_name()])

ex = Experiment('FretNet w/ HCQT on GuitarSet w/ 6-fold Cross Validation')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 200

    # Number of training iterations to conduct
    iterations = 2500

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 25

    # Number of samples to gather for a batch
    batch_size = 30

    # The fixed or initial learning rate
    learning_rate = 5E-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate
    # features (useful if testing out different parameters)
    reset_data = False

    # Flag to set aside one split for validation
    validation_split = True

    # Whether to perform data augmentation (pitch shifting) during training
    augment_data = False

    # Amount of semitones in each direction modeled for each note
    semitone_radius = 1.0

    # Flag to use rotarized pitch deviations for ground-truth
    rotarize_deviations = False

    # Switch to select type of continuous output layer for relative pitch prediction
    # (0 - Continuous Bernoulli | 1 - MSE | None - disable relative pitch prediction)
    cont_layer = 0

    # Multiplier for inhibition loss if applicable
    lmbda = 10

    # Path to inhibition matrix if applicable
    matrix_path = None

    # Flag to include an activation for silence in applicable output layers
    silence_activations = True

    # Inverse scaling multiplier for discrete tablature / ihibition loss if applicable
    gamma = 10

    # Flag to include an additional onset head in FretNet
    estimate_onsets = True

    # Flag to use HCQT features insead of CQT
    harmonic_dimension = True

    # The random seed for this experiment
    seed = 0

    # Switch to manage different file schema (0 - local | 1 - lab machine | 2 - valohai)
    file_layout = 0

    # Create the root directory for the experiment files
    if file_layout == 2:
        root_dir = os.path.join(os.getenv('VH_OUTPUTS_DIR'), EX_NAME)
    elif file_layout == 1:
        root_dir = os.path.join('/', 'storage', 'frank', 'continuous_experiments', EX_NAME)
    else:
        root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def fretnet_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints, batch_size, learning_rate,
                      gpu_id, reset_data, validation_split, augment_data, semitone_radius, rotarize_deviations,
                      cont_layer, lmbda, matrix_path, silence_activations, gamma, estimate_onsets, harmonic_dimension,
                      seed, file_layout, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    if harmonic_dimension:
        # Create an HCQT feature extraction module comprising
        # the first five harmonics and a sub-harmonic, where each
        # harmonic transform spans 4 octaves w/ 3 bins per semitone
        data_proc = HCQT(sample_rate=sample_rate,
                         hop_length=hop_length,
                         fmin=librosa.note_to_hz('E2'),
                         harmonics=[0.5, 1, 2, 3, 4, 5],
                         n_bins=144, bins_per_octave=36)
    else:
        # Create a CQT feature extraction module
        # spanning 8 octaves w/ 2 bins per semitone
        data_proc = CQT(sample_rate=sample_rate, hop_length=hop_length, n_bins=192, bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([
        # Discrete tablature -> stacked multi pitch array
        TablatureWrapper(profile=profile),
        # Stacked multi pitch array -> stacked offsets array
        StackedOffsetsWrapper(profile=profile),
        # Stacked multi pitch array -> stacked notes
        StackedNoteTranscriber(profile=profile),
        # Continuous tablature arrays -> stacked pitch list
        StackedPitchListTablatureWrapper(profile=profile,
                                         multi_pitch_key=tools.KEY_TABLATURE,
                                         multi_pitch_rel_key=utils.KEY_TABLATURE_REL)])

    if not estimate_onsets:
        # Infer the onsets directly from the multi pitch data
        validation_estimator.estimators.insert(1,
            # Stacked multi pitch array -> stacked onsets array
            StackedOnsetsWrapper(profile=profile))

    # Define tolerances to use when evaluating pitch lists
    tols = [1/2, 1/4, 1/8, 1/16] # semitones

    # Initialize the evaluation pipeline - ( Loss,
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           # Onsets/Offsets,
                                           OnsetsEvaluator(),
                                           OffsetsEvaluator(),
                                           # Multi Pitch,
                                           MultipitchEvaluator(),
                                           # String-Level Onsets/Offsets,
                                           TablatureOnsetEvaluator(profile=profile,
                                                                   results_key=f'string-{tools.KEY_ONSETS}'),
                                           TablatureOffsetEvaluator(profile=profile,
                                                                    results_key=f'string-{tools.KEY_OFFSETS}'),
                                           # Tablature (String-Level Multi Pitch),
                                           TablatureEvaluator(profile=profile),
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
                                           PitchListEvaluator(pitch_tolerances=tols)])

    # Build the path to GuitarSet
    if file_layout == 2:
        gset_base_dir = os.path.join(os.getenv('VH_INPUTS_DIR'),
                                     'data', 'frank-internship',
                                     'active', 'GuitarSet')
    elif file_layout == 1:
        gset_base_dir = os.path.join('/', 'storage', 'frank', 'GuitarSet')
    else:
        gset_base_dir = None

    # Keep all cached data/features here
    if file_layout == 1:
        gset_cache = os.path.join('/', 'storageNVME', 'frank')
    else:
        gset_cache = os.path.join('..', 'generated', 'data')
    gset_cache_train = os.path.join(gset_cache, 'train') # No extras
    gset_cache_val = os.path.join(gset_cache, 'val') # Includes extras

    try:
        # Initialize an empty dictionary to hold the average results across folds
        results = dict()

        # Perform six-fold cross-validation
        for k in range(6):
            print('--------------------')
            print(f'Fold {k}:')

            # Seed everything with the same seed
            tools.seed_everything(seed)

            # Set validation patterns for logging during training
            validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

            # Allocate training/testing splits
            train_splits = GuitarSet.available_splits()
            test_splits = [train_splits.pop(k)]

            if validation_split:
                # Allocate validation split
                val_splits = [train_splits.pop(k - 1)]

            if not augment_data:
                print('Loading training partition...')

            # Create a dataset corresponding to the training partition
            gset_train = GuitarSet(base_dir=gset_base_dir,
                                   splits=train_splits,
                                   hop_length=hop_length,
                                   sample_rate=sample_rate,
                                   num_frames=num_frames,
                                   data_proc=data_proc,
                                   profile=profile,
                                   reset_data=(reset_data and k == 0),
                                   save_loc=gset_cache_train,
                                   semitone_radius=semitone_radius,
                                   rotarize_deviations=rotarize_deviations,
                                   augment=augment_data,
                                   silence_activations=silence_activations,
                                   evaluation_extras=False)

            # Create a PyTorch data loader for the dataset
            train_loader = DataLoader(dataset=gset_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=4 * int(augment_data),
                                      drop_last=True)

            print(f'Loading testing partition (player {test_splits[0]})...')

            # Create a dataset corresponding to the testing partition
            gset_test = GuitarSet(base_dir=gset_base_dir,
                                  splits=test_splits,
                                  hop_length=hop_length,
                                  sample_rate=sample_rate,
                                  num_frames=None,
                                  data_proc=data_proc,
                                  profile=profile,
                                  reset_data=(reset_data and k == 0),
                                  store_data=(not validation_split),
                                  save_loc=gset_cache_val,
                                  semitone_radius=semitone_radius,
                                  rotarize_deviations=rotarize_deviations,
                                  silence_activations=silence_activations,
                                  evaluation_extras=True)

            if validation_split:
                print(f'Loading validation partition (player {val_splits[0]})...')

                # Create a dataset corresponding to the validation partition
                gset_val = GuitarSet(base_dir=gset_base_dir,
                                     splits=val_splits,
                                     hop_length=hop_length,
                                     sample_rate=sample_rate,
                                     num_frames=None,
                                     data_proc=data_proc,
                                     profile=profile,
                                     reset_data=(reset_data and k == 0),
                                     store_data=True,
                                     save_loc=gset_cache_val,
                                     semitone_radius=semitone_radius,
                                     rotarize_deviations=rotarize_deviations,
                                     silence_activations=silence_activations,
                                     evaluation_extras=True)
            else:
                # Perform validation on the testing partition
                gset_val = gset_test

            print('Initializing model...')

            # Initialize a new instance of the model
            fretnet = FretNet(dim_in=data_proc.get_feature_size(),
                              profile=profile,
                              in_channels=data_proc.get_num_channels(),
                              lmbda=lmbda,
                              matrix_path=matrix_path,
                              silence_activations=silence_activations,
                              semitone_radius=semitone_radius,
                              gamma=gamma,
                              cont_layer=cont_layer,
                              estimate_onsets=estimate_onsets,
                              device=gpu_id)
            fretnet.change_device()
            fretnet.train()

            # Initialize a new optimizer for the model parameters
            optimizer = torch.optim.Adam(fretnet.parameters(), lr=learning_rate)

            print('Training model...')

            # Train the model
            fretnet = train(model=fretnet,
                            train_loader=train_loader,
                            optimizer=optimizer,
                            iterations=iterations,
                            checkpoints=checkpoints,
                            log_dir=os.path.join(root_dir, 'models', 'fold-' + str(k)),
                            val_set=gset_val,
                            estimator=validation_estimator,
                            evaluator=validation_evaluator)

            print('Transcribing and evaluating test partition...')

            # Add a save directory to the evaluators
            validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
            # Reset the evaluation patterns to log everything
            validation_evaluator.set_patterns(None)

            # Compute the average results for the fold
            fold_results = validate(fretnet, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

            # Add the results to the tracked fold results
            results = append_results(results, fold_results)

            # Reset the evaluators for the next fold
            validation_evaluator.reset_results()

            # Log the average results for the fold in metrics.json
            ex.log_scalar('Fold Results', fold_results, k)

        # Log the average results across all folds in metrics.json
        ex.log_scalar('Overall Results', average_results(results), 0)

    finally:
        # Wait 1 minute to avoid zipping before files finish updating
        print('Waiting 1 minute to allow files to finish updating...')
        # Pause execution for 1 minute
        time.sleep(60)
        # Construct a path to save all generated materials
        zip_path = os.path.join(os.path.dirname(root_dir), EX_NAME + '.zip')
        # Save all experiment files as a single .zip file
        tools.zip_and_save(root_dir, zip_path)
