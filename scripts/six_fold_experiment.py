# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
#from tabcnn_variants import TabCNNLogisticContinuous as TabCNN
from GuitarSet import GuitarSetPlus as GuitarSet
from amt_tools.models import TabCNN
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOnsetsWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
from inference import StackedPitchListTablatureWrapper
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate, \
                               append_results, \
                               average_results
from evaluators import *

import amt_tools.tools as tools

import constants

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

torch.multiprocessing.set_start_method('spawn', force=True)

EX_NAME = '_'.join([TabCNN.model_name(),
                    GuitarSet.dataset_name(),
                    CQT.features_name()])

ex = Experiment('TabCNN (Continuous) w/ CQT on GuitarSet w/ 6-fold Cross Validation')


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

    # The initial learning rate
    learning_rate = 1.0

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # Flag to use one split for validation
    validation_split = True

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    #root_dir = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def tabcnn_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                     batch_size, learning_rate, gpu_id, reset_data, validation_split,
                     seed, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile()

    # Processing parameters
    dim_in = 192
    model_complexity = 1
    semitone_width = 1.0

    # Create the cqt data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([
        # Discrete tablature -> stacked multi pitch array
        TablatureWrapper(profile=profile),
        # Stacked multi pitch array -> stacked onsets array
        StackedOnsetsWrapper(profile=profile),
        # Stacked multi pitch array -> stacked offsets array
        StackedOffsetsWrapper(profile=profile),
        # Stacked multi pitch array -> stacked notes
        StackedNoteTranscriber(profile=profile),
        # Continuous tablature arrays -> stacked pitch list
        StackedPitchListTablatureWrapper(profile=profile,
                                         multi_pitch_key=tools.KEY_TABLATURE,
                                         multi_pitch_rel_key=constants.KEY_TABLATURE_REL)])

    # Fractions of semitone to use for tolerances when evaluating pitch lists
    tols = np.array([2, 4, 8, 16], dtype=float) ** -1

    # Initialize the evaluation pipeline
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           OnsetsEvaluator(),
                                           OffsetsEvaluator(),
                                           MultipitchEvaluator(),
                                           TablatureOnsetEvaluator(profile=profile,
                                                                   results_key=f'string-{tools.KEY_ONSETS}'),
                                           TablatureOffsetEvaluator(profile=profile,
                                                                    results_key=f'string-{tools.KEY_OFFSETS}'),
                                           TablatureEvaluator(profile=profile),
                                           SoftmaxAccuracy(),
                                           NoteEvaluator(results_key=tools.KEY_NOTE_ON),
                                           NoteEvaluator(offset_ratio=0.2,
                                                         results_key=tools.KEY_NOTE_OFF),
                                           TablatureNoteEvaluator(results_key=f'string-{tools.KEY_NOTE_ON}'),
                                           TablatureNoteEvaluator(offset_ratio=0.2,
                                                                  results_key=f'string-{tools.KEY_NOTE_OFF}'),
                                           PitchListEvaluator(pitch_tolerances=tols),])
                                           #TablaturePitchListEvaluator(pitch_tolerances=tols,
                                           #                            results_key=f'string-{tools.KEY_PITCHLIST}')])

    # Keep all cached data/features here
    gset_cache = os.path.join('..', 'generated', 'data')
    gset_cache_train = os.path.join(gset_cache, 'train') # No extras
    gset_cache_val = os.path.join(gset_cache, 'val') # Includes extras

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    try: # TODO - for Valohai, remove try/catch
        # Initialize an empty dictionary to hold the average results across fold
        results = dict()

        # Perform each fold of cross-validation
        for k in range(6):
            # Seed everything with the same seed
            tools.seed_everything(seed)

            print('--------------------')
            print(f'Fold {k}:')

            # Allocate training/testing splits
            train_splits = splits.copy()
            test_splits = [train_splits.pop(k)]

            if validation_split:
                # Allocate validation split
                val_splits = [train_splits.pop(k - 1)]

            print('Loading training partition...')

            # Create a dataset corresponding to the training partition
            gset_train = GuitarSet(base_dir=None,
                                   splits=train_splits,
                                   hop_length=hop_length,
                                   sample_rate=sample_rate,
                                   num_frames=num_frames,
                                   data_proc=data_proc,
                                   profile=profile,
                                   save_loc=gset_cache_train,
                                   semitone_width=semitone_width,
                                   augment=False,
                                   evaluation_extras=False)

            # Create a PyTorch data loader for the dataset
            train_loader = DataLoader(dataset=gset_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      drop_last=True)

            print(f'Loading testing partition (player {test_splits[0]})...')

            # Create a dataset corresponding to the testing partition
            gset_test = GuitarSet(base_dir=None,
                                  splits=test_splits,
                                  hop_length=hop_length,
                                  sample_rate=sample_rate,
                                  num_frames=None,
                                  data_proc=data_proc,
                                  profile=profile,
                                  store_data=(not validation_split),
                                  save_loc=gset_cache_val,
                                  semitone_width=semitone_width,
                                  evaluation_extras=True)

            if validation_split:
                print(f'Loading validation partition (player {val_splits[0]})...')

                # Create a dataset corresponding to the validation partition
                gset_val = GuitarSet(base_dir=None,
                                     splits=val_splits,
                                     hop_length=hop_length,
                                     sample_rate=sample_rate,
                                     num_frames=None,
                                     data_proc=data_proc,
                                     profile=profile,
                                     store_data=True,
                                     save_loc=gset_cache_val,
                                     semitone_width=semitone_width,
                                     evaluation_extras=True)
            else:
                # Validate on the test set
                gset_val = gset_test

            # Initialize a new instance of the model
            tabcnn = TabCNN(dim_in=dim_in,
                            profile=profile,
                            in_channels=data_proc.get_num_channels(),
                            model_complexity=model_complexity,
                            #lmbda=10,
                            #semitone_width=semitone_width,
                            #gamma=10,
                            device=gpu_id)
            tabcnn.change_device()
            tabcnn.train()

            # Initialize a new optimizer for the model parameters
            #optimizer = torch.optim.Adam(tabcnn.parameters(), lr=5E-4)
            optimizer = torch.optim.Adadelta(tabcnn.parameters(), learning_rate)

            print('Training model...')

            # Create a log directory for the training experiment
            model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

            # Set validation patterns for training
            validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

            # Train the model
            tabcnn = train(model=tabcnn,
                           train_loader=train_loader,
                           optimizer=optimizer,
                           iterations=iterations,
                           checkpoints=checkpoints,
                           log_dir=model_dir,
                           val_set=gset_val,
                           estimator=validation_estimator,
                           evaluator=validation_evaluator)

            print('Transcribing and evaluating test partition...')

            # Add a save directory to the evaluators and reset the patterns
            validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
            validation_evaluator.set_patterns(None)

            # Get the average results for the fold
            fold_results = validate(tabcnn, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

            # Add the results to the tracked fold results
            results = append_results(results, fold_results)

            # Reset the results for the next fold
            validation_evaluator.reset_results()

            # Log the fold results for the fold in metrics.json
            ex.log_scalar('Fold Results', fold_results, k)

        # Log the average results for the fold in metrics.json
        ex.log_scalar('Overall Results', average_results(results), 0)

    finally: # TODO - the following is for Valohai, remove
        import time
        #print('Waiting 2 minutes to allow files to finish updating...')
        #time.sleep(120) # wait 2 minutes to avoid zipping before files update
        #tools.zip_and_save(root_dir, os.path.join(os.path.dirname(root_dir), EX_NAME + '.zip'))
