# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.metrics import FalseAlarmErrors
from yousician_private import SyntheticGuitar_V1
from GuitarSet import GuitarSet
from amt_tools.models import TabCNN
from amt_tools.features import CQT

from amt_tools.train import train
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join(['TabCNN_SynthV0_Benchmark_4'])

ex = Experiment('Tablature Transcription Trained on Synthetic Data Evaluated on Synthetic Data')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 200

    # Number of training iterations to conduct
    iterations = 500

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 25

    # Number of samples to gather for a batch
    batch_size = 50

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
    #root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)
    root_dir = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def experiment(sample_rate, hop_length, num_frames, iterations, checkpoints,
               batch_size, learning_rate, gpu_id, reset_data, validation_split,
               seed, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Processing parameters
    dim_in = 192
    model_complexity = 1

    # Create the data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile),
                                           NoteTranscriber(profile=profile)])

    # Initialize the evaluation pipeline
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=profile),
                                           FalseAlarmErrors(profile=profile),
                                           SoftmaxAccuracy(key=tools.KEY_TABLATURE),
                                           NoteEvaluator(key=tools.KEY_NOTE_ON)])

    # Keep all cached data/features here
    data_cache = os.path.join('..', 'generated', 'data')

    # Initialize an empty dictionary to hold the average results across fold
    results = dict()

    # Seed everything with the same seed
    tools.seed_everything(seed)

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    synth_train = SyntheticGuitar_V1(base_dir=None,
                                     splits=['train'],
                                     hop_length=hop_length,
                                     sample_rate=sample_rate,
                                     num_frames=num_frames,
                                     data_proc=data_proc,
                                     profile=profile,
                                     store_data=False,
                                     save_data=True,
                                     save_loc=data_cache)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=synth_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)

    print(f'Loading testing partition...')

    # Create a dataset corresponding to the testing partition
    """gset_test = GuitarSet(base_dir=None,
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          num_frames=None,
                          data_proc=data_proc,
                          profile=profile,
                          store_data=True,
                          save_data=False,
                          save_loc=data_cache)"""
    synth_test = SyntheticGuitar_V1(base_dir=None,
                                    splits=['test'],
                                    hop_length=hop_length,
                                    sample_rate=sample_rate,
                                    num_frames=None,
                                    data_proc=data_proc,
                                    profile=profile,
                                    store_data=True,
                                    save_data=False,
                                    save_loc=data_cache)

    print('Initializing model...')

    # Initialize a new instance of the model
    tabcnn = TabCNN(dim_in=dim_in,
                    profile=profile,
                    in_channels=data_proc.get_num_channels(),
                    model_complexity=model_complexity,
                    device=gpu_id)
    tabcnn.change_device()
    tabcnn.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adadelta(tabcnn.parameters(), learning_rate)

    print('Training model...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Set validation patterns for training
    #validation_evaluator.set_patterns(['loss', 'f1', 'tdr', 'acc', 'error'])

    # Train the model
    tabcnn = train(model=tabcnn,
                   train_loader=train_loader,
                   optimizer=optimizer,
                   iterations=iterations,
                   checkpoints=checkpoints,
                   log_dir=model_dir,
                   val_set=synth_test,
                   estimator=validation_estimator,
                   evaluator=validation_evaluator)

    print('Transcribing and evaluating test partition...')

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the fold
    fold_results = validate(tabcnn, synth_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Add the results to the tracked fold results
    results = append_results(results, fold_results)

    # Reset the results for the next fold
    validation_evaluator.reset_results()

    # Log the average results for the fold in metrics.json
    ex.log_scalar('Overall Results', results, 0)

    # TODO - the following is for Valohai, remove
    import time
    print('Waiting 5 minutes for files to finish updating...')
    time.sleep(300) # wait 5 minutes to avoid zipping before files update
    tools.zip_and_save(root_dir, os.path.join(tools.DEFAULT_EXPERIMENTS_DIR, EX_NAME + '.zip'))
