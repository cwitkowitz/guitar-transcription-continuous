# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.models import LogisticTablatureEstimator
from amt_tools.models import TranscriptionModel, LogisticBank
from .tabcnn_variants import TabCNNLogisticContinuous
from .continuous_layers import CBernoulliBank, L2LogisticBank

import guitar_transcription_continuous.utils as utils

import amt_tools.tools as tools

# Regular imports
from torch import nn

import math

# TODO - add in onset head with switch?


class FretNet(TabCNNLogisticContinuous):
    """
    TODO
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, semitone_radius=0.5,
                 gamma=1, l2_layer=False, matrix_path=None, silence_activations=False, lmbda=1,
                 device='cpu'):
        """
        TODO
        """

        TranscriptionModel.__init__(self, dim_in, profile, in_channels, model_complexity, 9, device)

        self.semitone_radius = semitone_radius
        self.gamma = gamma

        # Initialize a flag to check whether to pad input features
        self.online = False

        # Number of filters for each convolutional layer
        nf1 = 16 * model_complexity
        nf2 = 32 * model_complexity
        nf3 = 48 * model_complexity

        # Kernel size for each convolutional layer
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks2

        # Padding amount for each convolutional layer
        pd1 = (1, 1)
        pd2 = (1, 0)
        pd3 = pd2

        # Reduction size for each pooling operation
        rd2 = (2, 1)
        rd3 = rd2

        # Dropout percentages for each dropout operation
        dp2 = 0.5
        dp3 = 0.25
        dpx = 0.10

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, nf1, ks1, padding=pd1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
            nn.Conv2d(nf1, nf1, ks1, padding=pd1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf1, nf2, ks2, padding=pd2),
            nn.BatchNorm2d(nf2),
            nn.ReLU(),
            nn.Conv2d(nf2, nf2, ks2, padding=pd2),
            nn.BatchNorm2d(nf2),
            nn.ReLU(),
            nn.MaxPool2d(rd2),# padding=pd2),
            nn.Dropout(dp2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf2, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU(),
            nn.Conv2d(nf3, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU(),
            nn.MaxPool2d(rd3),# padding=pd3),
            nn.Dropout(dp3)
        )

        def pooling_reduction(dim_in, times=1):
            return dim_in if times <= 0 else pooling_reduction(math.ceil(dim_in / 2), times - 1)

        features_dim_in = nf3 * pooling_reduction(dim_in, times=2)

        features_dim_int = features_dim_in // 2

        self.tablature_layer = LogisticTablatureEstimator(dim_in=features_dim_int,
                                                          profile=profile,
                                                          matrix_path=matrix_path,
                                                          silence_activations=silence_activations,
                                                          lmbda=lmbda,
                                                          device=device)

        self.tablature_head = nn.Sequential(
            nn.Linear(features_dim_in, features_dim_int),
            nn.ReLU(),
            nn.Dropout(dpx),
            self.tablature_layer
        )

        dim_out = self.profile.get_num_dofs() * self.profile.num_pitches

        if l2_layer:
            self.relative_layer = L2LogisticBank(features_dim_int, dim_out)
        else:
            self.relative_layer = CBernoulliBank(features_dim_int, dim_out)

        self.relative_head = nn.Sequential(
            nn.Linear(features_dim_in, features_dim_int),
            nn.ReLU(),
            nn.Dropout(dpx),
            self.relative_layer
        )

    def forward(self, feats):
        """
        TODO
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis,
        # so that each windowed group of frames is treated as one
        # independent sample. This is not done during pre-processing
        # in order to maintain consistency with the notion of batch size
        feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)

        # Obtain the feature embeddings from the model
        embeddings = self.conv3(self.conv2(self.conv1(feats)))

        # Flatten spatial features into one embedding
        embeddings = embeddings.flatten(1)
        # Size of the embedding
        embedding_size = embeddings.size(-1)
        # Restore proper batch dimension, unsqueezing sequence-frame axis
        embeddings = embeddings.view(batch_size, -1, embedding_size)

        # Process the embeddings with all the output heads
        output[tools.KEY_TABLATURE] = self.tablature_head(embeddings).pop(tools.KEY_TABLATURE)
        output[utils.KEY_TABLATURE_REL] = self.relative_head(embeddings)

        return output
