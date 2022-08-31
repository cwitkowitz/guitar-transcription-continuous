# My imports
from guitar_transcription_inhibition.models import TabCNNLogistic
from amt_tools.models import TabCNN, LogisticBank
from continuous_layers import CBernoulliBank

import guitar_transcription_continuous.constants as constants

import amt_tools.tools as tools

# Regular imports
import torch


class TabCNNMultipitch(TabCNN):
    """
    Implements TabCNN for discrete multipitch estimation instead of tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and replace the final layer.

        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Break off the output layer to establish an explicit reference
        tablature_layer, self.dense = self.dense[-1], self.dense[:-1]

        # Determine the number of input neurons to the current tablature layer
        n_neurons = tablature_layer.dim_in

        # Determine the number of distinct pitches
        n_multipitch = self.profile.get_range_len()

        # Create a layer for multipitch estimation to replace the tablature layer
        self.multipitch_layer = LogisticBank(n_neurons, n_multipitch)

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ multipitch Tensor (B x T x O)
          Dictionary containing model output
          B - batch size,
          T - number of time steps (frames),
          O - number of discrete pitches (dim_out)
        """

        # Run the standard steps to extract output embeddings
        output = super().forward(feats)

        # Extract the embeddings from the output dictionary (labeled as tablature)
        embeddings = output.pop(tools.KEY_TABLATURE)

        # Process the embeddings with the multipitch output layer
        output[tools.KEY_MULTIPITCH] = self.multipitch_layer(embeddings)

        return output

    def post_proc(self, batch):
        """
        Calculate multipitch loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multipitch and potentially loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain the multipitch estimation
        multipitch_est = output[tools.KEY_MULTIPITCH]

        # Keep track of loss
        total_loss = 0

        # Check to see if ground-truth multipitch is available
        if tools.KEY_MULTIPITCH in batch.keys():
            # Calculate the loss and add it to the total (note that loss is computed
            # w.r.t. the multi pitch adjusted to match the ground-truth pitch contours)
            total_loss += self.multipitch_layer.get_loss(multipitch_est,
                                                         batch[constants.KEY_MULTIPITCH_ADJ])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_PITCH : total_loss.clone(),
                                      tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize multipitch estimation by taking sigmoid and thresholding activations
        output[tools.KEY_MULTIPITCH] = self.multipitch_layer.finalize_output(multipitch_est, 0.5)

        return output


class TabCNNContinuousMultipitch(TabCNNMultipitch):
    """
    Implements TabCNN for continuous multipitch estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, semitone_width=0.5, gamma=1, device='cpu'):
        """
        Initialize the model and include an additional output
        layer for the estimation of relative pitch deviation.

        Parameters
        ----------
        See TabCNN class for others...

        semitone_width : float
          Scaling factor for relative pitch estimates
        gamma : float
          Inverse scaling multiplier for the discrete multipitch loss
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        self.semitone_width = semitone_width
        self.gamma = gamma

        # Create another output layer to estimate relative pitch deviation
        self.relative_layer = CBernoulliBank(self.multipitch_layer.dim_in,
                                             self.multipitch_layer.dim_out)

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ multipitch/relative Tensors (both B x T x O)
          Dictionary containing model output
          B - batch size,
          T - number of time steps (frames),
          O - number of discrete pitches (dim_out)
        """

        # Run the standard steps to extract output embeddings
        output = TabCNN.forward(self, feats)

        # Extract the embeddings from the output dictionary (labeled as tablature)
        embeddings = output.pop(tools.KEY_TABLATURE)

        # Process the embeddings with both output layers
        output[tools.KEY_MULTIPITCH] = self.multipitch_layer(embeddings)
        output[constants.KEY_MULTIPITCH_REL] = self.relative_layer(embeddings)

        return output

    def post_proc(self, batch):
        """
        Calculate multipitch and relative loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multipitch, relative deviation, and potentially loss
        """

        # Calculate multipitch loss
        output = super().post_proc(batch)

        # Obtain the relative pitch deviation estimates
        relative_est = output[constants.KEY_MULTIPITCH_REL]

        # Unpack the loss if it exists
        loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

        if loss is None:
            # Create a new dictionary to hold the loss
            loss = {}

        # Check to see if ground-truth relative multipitch is available
        if constants.KEY_MULTIPITCH_REL in batch.keys():
            # Normalize the ground-truth relative multi pitch data (-1, 1)
            normalized_relative_multi_pitch = batch[constants.KEY_MULTIPITCH_REL] / self.semitone_width
            # Compress the relative multi pitch data to fit within sigmoid range (0, 1)
            compressed_relative_multi_pitch = (normalized_relative_multi_pitch + 1) / 2
            # Compute the loss for the relative pitch deviation
            relative_loss = self.relative_layer.get_loss(relative_est, compressed_relative_multi_pitch)
            # Add the relative pitch deviation loss to the tracked loss dictionary
            loss[constants.KEY_LOSS_PITCH_REL] = relative_loss
            # Add the relative pitch deviation loss to the (scaled) total loss
            total_loss = (1 / self.gamma) * total_loss + relative_loss

        # Determine if loss is being tracked
        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Normalize relative multi pitch estimates between (-1, 1)
        relative_est = 2 * self.relative_layer.finalize_output(relative_est) - 1

        # Finalize the estimates by re-scaling to the chosen semitone width
        output[constants.KEY_MULTIPITCH_REL] = self.semitone_width * relative_est

        return output


def switch_keys_dict(d, k1, k2):
    """
    Switch the keys for two entries in a dictionary.

    Parameters
    ----------
    d : dict
      Dictionary of interest
    k1 : object
      Key for first entry
    k2 : object
      Key for second entry

    Returns
    ----------
    success : bool
      Indicator of whether operation was successful
    """

    # Default status to unsuccessful
    success = False

    # Make sure the two keys are present in the dictionary
    if tools.query_dict(d, k1) and tools.query_dict(d, k2):
        # Obtain the entries
        e1, e2 = d[k1], d[k2]
        # Switch the keys
        d[k1], d[k2] = e2, e1
        # Mark operation as successful
        success = True

    return success


class TabCNNLogisticContinuous(TabCNNLogistic):
    """
    Implements TabCNN w/ logistic formulation for continuous tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1,
                 semitone_width=0.5, gamma=1, lmbda=1, device='cpu'):
        """
        Initialize the model and include an additional output
        layer for the estimation of relative pitch deviation.

        Parameters
        ----------
        See TabCNN/LogisticTablatureEstimator class for others...

        semitone_width : float
          Scaling factor for relative pitch estimates
        gamma : float
          Inverse scaling multiplier for the discrete tablature loss
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, None, True, lmbda, device)

        self.semitone_width = semitone_width
        self.gamma = gamma

        # Extract tablature parameters
        num_strings = self.profile.get_num_dofs()
        num_pitches = self.profile.num_pitches

        # Create another output layer to estimate relative pitch deviation
        self.relative_layer = CBernoulliBank(self.tablature_layer.dim_in, num_strings * num_pitches)

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ tablature/relative Tensors (both B x T x O)
          Dictionary containing continuous tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Run the standard steps
        output = TabCNN.forward(self, feats)

        # Extract the embeddings from the output dictionary (labeled as tablature)
        embeddings = output.pop(tools.KEY_TABLATURE)

        # Process the embeddings with both output layers
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings).pop(tools.KEY_TABLATURE)
        output[constants.KEY_TABLATURE_REL] = self.relative_layer(embeddings)

        return output

    def post_proc(self, batch):
        """
        Calculate tablature/inhibition/relative loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature, relative deviation, and potentially loss
        """

        # Switch keys for original and adjusted ground-truth tablature
        # so tablature loss is computed w.r.t. the adjusted targets
        switch_keys_dict(batch, tools.KEY_TABLATURE, constants.KEY_TABLATURE_ADJ)

        # Call the post-processing method of the tablature layer
        output = super().post_proc(batch)

        # Switch back the keys for proper evaluation of tablature
        # TODO - wouldn't need to be this complicated if batch could be deep-copied in this scope
        switch_keys_dict(batch, tools.KEY_TABLATURE, constants.KEY_TABLATURE_ADJ)

        # Obtain the relative pitch deviation estimates
        relative_est = output[constants.KEY_TABLATURE_REL]

        # Unpack the loss if it exists
        loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

        if loss is None:
            # Create a new dictionary to hold the loss
            loss = {}

        # Check to see if ground-truth relative tablature is available
        if constants.KEY_TABLATURE_REL in batch.keys():
            # Normalize the ground-truth relative tablature data (-1, 1)
            normalized_relative_tablature = batch[constants.KEY_TABLATURE_REL] / self.semitone_width
            # Compress the relative tablature data to fit within sigmoid range (0, 1)
            compressed_relative_tablature = (normalized_relative_tablature + 1) / 2
            # Compute the loss for the relative pitch deviation
            relative_loss = self.relative_layer.get_loss(relative_est, compressed_relative_tablature)
            # Add the relative pitch deviation loss to the tracked loss dictionary
            loss[constants.KEY_LOSS_TABS_REL] = relative_loss
            # Add the relative pitch deviation loss to the (scaled) total loss
            total_loss = (1 / self.gamma) * total_loss + relative_loss

        # Determine if loss is being tracked
        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Normalize relative tablature estimates between (-1, 1)
        relative_est = 2 * self.relative_layer.finalize_output(relative_est) - 1

        # Finalize the estimates by re-scaling to the chosen semitone width
        output[constants.KEY_TABLATURE_REL] = self.semitone_width * relative_est

        return output
