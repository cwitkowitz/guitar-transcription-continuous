# My imports
from amt_tools.models import TabCNN, LogisticBank

import amt_tools.tools as tools

import constants


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
            output[tools.KEY_LOSS] = {tools.KEY_MULTIPITCH : total_loss.clone(),
                                      tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize multipitch estimation by taking sigmoid and thresholding activations
        output[tools.KEY_MULTIPITCH] = self.multipitch_layer.finalize_output(multipitch_est, 0.5)

        return output


class TabCNNContinuousMultipitch(TabCNNMultipitch):
    """
    Implements TabCNN for continuous multipitch estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and include an additional output
        layer for the estimation of relative pitch deviation.

        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Create another output layer to estimate relative pitch deviation
        self.relative_layer = LogisticBank(self.multipitch_layer.dim_in,
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
            # Compute the loss for the relative pitch deviation
            relative_loss = self.relative_layer.get_loss(relative_est,
                                                         batch[constants.KEY_MULTIPITCH_REL])
            # Add the relative pitch deviation loss to the tracked loss dictionary
            loss[constants.KEY_LOSS_PITCH_REL] = relative_loss
            # Add the relative pitch deviation loss to the total loss
            total_loss += relative_loss

        # Determine if loss is being tracked
        if total_loss:
            # Add the loss to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Finalize the relative pitch deviation estimates by zero-centering and scaling between -1 and 1
        output[constants.KEY_MULTIPITCH_REL] = 2 * (self.relative_layer.finalize_output(relative_est) - 0.5)

        return output
