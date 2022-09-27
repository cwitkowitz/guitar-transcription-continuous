# My imports
from guitar_transcription_inhibition.models import TabCNNLogistic
from amt_tools.models import TabCNN as _TabCNN, LogisticBank
from .continuous_layers import CBernoulliBank, L2LogisticBank

import guitar_transcription_continuous.utils as utils

import amt_tools.tools as tools

# Regular imports
import torch


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


class TabCNN(_TabCNN):
    """
    Simple wrapper for TabCNN to use adjusted tablature targets for training.
    """

    def post_proc(self, batch):
        """
        Calculate loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature as well as loss
        """

        # Switch keys for original and adjusted ground-truth tablature
        # so tablature loss is computed w.r.t. the adjusted targets
        switch_keys_dict(batch, tools.KEY_TABLATURE, utils.KEY_TABLATURE_ADJ)

        # Convert the adjusted tablature from logistic to tablature representation
        batch[tools.KEY_TABLATURE] = tools.logistic_to_tablature(batch[tools.KEY_TABLATURE], self.profile, True)

        # Call the post-processing method of the parent
        output = super().post_proc(batch)

        # Convert the back the adjusted tablature from tablature to representation
        batch[tools.KEY_TABLATURE] = tools.tablature_to_logistic(batch[tools.KEY_TABLATURE], self.profile, True)

        # Switch back the keys for proper evaluation of tablature
        # TODO - wouldn't need to be this complicated if batch could be deep-copied in this scope
        switch_keys_dict(batch, tools.KEY_TABLATURE, utils.KEY_TABLATURE_ADJ)

        return output


class TabCNNLogisticContinuous(TabCNNLogistic):
    """
    Implements TabCNN w/ logistic formulation for continuous tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, semitone_radius=0.5,
                 gamma=1, cont_layer=1, matrix_path=None, silence_activations=False, lmbda=1,
                 device='cpu'):
        """
        Initialize the model and include an additional output
        layer for the estimation of relative pitch deviation.

        Parameters
        ----------
        See TabCNN/LogisticTablatureEstimator class for others...

        semitone_radius : float
          Scaling factor for relative pitch estimates
        gamma : float
          Inverse scaling multiplier for the discrete tablature loss
        cont_layer : bool
          Switch to select type of continuous output layer for relative pitch prediction
          (0 - Continuous Bernoulli | 1 - MSE | None - disable relative pitch prediction)
        """

        super().__init__(dim_in, profile, in_channels, model_complexity,
                         matrix_path, silence_activations, lmbda, device)

        self.semitone_radius = semitone_radius
        self.gamma = gamma
        self.cont_layer = cont_layer

        # Determine output dimensionality when not explicitly modeling silence
        tablature_dim_out = self.profile.get_num_dofs() * self.profile.num_pitches

        if self.cont_layer is not None:
            # Create another output layer to estimate relative pitch deviation
            if cont_layer:
                # Train continuous relative pitch layer with MSE loss
                self.relative_layer = L2LogisticBank(self.tablature_layer.dim_in, tablature_dim_out)
            else:
                # Train continuous relative pitch layer with Continuous Bernoulli loss
                self.relative_layer = CBernoulliBank(self.tablature_layer.dim_in, tablature_dim_out)

    def forward(self, feats):
        """
        Perform the main processing steps for the variant.

        Parameters
        ----------
        feats : Tensor (B x T x C x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          C - number of channels in features
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

        # Process embeddings with discrete tablature layer
        output[tools.KEY_TABLATURE] = self.tablature_layer(embeddings).pop(tools.KEY_TABLATURE)

        if self.cont_layer is not None:
            # Process embeddings with relative tablature layer
            output[utils.KEY_TABLATURE_REL] = self.relative_layer(embeddings)

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
        switch_keys_dict(batch, tools.KEY_TABLATURE, utils.KEY_TABLATURE_ADJ)

        # Call the post-processing method of the parent
        output = super().post_proc(batch)

        # Switch back the keys for proper evaluation of tablature
        # TODO - wouldn't need to be this complicated if batch could be deep-copied in this scope
        switch_keys_dict(batch, tools.KEY_TABLATURE, utils.KEY_TABLATURE_ADJ)

        if self.cont_layer is not None:
            # Obtain the relative pitch deviation estimates
            relative_est = output[utils.KEY_TABLATURE_REL]

            # Unpack the loss if it exists
            loss = tools.unpack_dict(output, tools.KEY_LOSS)
            total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

            if loss is None:
                # Create a new dictionary to hold the loss
                loss = {}

            # Check to see if ground-truth relative tablature is available
            if utils.KEY_TABLATURE_REL in batch.keys():
                # Normalize the ground-truth relative tablature data (-1, 1)
                normalized_relative_tablature = batch[utils.KEY_TABLATURE_REL] / self.semitone_radius
                # Compress the relative tablature data to fit within sigmoid range (0, 1)
                compressed_relative_tablature = (normalized_relative_tablature + 1) / 2
                # Compute the loss for the relative pitch deviation
                relative_loss = self.relative_layer.get_loss(relative_est, compressed_relative_tablature)
                # Add the relative pitch deviation loss to the tracked loss dictionary
                loss[utils.KEY_LOSS_TABS_REL] = relative_loss
                # Scale down the current total loss if Continuous Bernoulli layer
                total_loss /= (self.gamma ** int(not self.cont_layer))
                # Add the scaled relative pitch deviation loss to the total loss
                total_loss += (self.gamma ** int(self.cont_layer)) * relative_loss

            # Determine if loss is being tracked
            if total_loss:
                # Add the loss to the output dictionary
                loss[tools.KEY_LOSS_TOTAL] = total_loss
                output[tools.KEY_LOSS] = loss

            # Normalize relative tablature estimates between (-1, 1)
            relative_est = 2 * self.relative_layer.finalize_output(relative_est) - 1

            # Finalize the estimates by re-scaling to the chosen semitone width
            output[utils.KEY_TABLATURE_REL] = self.semitone_radius * relative_est

        return output
