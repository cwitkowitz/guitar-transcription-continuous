# My imports
from amt_tools.models import TabCNN, LogisticBank

import amt_tools.tools as tools


class TabCNNMultipitch(TabCNN):
    """
    Implements TabCNN for multipitch estimation instead of tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and replace the final layer.
        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Determine the number of input neurons to the current softmax layer
        n_neurons = self.dense[-1].dim_in

        # Determine the number of distinct pitches
        n_multipitch = self.profile.get_range_len()

        # Create a layer for multipitch estimation and replace the tablature layer
        self.dense[-1] = LogisticBank(n_neurons, n_multipitch)

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
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Run the standard steps
        output = super().forward(feats)

        # Correct the label from tablature to multipitch
        output[tools.KEY_MULTIPITCH] = output.pop(tools.KEY_TABLATURE)

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

        # Obtain a pointer to the output layer
        multipitch_output_layer = self.dense[-1]

        # Obtain the multipitch estimation
        multipitch_est = output[tools.KEY_MULTIPITCH]

        # Keep track of loss
        total_loss = 0

        # Check to see if ground-truth multipitch is available
        if tools.KEY_MULTIPITCH in batch.keys():
            # Calculate the loss and add it to the total
            total_loss += multipitch_output_layer.get_loss(multipitch_est, batch[tools.KEY_MULTIPITCH])

        if total_loss:
            # Add the loss to the output dictionary
            output[tools.KEY_LOSS] = {tools.KEY_MULTIPITCH : total_loss,
                                      tools.KEY_LOSS_TOTAL : total_loss}

        # Finalize multipitch estimation
        output[tools.KEY_MULTIPITCH] = multipitch_output_layer.finalize_output(multipitch_est)

        return output


class TabCNNMultipitchRegression(TabCNNMultipitch):
    """
    Implements TabCNN for multipitch estimation instead of tablature estimation.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, device='cpu'):
        """
        Initialize the model and replace the final layer.
        Parameters
        ----------
        See TabCNN class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Create a new field for easy access
        self.multipitch_layer = self.dense[-1]

        # Create a layer of the same size for estimating relative pitch
        self.relative_layer = self.multipitch_layer.copy()

        print()

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
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Run the standard steps
        output = super().forward(feats)

        # TODO - estimate relative pitch here

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

        # Calculate tablature loss
        output = super().post_proc(batch)

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Unpack the loss if it exists
        loss = tools.unpack_dict(output, tools.KEY_LOSS)
        total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

        if loss is None:
            # Create a new dictionary to hold the loss
            loss = {}

        return output
