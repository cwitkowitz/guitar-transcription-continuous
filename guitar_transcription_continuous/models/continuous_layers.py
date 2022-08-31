# My imports
from amt_tools.models import OutputLayer, LogisticBank

# Regular imports
import torch


class ContinuousBank(LogisticBank):
    """
    Implements a multi-label continuous-valued [0, 1] output layer.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize fields of the output layer.

        Parameters
        ----------
        See LogisticBank class...
        """

        super().__init__(dim_in, dim_out, None)


class L2LogisticBank(ContinuousBank):
    """
    Implements a multi-label continuous-valued [0, 1] output layer with MSE loss.
    """

    def get_loss(self, estimated, reference):
        """
        Compute the mean-squared error of the estimated values relative to the ground-truth.

        Parameters
        ----------
        estimated : Tensor (B x T x O)
          estimated logits for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x O x T)
          ground-truth continuous values [0, 1] for a batch of tracks
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)

        Returns
        ----------
        mse : Tensor (1-D)
          Mean-squared error for entire batch
        """

        # Make clones so as not to modify originals out of function scope
        estimated = estimated.clone()
        reference = reference.clone()

        # Switch the frame and key dimension
        estimated = estimated.transpose(-2, -1)

        # Compute the mean squared error relative to the ground-truth
        mse = torch.abs(reference.float() - torch.sigmoid(estimated.float())) ** 2

        # Average mean squared error across frames
        mse = torch.mean(mse, dim=-1)
        # Sum mean squared error across keys
        mse = torch.sum(mse, dim=-1)
        # Average mean squared error across the batch
        mse = torch.mean(mse)

        return mse


class CBernoulliBank(ContinuousBank):
    """
    Implements a multi-label Continuous Bernoulli output layer.
    """

    def get_loss(self, estimated, reference):
        """
        Compute the negative log-likelihood of the ground-truth w.r.t.
        distributions parameterized by the estimated continuous values.

        Parameters
        ----------
        estimated : Tensor (B x T x O)
          estimated logits for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x O x T)
          ground-truth continuous values [0, 1] for a batch of tracks
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)

        Returns
        ----------
        nll : Tensor (1-D)
          Negative log-likelihood for entire batch
        """

        # Make clones so as not to modify originals out of function scope
        estimated = estimated.clone()
        reference = reference.clone()

        # Switch the frame and key dimension
        estimated = estimated.transpose(-2, -1)

        # Obtain Continuous Bernoulli distributions parameterized by the estimated logits
        distributions = torch.distributions.ContinuousBernoulli(logits=estimated.float())
        # Compute the negative log likelihood of the ground-truth w.r.t. the distributions
        nll = -distributions.log_prob(reference.float())

        # Average negative log likelihood across frames
        nll = torch.mean(nll, dim=-1)
        # Sum negative log likelihood across keys
        nll = torch.sum(nll, dim=-1)
        # Average negative log likelihood across the batch
        nll = torch.mean(nll)

        return nll

    def finalize_output(self, raw_output):
        """
        Convert output logits into actual continuous value predictions.

        Parameters
        ----------
        raw_output : Tensor (B x T x O)
          Raw logits used for calculating loss
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)

        Returns
        ----------
        final_output : Tensor (B x O x T)
          Continuous values serving as final predictions
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)
        """

        final_output = OutputLayer.finalize_output(self, raw_output)

        # Obtain mean of the distributions parameterized by the estimated logits
        final_output = torch.distributions.ContinuousBernoulli(logits=final_output).mean
        # Switch the frame and key dimension
        final_output = final_output.transpose(-2, -1)

        return final_output
