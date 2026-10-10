"""Distribution utilities."""

import math

import torch
import torch.nn.functional as F


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-det-Jacobian of the tanh transform.

    See https://github.com/google-deepmind/distrax/issues/216.
    """
    # log(1 - tanh^2(x)) = log(4) - 2*log(exp(x) + exp(-x))
    #                    = 2*(log(2) - x - softplus(-2x))
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
