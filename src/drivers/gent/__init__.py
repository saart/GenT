import os

import numpy as np
import torch

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

os.environ["WANDB_MODE"] = "disabled"
