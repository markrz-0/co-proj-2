import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct

DATA_DIR = "data"
NUM_CASES = 100

