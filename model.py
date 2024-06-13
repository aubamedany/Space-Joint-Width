from gd import GDModel
from preprocess import *


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        