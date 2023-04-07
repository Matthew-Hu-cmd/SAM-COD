import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import build_sam, SamPredictor 

class BBox_gen():
    


predictor = SamPredictor(build_sam(checkpoint="</path/to/model.pth>"))
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)