from pydantic import BaseModel

from typing import List, Optional, Dict


class structure(BaseModel):
    id: str = None  # the pdb ID of the structure of interest
    cif: str = None  # the cif file of the
    width: int = 800  # the width of the image made from the cif file
    height: int = 600  # the height of the image made from the cif file
    size: int = 500  # the size of the larger


class background(BaseModel):
    image_path: str  # the path of the background image
    size: int = None  # the size of the image


class style(BaseModel):
    image_path: str  # the path of the style image


class gatys_transfer_parameters(BaseModel):
    """
    hyperparameters for Gatys et al. 2016 style transfer method
    """

    style_weight: float = 10e-2  # weight for style
    content_weight: float = 1e4  # content weight
    epochs: int = 10  # number of "epochs" to train
    steps_per_epoch: int = 100  # number of steps per epoch
    total_variation_weight: int = 0  # variation weight to penalize difference from original image


class style_transfer(BaseModel):
    background_parameters: gatys_transfer_parameters
    content_parameters: gatys_transfer_parameters
