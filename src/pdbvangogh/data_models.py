from pydantic import BaseModel

from typing import List, Optional, Dict

class structure(BaseModel):
    id : str = None # the pdb ID of the structure of interest
    cif : str = None # the cif file of the 
    width : int = 800 # the width of the image made from the cif file
    height : int = 600 # the height of the image made from the cif file
    size : int = 500 # the size of the larger 


class background(BaseModel):
    image_path : str # the path of the background image
    size : int = None # the size of the image


class style(BaseModel):
    image_path : str # the path of the style image


class vgg19_transfer(BaseModel):
    style_weight : float = 10e-2 # weight for style
    

class style_transfer(BaseModel):
    background_parameters = {} : Dict
    content_parameters = {} : Dict
