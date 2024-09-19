from pdbvangogh import pdb
from pdbvangogh.img import *
from pdbvangogh.models import *
from pdbvangogh.data_models import *
import os


def test_func():
    return True


def pdbvangogh(
    background_image,
    style_image,
    save_prefix,
    out_dir=None,
    content_image=None,
    content_cif=None,
    pdb_id=None,
    background_size=500,
    content_size=800,
    content_rotation = 0,
    background_hyperparameters=gatys_transfer_parameters(style_weight=1e-2, epochs=10, steps_per_epoch=100),
    content_hyperparameters=gatys_transfer_parameters(style_weight=1e-4, epochs=10, steps_per_epoch=100),
    transfer_method="gatys",
    debug=False,
):
    """
    Apply style transfer to macromolecule overlaid on background image

    Parameters:
    - background_image: A png of the background image
    - style_image: A png of the style image
    - save_prefix: prefix for output files
    - out_dir: directory in which to save output files
    - pdb_id: 4 character PDB ID of macromolecular structure of interest
    - content_image: visualization of macromolecule ONLY with no content outside of macromolecule. This is an ALTERNATIVE to pdb_id
    - background_size: maximum dimension of background image
    - content_size: maximum dimension of macromolecule image
    - background_hyperparameters: hyperparameter object for style transfer on background image. Must be appropriate for transfer method, e.g. gatys_transfer_parameters for gatys method
    - content_hyperparameters: hypterparameter object for style transfer on macromolecule image. Must be appropriate for transfer method, e.g. gatys_transfer_parameters for gatys method
    - transfer_method: style transfer method to be used. Currently, only acceptable input is "gatys"
    - debug: boolean for whether to print debugging messages

    Outputs:
    - A png image of a macromolecule overlaid on the chosen background with style applied
    - pngs of some intermediate files
    """
    from copy import deepcopy
    from PIL import Image
    assert pdb_id is not None or content_image is not None or content_cif is not None
    print(content_cif)
    if pdb_id is not None:
        assert out_dir is not None
        pdb.download_cif(pdb_id, out_dir)
        content_image = os.path.join(out_dir, f"{pdb_id}.png")
        pdb.visualize_cif_and_save_image(cif_file_path=os.path.join(out_dir, f"{pdb_id}.cif"), 
                                         image_output_path=content_image
                                         )
    if content_cif is not None:
        content_image = os.path.join(out_dir, f"{deepcopy(content_cif).split('.cif')[0]}.png")
        pdb.visualize_cif_and_save_image(cif_file_path=content_cif, 
                                         image_output_path=content_image 
                                         )
    background_image = load_img(background_image)
    background_image_resized = tf.image.resize(background_image, [background_size, background_size], preserve_aspect_ratio=True, antialias=False, name=None)
    content_image = load_img(content_image)
    # Apply content rotation (clockwise by the indicated number of degrees, or closest multiple of 90 degrees)
    content_image = tf.image.rot90(content_image, k=content_rotation//90)
    content_image_resized = tf.image.resize(content_image, [content_size, content_size], preserve_aspect_ratio=True, antialias=False, name=None)
    style_image = load_img(style_image)
    # apply selected style transfer method. Only Gatys et al. 2016 method currently implemented
    assert transfer_method in ["gatys"]  # add other transfer methods here when implemented
    if transfer_method == "gatys":
        # implement style transfer using Gatys et al. 2016 method.
        # style molecule only
        styled_content_image = style_transfer_gatys(content_image=content_image_resized, style_image=style_image, hyperparameters=content_hyperparameters)
        styled_content_image.save(f"{save_prefix}_unmasked_styled_content.png")
        # style background image
        styled_background_image = style_transfer_gatys(content_image=background_image_resized, style_image=style_image, hyperparameters=background_hyperparameters)
        masked_styled_content = sobel_mask(
            reference_image=content_image, query_image=load_img(f"{save_prefix}_unmasked_styled_content.png"), final_size=content_size, save_path=f"{save_prefix}_masked_styled_content.png"
        )
        styled_background_image.save(f"{save_prefix}_background_styled_content.png")
        if debug:
            print(background_image_resized.shape, type(background_image_resized), type(styled_background_image), content_image_resized.shape, type(content_image_resized), type(styled_content_image))
    # overlay styled molecule and background images to produce final image
    overlay_images(background_path=f"{save_prefix}_background_styled_content.png", foreground_path=f"{save_prefix}_masked_styled_content.png", save_path=f"{save_prefix}_overlayed_image.png")
