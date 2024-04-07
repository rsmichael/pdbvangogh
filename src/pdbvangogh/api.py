from pdbvangogh.pdb import *
from pdbvangogh.img import *
from pdbvangogh.models import *
from pdbvangogh.data_models import *

def test_func():
    return True

def pdbvangogh(background_image, content_image, style_image, save_prefix, background_style_weight=1e-2, content_style_weight=1e-4, content_size=50, background_size=100, epochs=10, steps_per_epoch=100, debug=False):
    background_image = load_img(background_image)
    background_image_resized = tf.image.resize(background_image, [background_size, background_size], preserve_aspect_ratio=True, antialias=False, name=None)
    content_image = load_img(content_image)
    content_image_resized = tf.image.resize(content_image, [content_size, content_size], preserve_aspect_ratio=True, antialias=False, name=None)
    style_image = load_img(style_image)
    styled_content_image = style_transfer_vgg19(content_image=content_image_resized, style_image=style_image, style_weight=content_style_weight, epochs=epochs, steps_per_epoch=steps_per_epoch)
    styled_content_image.save(f"{save_prefix}_unmasked_styled_content.png")
    styled_background_image = style_transfer_vgg19(content_image=background_image_resized, style_image=style_image, style_weight=background_style_weight, epochs=epochs, steps_per_epoch=steps_per_epoch)
    masked_styled_content = sobel_mask(reference_image = content_image,
                                     query_image = load_img(f'{save_prefix}_unmasked_styled_content.png'),
                                     final_size = content_size,
                                   save_path = f'{save_prefix}_masked_styled_content.png')
    styled_background_image.save(f"{save_prefix}_background_styled_content.png")
    if debug:
        print(background_image_resized.shape, type(background_image_resized), type(styled_background_image), content_image_resized.shape, type(content_image_resized), type(styled_content_image))
    overlay_images(background_path=f"{save_prefix}_background_styled_content.png", foreground_path=f"{save_prefix}_masked_styled_content.png", save_path=f"{save_prefix}_overlayed_image.png")
