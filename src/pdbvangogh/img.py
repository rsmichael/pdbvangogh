import tensorflow as tf
import numpy as np
import PIL


def tensor_to_image(tensor):
    """
    convert a tensorflow tensor representation of an image to a PIL.image object

    This function is adapted from: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim=512):
    """
    load an image and cap its size at max_dim

    This function is adapted from: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    """
    visualize a tensorflow or PIL.image object

    If the tensorflow object has 4 dimensions, reduce it to 3

    This function is adapted from: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def resize_pil(image, new_width):
    """
    resize a PIL.image object by adjusting its width
    """
    from PIL import Image

    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image


def overlay_images(background_path, foreground_path, position=(0, 0), background_size=None, foreground_size=None, save_path="overlayed_image.png"):
    from PIL import Image

    """
    Overlay a foreground image on top of a background image.

    Parameters:
    - background_path: Path to the background image.
    - foreground_path: Path to the foreground image.
    - position: A tuple (x, y) specifying the top-left corner of the foreground image.
    - save_path: Path where the resulting image will be saved.
    """
    # Open the background and foreground images
    background = Image.open(background_path).convert("RGBA")
    foreground = Image.open(foreground_path).convert("RGBA")

    # resize images if sizes are specified
    if background_size is not None:
        background = resize_pil(background, background_size)
    if foreground_size is not None:
        foreground = resize_pil(foreground, foreground_size)

    # Overlay the foreground image on top of the background image at the specified position
    background.paste(foreground, position, mask=foreground)

    # Save the resulting image
    background.save(save_path)

    print(f"Image saved as {save_path}")


def compute_sobel_mask(image):
    """
    compute the Sobel mask of a PIL.image object

    This function is adapted from: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    sobel_x = tf.image.sobel_edges(image)[..., 0]
    sobel_y = tf.image.sobel_edges(image)[..., 1]
    sobel_edges = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))

    # # Create a binary mask where edge values are above a threshold
    threshold = tf.reduce_mean(sobel_edges) * 0.3  # Example threshold
    mask = sobel_edges > threshold
    mask = tf.reduce_any(mask, axis=-1, keepdims=True)
    return mask


def apply_mask(mask, image):
    """
    Apply a sobel mask to a PIL.image object

    This function is adapted from: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    selected_parts = image * tf.cast(mask, tf.float32)
    return selected_parts


def black_to_transparent(image_path, output_path, threshold=0):
    from PIL import Image

    """
    Convert black pixels to transparent in an image.

    Parameters:
    - image_path: str, the path to the input image.
    - output_path: str, the path where the output image will be saved.
    - threshold: int, the maximum value for R, G, and B components to be considered black.
                 Default is 0 (pure black), increase it to include darker colors.
    """
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Convert to RGBA if not already in this mode

    # Load pixels
    datas = img.getdata()

    # Create a new data list
    newData = []
    for item in datas:
        # Change all pixels that are pure black to transparent
        if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold:
            newData.append((255, 255, 255, 0))  # Making the pixel fully transparent
        else:
            newData.append(item)

    # Update image data
    img.putdata(newData)
    img.save(output_path, "PNG")








def sobel_mask(reference_image, query_image, final_size, save_path="intermediate_image.png"):
    """
    Apply the sobel mask of a reference image in PIL.image format to a query image

    Resize the image to a designated final_size

    Save the image at save_path
    """
    # resize the reference image to match the size of the query image
    # query_image_size = (query_image.shape[1], query_image.shape[2])
    if type(query_image) == PIL.Image:
        query_image_size = query_image.size
    else:
        query_image_size = (query_image.shape[1], query_image.shape[2])
    reference_image = tf.image.resize(
        reference_image,
        query_image_size,
        # preserve_aspect_ratio=True,
        antialias=False,
        name=None,
    )
    sobel_mask = compute_sobel_mask(reference_image)
    masked_image = apply_mask(sobel_mask, query_image)
    masked_image_resized = tf.image.resize(masked_image, [final_size, final_size], preserve_aspect_ratio=True, antialias=False, name=None)
    tensor_to_image(masked_image_resized).save(save_path)
    black_to_transparent(image_path=save_path, output_path=save_path)
    masked_image_final = load_img(save_path)
    return masked_image


def tf_image_to_pil_image(tf_image):
    """
    Convert a TensorFlow image tensor to a PIL Image.

    Parameters:
    - tf_image: A TensorFlow tensor representing an image, with pixel values in either the 0-1 or 0-255 range.

    Returns:
    - A PIL Image object corresponding to the input TensorFlow tensor.
    """
    # Check if the TensorFlow tensor is in the 0-1 range
    if tf.reduce_max(tf_image) <= 1.0:
        # Scale the tensor values to the 0-255 range
        tf_image = tf_image * 255

    # Convert the TensorFlow tensor to a NumPy array
    np_image = tf_image.numpy().astype(np.uint8)
    if np_image.ndim == 4 and np_image.shape[0] == 1:
        np_image = np_image.squeeze(0)
    # Convert the NumPy array to a PIL Image
    pil_image = PIL.Image.fromarray(np_image)

    return pil_image
