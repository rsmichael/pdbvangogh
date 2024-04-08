import tensorflow as tf
import time
from pdbvangogh.img import tensor_to_image, load_img, imshow
from pdbvangogh import data_models

def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values.
    Acknowlegement: https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):
    """
    Model for vgg19-based style transfer
    Acknowledgement: https://www.tensorflow.org/tutorials/generative/style_transfer
    """

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[: self.num_style_layers], outputs[self.num_style_layers :])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {"content": content_dict, "style": style_dict}


def style_content_loss(outputs, style_targets, content_targets, num_content_layers, num_style_layers, style_weight, content_weight):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def am_i_in_a_jupyter_notebook():
    try:
        from IPython import get_ipython

        # Check if the get_ipython function exists
        # If we're in Jupyter, get_ipython will not be None
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        # IPython is not installed, definitely not in a notebook
        return False
    except AttributeError:
        # IPython is installed but we're not in a Jupyter environment
        return False
    return True


def style_transfer_vgg19(content_image, style_image, hyperparameters=data_models.vgg19_transfer_parameters()):
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    content_layers = ["block5_conv2"]
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]
    start = time.time()
    image = tf.Variable(content_image)

    @tf.function()
    def train_step(image, style_targets, content_targets):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets, num_content_layers, num_style_layers, hyperparameters.style_weight, hyperparameters.content_weight)
            if hyperparameters.total_variation_weight > 0:
                loss += hyperparameters.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    epochs = epochs
    steps_per_epoch = steps_per_epoch
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    step = 0
    in_jupyter_notebook = am_i_in_a_jupyter_notebook()
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, style_targets, content_targets)
            print(".", end="", flush=True)
        if in_jupyter_notebook:
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    return tensor_to_image(image)
