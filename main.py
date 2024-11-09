import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def z_normalize(image):
    """This function computes the channel-wise z-normalization.

    :param image: The image to be normalized with channels first, i.e., (C, H, W) or (C, W, H)
    :return: The z-normalized image
    """
    # z-score normalization
    means = np.mean(image, axis=(1, 2), keepdims=True)
    stds = np.std(image, axis=(1, 2), keepdims=True)

    image = (image - means) / stds

    return image

def load_train_data():
    """This function loads the training data.

    :return: Training image and training labels. Training image comes in (C, H, W). Training Labels comes in (H, W)
    """
    train_image = Image.open('./data/train/flowers.png')
    train_labels = Image.open('./data/train/flowers_labels.png')

    # (H, W, C)
    train_image = np.array(train_image)
    train_labels = np.array(train_labels)

    # (C, H, W)
    train_image = train_image.transpose(2, 0, 1)
    train_labels = train_labels.transpose(2, 0, 1)[0, :, :]

    return train_image, train_labels

def window_X_data(train_image, kernel_size: int = 3):
    """This function computes the (kernel_size * kernel_size) windows of an image.
    The return value is an array (N, C*kernel_size*kernel_size) of windows of an image.
    N is the number of valid windows to fit in the image, i.e., we do not use padding.
    C is the number of channels in the image.

    :param train_image: The train image to compute windows from.
    :param kernel_size: The kernel size to use for the classifier.
    :return: An array (N, C*kernel_size*kernel_size) of windows of an image.
    """
    channel_dim, height, width = train_image.shape

    windowed = np.lib.stride_tricks.sliding_window_view(
        train_image,
        (channel_dim, kernel_size, kernel_size)
    )

    windowed = windowed.reshape(
        (windowed.shape[0]*windowed.shape[1]*windowed.shape[2],
         windowed.shape[3]*windowed.shape[4]* windowed.shape[5])
    )
    return windowed


def preprocess_x(image, kernel_size: int = 3):
    """This function preprocess an image, i.e., it applies z-normalization and computes valid windows of the image.

    :param image: The image to be preprocessed.
    :param kernel_size: The kernel size to use for the classifier.
    :return: An array (N, C*kernel_size*kernel_size) of windows of an image.
    """
    image = z_normalize(image)
    image_windows = window_X_data(image, kernel_size)

    return image_windows


def preprocess_y(train_labels, kernel_size: int = 3):
    """This function preprocess the labels of training data.

    :param train_labels: The labels of the training data.
    :param kernel_size: The kernel size to use for the classifier.
    :return: An array (H', W') of labels of training data. The image returned here is smaller than the input image
    since we make sure, that it contains only labels for which we can fit a valid window into the image.
    """
    # transform to {0, 1} labels
    train_labels = train_labels / 255.

    padding = (kernel_size - 1) // 2

    height, width = train_labels.shape

    window_labels = train_labels[padding:(height - padding), padding:(width - padding)]

    window_labels = window_labels.flatten()

    return window_labels

def load_test_data():
    """
    This function loads the test data.

    :return: Test image (without labels) in format (C, H, W).
    """
    test_image = Image.open('./data/test/flowers_test.png')

    # (H, W, C)
    test_image = np.array(test_image)

    # (C, H, W)
    test_image = test_image.transpose(2, 0, 1)

    return test_image

def loss(windows, labels, theta):
    """This function computes the loss of the model.

    :param windows: The image windows of the training data.
    :param labels: The labels of the training data.
    :param theta: The current parameters of the classifier.
    :return:
    """
    labels_pred = y_theta(windows, theta)
    lossEval = -np.mean(labels * np.log(labels_pred) + (1 - labels) * np.log(1 - labels_pred))

    return lossEval

def f_theta(windows, theta):
    """This function computes the f(theta) of the model, also known as penultimate activation.

    :param windows: The image windows of the training data.
    :param theta: The current parameters of the classifier.
    :return: f(theta) of the model.
    """
    return np.dot(windows, theta)

def y_theta(windows, theta):
    """This function computes the y(theta) of the model, i.e., a sigmoid function, but with base 2.

    :param windows: The image windows of the training data.
    :param theta: The current parameters of the classifier.
    :return: A vector of size (N) with values between 0 and 1.
    """
    z = f_theta(windows, theta)
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.power(2, -z_clipped))

def gradient(windows, labels, theta):
    """
    This function computes the gradient of the loss function with respect to the current parameters.

    :param windows: The image windows of the training data.
    :param labels: The labels of the training data.
    :param theta: The current parameters of the classifier.
    :return: A vector of the same shape as theta.
    """

    labels_pred = y_theta(windows, theta)

    epsilon = 1e-15
    labels_pred = np.clip(labels_pred, epsilon, 1 - epsilon)

    gradient = np.mean(((labels_pred - labels) / (labels_pred * (1 - labels_pred)))[:, np.newaxis] * windows, axis=0)

    return gradient;

def train(X_windows, y_windows, theta, lr, steps, log_step):
    """
    This function trains the model.

    θ ← θ − λ∇θφ

    :param X_windows: The image windows of the training data.
    :param y_windows: The labels of the training data.
    :param theta: The initial parameters of the classifier.
    :param lr: The learning rate.
    :param steps: The number of training steps.
    :param log_step: Log interval
    :return: Updated parameters of the classifier.
    """
    
    for step in range(steps):
        gradient_ = gradient(X_windows, y_windows, theta)
        loss_ = loss(X_windows, y_windows, theta)
        theta = theta - lr * gradient_

        if step % log_step == 0:
            print(f"Step {step}: Current loss = {loss_} Current theta = {theta}")
    
    return theta

def clip_valid_prediction_range(image, kernel_size: int = 3):
    """Computes the prediction range of an image, i.e., the center pixels of the image for which we can fit in windows of
    size (C, kernel_size, kernel_size).

    :param image: The image to be clipped.
    :param kernel_size: The kernel size to use for the classifier.
    :return: The clipped image.
    """
    padding = (kernel_size - 1) // 2

    if len(image.shape) == 2:
        # label mask
        height, width = image.shape[-2:]
        return image[padding:(height - padding), padding:(width - padding)]
    else:
        # RGB mask
        _, height, width = image.shape[-2:]
        return image[:, padding:(height - padding), padding:(width - padding)]

def predict(image, theta, kernel_size: int = 3):
    """This function predicts the labels of an image.

    :param image: RGB Image (not z normalized)
    :param theta: parameters for logistic regression model
    :param kernel_size: kernel size used for filter
    :return:
    """
    _, height, width = image.shape
    padding = (kernel_size - 1) // 2

    image_windows = preprocess_x(image, kernel_size)
    y_predicted = y_theta(image_windows, theta)

    # reshape image shape
    return y_predicted.reshape(height - 2*padding, width - 2*padding)


def plot(image, theta, labels = None, kernel_size: int = 3):
    """
    This function plots either (Ground Truth Labels, Fractional Predictions, Rounded Predictions) if labels are provided
    or (RGB Image, Fractional Predictions, Rounded Predictions) if labels are not provided.

    :param image: The image we want to plot the predictions for.
    :param theta: The parameters for the model.
    :param labels: The labels of the training data (optional).
    :param kernel_size: The kernel size used for model.
    :return: None
    """
    _, height, width = image.shape

    y_pred = predict(image, theta, kernel_size)

    fig, axes = plt.subplots(1, 3)
    if labels is None:
        axes[0].set_title('RGB Image')
        axes[0].imshow(image.transpose(1, 2, 0))
    else:
        labels = clip_valid_prediction_range(labels, kernel_size)
        axes[0].set_title('Ground Truth Labels')
        axes[0].imshow(labels, cmap='gray')

    axes[1].set_title('Predicted (fractional)')
    axes[1].imshow(y_pred, cmap='gray')

    axes[2].set_title('Predicted (rounded)')
    axes[2].imshow(y_pred.round(), cmap='gray')

    plt.show()


def main():

    kernel_size = 5
    assert(kernel_size % 2 == 1)

    X_train, y_train = load_train_data()
    channels, height, width = X_train.shape
    X_windows = preprocess_x(X_train, kernel_size)
    y_windows = preprocess_y(y_train, kernel_size)

    theta = np.random.randn(kernel_size*kernel_size*channels)
    lr = 0.00003
    steps = 200
    log_step = 20

    theta = train(X_windows, y_windows, theta, lr, steps, log_step)

    plot(X_train, theta, labels=y_train, kernel_size=kernel_size)

    # Test Image
    x_test = load_test_data()
    plot(x_test, theta, kernel_size = kernel_size)

if __name__ == '__main__':
    main()
