import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli


def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters

    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):
    """
    Resize a given image according the the new dimension

    """
    return scipy.misc.imresize(image, new_dim)


def random_flip(image, steer_angle, flipping_prob=0.5):
    """
    The image will be flipped to negate the steering angle.
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steer_angle
    else:
        return image, steer_angle


def random_shear(image, steer_angle, shear_range=200):
    """
    applying random shear on the source image

    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steer_angle += dsteering

    return image, steer_angle


def random_rotation(image, steer_angle, rotation_amount=15):
    """

    """
    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steer_angle + (-1) * rad


def data_aug(image, steer_angle):
    """
    Implements various data augmentation techniques including:
    random sheer
    random flip
    crop
    resize
    """
    top_crop = 0.35
    bottom_crop = 0.1
    resize_dim = (64, 64)
    do_shear_prob = 0.9

    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steer_angle = random_shear(image, steer_angle)

    image = crop(image, top_crop, bottom_crop)

    image, steer_angle = random_flip(image, steer_angle)

    image = resize(image, resize_dim)

    return image, steer_angle


def generate_batch(batch_size=64):
    """
    The generator yields the next training batch depending upon the batch size.
    The data is augmented.

    """
    while True:
        X_batch = []
        y_batch = []

        data = pd.read_csv('./data/driving_log.csv')
        num_of_img = len(data)
        rnd_indices = np.random.randint(0, num_of_img, batch_size)

        images = []
        for i in rnd_indices:
            # Selecting random Images from the Left,Right or Center camera
            rnd_image = np.random.randint(0, 3)
            if rnd_image == 0:
                img = data.iloc[i]['left'].strip()
                angle = data.iloc[i]['steering'] + 0.23
                images.append((img, angle))

            elif rnd_image == 1:
                img = data.iloc[i]['center'].strip()
                angle = data.iloc[i]['steering']
                images.append((img, angle))
            else:
                img = data.iloc[i]['right'].strip()
                angle = data.iloc[i]['steering'] - 0.23
                images.append((img, angle))

        for img_file, angle in images:
            raw_image = plt.imread('./data/' + img_file)
            raw_angle = angle
            new_image, new_angle = data_aug(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)

