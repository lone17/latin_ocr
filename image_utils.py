import cv2
import json
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data_dir = 'real/'
max_h_real = 588
max_w_real = 3357
img_h = 100
img_w = 3880
# with open(data_dir + 'label.json', 'r', encoding='utf8') as f:
#     content = json.load(f)
#
# with open(data_dir + 'src-all.txt', 'r') as f:
#     files = f.read().split('\n')[:-1]

# images = []
# for file in files:
#     image = cv2.imread(data_dir + file, 0)
#     images.append(image)

# with open('real_images.pickle', 'rb') as f:
#     images = pickle.load(f)

# heights = [i.shape[0] for i in images]
# widths = [i.shape[1] for i in images]

# plt.subplot(1, 2, 1)
# p = sns.distplot(heights)
# plt.subplot(1, 2, 2)
# p = sns.distplot(widths)
# plt.show()

def implt(img, cmp=None, t=''):
    """ Show image using plt """
    plt.imshow(img, cmap=cmp)
    plt.title(t)

def resize(img, height, always=True):
    """ Resize image to given height """
    if (img.shape[0] > height or always):
        rate = height / img.shape[0]
        return cv2.resize(img, (int(rate * img.shape[1]), height))
    return img

def pad(img, width, vertical=False):
    if vertical:
        img = cv2.copyMakeBorder(img, 0, width - img.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=0)
        img = img[:,:,None]
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, width - img.shape[1],
                                  cv2.BORDER_CONSTANT, value=0)

    return img

def preprocess(image, padding=True, width=img_w, plot=False):
    image = resize(image, 100, always=True)
    if plot:
        plt.subplot(4, 1, 1)
        implt(image, t='original')
    image2 = cv2.fastNlMeansDenoising(image, None, templateWindowSize=7, searchWindowSize=21, h=17)
    # image2 = image
    if plot:
        plt.subplot(4, 1, 2)
        implt(image2, t='denoised')
    ret, image3 = cv2.threshold(image2, 220, 255, cv2.THRESH_BINARY_INV)
    # image3 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 7)
    if plot:
        plt.subplot(4, 1, 3)
        implt(image3, t='threshed')
    image4 = cv2.morphologyEx(image3, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    if plot:
        plt.subplot(4, 1, 4)
        implt(image4, t='processed')
        plt.show()

    result = image4
    if padding:
            result = pad(image4, width)

    return result.swapaxes(0,1)[:,:,None]

# for i in np.random.randint(0, len(images), 10):
#     print(files[i])
#     preprocess(images[i], True)

# processed_images = []
# for img in images:
#     img = preprocess(img)
#     processed_images.append(img)

def make_X_from_images(images, file_list, out):
    with open(images, 'rb') as f:
        images = pickle.load(f)
    with open(file_list, 'r') as f:
        files = f.read().split('\n')

    X = np.zeros((len(files), img_w, img_h, 1), dtype=np.uint8)
    for i in range(len(files)):
        print(files[i])
        X[i] = preprocess(images[i])
        np.save(out, X)
