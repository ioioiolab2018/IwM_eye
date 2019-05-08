from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.filters import frangi, threshold_triangle
from skimage.morphology import remove_small_objects


def statistics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    return accuracy, sensitivity, specificity


def compare_images(img, model):
    model[model > 0] = 1
    tp = sum(img[img == model])
    tn = -sum(img[img == model] - 1)
    fp = sum(img[img == 1]) - tp
    fn = sum(img[img == 0] + 1) - tn
    return tp, fp, fn, tn


def load_image(name):
    img = io.imread('resources/images/' + name + '.jpg')
    mask = io.imread('resources/masks/' + name + '_mask.tif')
    model = io.imread('resources/models/' + name + '.tif')
    return img, mask, model


def apply_mask(img, mask):
    image = np.array(img)
    mask = np.asarray(mask)
    mask = mask[:, :, 1]
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    red[mask == 0] = 0
    green[mask == 0] = 0
    blue[mask == 0] = 0
    return red, green, blue


def process_image(red, green, blue, mask):
    mask = np.asarray(mask)
    mask = mask[:, :, 1]
    red = rescale_intensity(red, in_range=(red.min(), red.max()))
    green = equalize_adapthist(green)
    plt.imshow(green, cmap="gray")
    plt.show()

    lowpass = ndimage.gaussian_filter(green, 20)
    green = green - lowpass
    green = rescale_intensity(green, in_range=(green.min(), green.max()))
    plt.imshow(green, cmap="gray")
    plt.show()

    fran = frangi(green, scale_range=(0, 6), scale_step=1)
    res1 = np.asarray(fran)
    # res1 = fran
    plt.imshow(res1, cmap="gray")
    plt.show()

    thresh = threshold_triangle(res1)
    res2 = res1 >= thresh
    res2 = remove_small_objects(res2, 30, 20)
    res2[mask == 0] = 0
    result = np.zeros_like(res2)
    result[res2] = 1
    # result[mask == 0] = 0

    plt.imshow(res2, cmap="gray")
    plt.imsave("new", res2, cmap="gray")
    plt.show()
    return result


# imageColor = data.load('C:/Users/Piotr/Documents/GitHub/IwM_eye/resources/images/05_h.jpg')
# imageColor = data.load('C:/Users/pawel/Desktop/Projekty/IwM/IwM_eye/resources/images/05_h.jpg')
# imageColor = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/images/08_h.JPG')
# manual = data.load('C:/Users\Piotr/Documents/GitHub/IwM_eye/resources/05_h.tif')
# model = data.load('C:/Users/pawel/Desktop/Projekty/IwM/IwM_eye/resources/05_h.tif')
# mask = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/Nowy folder/im0255.ah.ppm')

# green = imageColor[:, :, 1]
# green = imageColor[:, :, 1] - np.min(imageColor, axis=2)

# red = imageColor[:, :, 0]
# imageColor = np.array(imageColor)
# x, y, z = np.shape(imageColor)


imageColor, mask, model = load_image('05_h')
red, green, blue = apply_mask(imageColor, mask)
result = process_image(red, green, blue, mask)

tp, fp, fn, tn = compare_images(result, model)
print("Tp: " + str(tp), "\nFp: " + str(fp), "\nFn: " + str(fn), "\nTn: " + str(tn))
accuracy, sensitivity, specificity = statistics(tp, fp, fn, tn)
print("Accuracy: " + str(accuracy), "\nSensitivity: " + str(sensitivity), "\nSpecificity: " + str(specificity))
