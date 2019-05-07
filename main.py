import scipy

import matplotlib.pyplot as plt
import numpy as np
from imageio import imsave
from scipy import ndimage

from skimage import data, img_as_ubyte
from skimage import measure
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.feature import blob_log
from skimage.filters import sobel, gaussian, rank, threshold_otsu, frangi, threshold_local, \
    threshold_triangle
from skimage.transform import resize
from skimage.morphology import disk, opening, square, erosion, black_tophat, white_tophat, closing, binary_closing, \
    remove_small_objects, remove_small_holes
from scipy import signal

imageColor = data.load('C:/Users/Piotr/Documents/GitHub/IwM_eye/resources/images/05_h.jpg')
# imageColor = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/images/08_h.JPG')
manual = data.load('C:/Users\Piotr/Documents/GitHub/IwM_eye/resources/05_h.tif')
# mask = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/Nowy folder/im0255.ah.ppm')


imageColor = np.array(imageColor)
x, y, z = np.shape(imageColor)

green = imageColor[:, :, 1]

# green = imageColor[:, :, 1] - np.min(imageColor, axis=2)

red = imageColor[:, :, 0]
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

res1 = fran

plt.imshow(res1, cmap="gray")
plt.show()

thresh = threshold_triangle(res1)
res2= res1 >= thresh


res2 = remove_small_objects(res2, 30, 20)

TP = 0
TN = 0
FP = 0
FN = 0

result = np.zeros_like(res2)
result[res2]=1
print(manual)
for i in range(len(manual)):
    for j in range(len(manual[0])):
        if result[i][j] == 1:
            if manual[i][j] == result[i][j]:
                TP += 1
            else:
                FP += 1
        if result[i][j] == 0:
            if manual[i][j] == result[i][j]:
                TN += 1
            else:
                FN += 1

print(TP, TN, FP, FN)
plt.imshow(res2, cmap="gray")
plt.imsave("new", res2, cmap="gray")
plt.show()
