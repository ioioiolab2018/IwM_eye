import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage
from skimage import measure
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.feature import blob_log
from skimage.filters import sobel, gaussian, rank, threshold_otsu, frangi, threshold_adaptive, threshold_local, \
    threshold_triangle
from skimage.transform import resize
from skimage.morphology import disk, opening, square, erosion, black_tophat, white_tophat, closing, binary_closing, \
    remove_small_objects, remove_small_holes

imageColor = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/Nowy folder/im0255.ppm')
# imageColor = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/images/08_h.JPG')
# manual = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/01_h.tif')
mask = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/Nowy folder/im0255.ah.ppm')

neigh = KNeighborsClassifier(n_neighbors=5)
imageColor = imageColor[:,:, 1]


y = []
X = []
for i in range(10, len(imageColor) - 10):
    for j in range(10, len(imageColor[0]) - 10):
        r = []
        for x in imageColor[i - 5:i + 5][j - 5:j + 5]:
            r.extend(x)
        X.append(r)
        y.append(mask[i][j])

neigh.fit(X, y)

image = data.load('C:/Users/Kamil/Documents/GitHub/IwM_eye/resources/Nowy folder/im0001.ppm')
res = np.copy(image)
res = res[:][:][1]
for i in range(10, len(image) - 10):
    for j in range(10, len(image) - 10):
        res[i][j] = neigh.predict(image[i - 5:i + 5][j - 5:j + 5])

plt.imshow(res, cmap="gray")
plt.show()

imageColor = np.array(imageColor)
x, y, z = np.shape(imageColor)

# green = imageColor[:, :, 1]
green = imageColor[:, :, 1] - np.min(imageColor, axis=2)

red = imageColor[:, :, 0]
red = rescale_intensity(red, in_range=(red.min(), red.max()))
# green = equalize_adapthist(green)

plt.imshow(green, cmap="gray")
plt.show()
# green = rescale_intensity(green, in_range=(green.min(), green.max()))


# lowpass = ndimage.gaussian_filter(green, 20)
# green = green + lowpass/2


plt.imshow(green, cmap="gray")
plt.show()

fran = frangi(green, scale_range=(0, 6), scale_step=1)

res1 = fran

plt.imshow(res1, cmap="gray")
plt.show()

thresh = threshold_triangle(res1)
res2 = res1 > thresh

res2 = remove_small_objects(res2, 30, 2)

plt.imshow(res2, cmap="gray")
plt.imsave("new", res2)
plt.show()
