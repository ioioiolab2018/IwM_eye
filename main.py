import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.filters import sobel, gaussian, rank, threshold_otsu, frangi
from skimage.transform import resize
from skimage.morphology import disk, opening, square, erosion

images_patch = "resources/images/"
fig, ax = plt.subplots()
# for open ppm files
# im = Image.open(images_patch+"im0001.ppm")
# im.show()


imageColor = data.load("C:/Users/Piotr/Documents/GitHub/IwM_eye/resources/images/01_dr.JPG")
imageColor = np.array(imageColor)
x, y, z = np.shape(imageColor)
imageColor = resize(imageColor, (int(x / 1.3), int(y / 1.3)))

imageColor = gaussian(imageColor, sigma=5, multichannel=True)
imageColor[:, :, 2] = 0
imageColor[:, :, 0] = 0

green = imageColor[:, :, 1]
# green = rank.median(green, disk(50))

p1, p99 = np.percentile(green, (1, 99))
green = rescale_intensity(green, in_range=(p1, p99))

green = frangi(green)

p1, p99 = np.percentile(green, (1, 99))
green = rescale_intensity(green, in_range=(p1, p99))

# thresh = threshold_otsu(green)
#
# green = green > thresh
green = np.asarray(green)
p1=0.07
green[green > p1] = 1
green[ green <= p1] = 0

ax.imshow(green, cmap="gray")
plt.show()
denoised = green
