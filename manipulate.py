import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
# from main import rgb2gl
import matplotlib.image as mpimg

def rgb2gl(img):

    img_gl = np.zeros([img.shape[0],img.shape[1]])
    dim = img.shape[-1]
    for k in range(dim):
        img_gl += img[:,:,k]
    img_gl /= 3
    img_gl = img_gl.astype(np.uint8)
    return img_gl

images_path = 'images/AA/x'
k = 4
file_list = sorted(os.listdir(images_path))
filename = os.path.join(images_path, file_list[k])

img = cv2.imread(filename)
# img = mpimg.imread(filename)
# img = rgb2gl(img)
img_median = median_filter(img,size=20)

print('org std = ' + str(np.std(img)) + ', median std = ' + str(np.std(img_median)))
# img2 = mpimg.imread(filename)
f, axarr = plt.subplots(2,2)

axarr[0,0].imshow(img)
axarr[0,1].imshow(img_median)
# axarr[1,0].hist(img)
# plt.hist(img)
# axarr[1,1].imshow(image_datas[3])

# plt.imshow(img)
# plt.axis("off")
plt.show()