import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import median_filter

from PIL import Image

def rgb2gl(img):

    img_gl = np.zeros([img.shape[0],img.shape[1]])
    dim = img.shape[-1]
    for k in range(dim):
        img_gl += img[:,:,k]
    img_gl /= 3
    img_gl = img_gl.astype(np.uint8)
    return img_gl


file1 = 'images/how sem works.png'
file2 = 'images/Misc_pollen.jpg'
images_path = 'images/DIP3E_Original_Images_CH01'
filename = file1

# CV2 image display
img = cv2.imread(filename)
cv2.imshow('image', img)
cv2.waitKey(0)

img_gl = rgb2gl(img)
cv2.imshow('image', img_gl)
cv2.waitKey(0)

file_list = os.listdir(images_path)

contrast_vec = np.arange(0)
contrast_vec_median = np.arange(0)

## save different contrast of a given image
# folder = 'images'
# filename = 'resolution.jpg'
# img = cv2.imread(os.path.join(folder,filename))
# change_contrast = np.arange(0,1,0.05)
# images_path = 'images/res_contrast'
# for k in change_contrast:
#     img_contrast = img*k
#     img_contrast = img_contrast.astype(np.uint8)
#     idx = format(k,'.3f')
#     filename_contrast = str(filename[:-4] + idx + filename[-4:])
#     cv2.imwrite(os.path.join(images_path,filename_contrast),img_contrast)
median_size = 3
images_path = 'images/AA/x'
file_list = sorted(os.listdir(images_path))

change_contrast = [-5,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
for k in file_list:
    filename = os.path.join(images_path,k)
    img = cv2.imread(filename)
    img_median = median_filter(img, size=median_size)
    score_img = np.std(img)
    score_img_median = np.std(img_median)
    contrast_vec = np.append(contrast_vec,score_img)
    contrast_vec_median = np.append(contrast_vec_median,score_img_median)

    print(filename + ' original std = ' + str(score_img) +', median(' +str(median_size) + ')=' +str(score_img_median) )



f,axarr = plt.subplots(2,2)
# plt.plot(change_contrast,contrast_vec,"or")


axarr[0][0].plot(change_contrast,contrast_vec,"-or")
axarr[0][0].set_title("SEM measurements post-processing - original")
axarr[0][0].set_xlabel(' % of orignal contrast image ')
axarr[0][0].set_ylabel(" Contrast Value")
axarr[0][0].grid()

axarr[1][0].plot(change_contrast,contrast_vec_median,"-or")
axarr[1][0].set_xlabel(' % of orignal contrast image ')
axarr[1][0].set_ylabel(" std Value - median(20)")
axarr[1][0].set_title("SEM measurements post-processing - median(20)")
axarr[1][0].grid()


plt.show()


# matplotlib image display
# img2 = mpimg.imread(filename)
# plt.imshow(img2)
# # plt.axis("off")
# plt.show()
# #
# # PIL image display
# img3 = Image.open(filename)
# img3.show()
#
# print(img3.format)
# print(img3.mode)