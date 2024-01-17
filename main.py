import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

folder = 'images'
filename = 'resolution.jpg'
img = cv2.imread(os.path.join(folder,filename))
change_contrast = np.arange(0,1,0.05)

images_path = 'images/res_contrast'


for k in change_contrast:
    img_contrast = img*k
    img_contrast = img_contrast.astype(np.uint8)
    idx = format(k,'.3f')
    filename_contrast = str(filename[:-4] + idx + filename[-4:])
    cv2.imwrite(os.path.join(images_path,filename_contrast),img_contrast)


file_list = os.listdir(images_path)


for k in file_list:
    filename = os.path.join(images_path,k)
    img = cv2.imread(filename)
    print(filename + ' has contrast of ' + str(np.mean(img)))

    # cv2.imshow(img)
    contrast_vec = np.append(contrast_vec,np.mean(img))


plt.plot(change_contrast,contrast_vec[:-1],"or")
plt.xlabel(' % of orignal contrast image ')
plt.ylabel(" Contrast Value")
plt.title("changing contrast of resoltion image")
plt.show()


# # matplotlib image display
# img2 = mpimg.imread(filename)
# plt.imshow(img2)
# plt.axis("off")
# plt.show()
#
# # PIL image display
# img3 = Image.open(filename)
# img3.show()
#
# print(img3.format)
# print(img3.mode)