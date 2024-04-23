# Testing

1. Read an image-
•	Code- 

import numpy as np
import cv2
print("OpenCV-Python Version {}".format(cv2.__version__))
from cv2 import imread
img = imread("/content/img1.png")
print('Datatype:',img.dtype)
print('\nDimensions', img.shape)
print(img[0,0])

2. Display an image 
•	Code-

import matplotlib.pyplot as plt
from cv2 import imread
img = imread("/content/img1.png")
plt.imshow(img)
plt.title('Displaying image using Matplotlib')
print(img[0,0])
plt.show()


3. Color conversion:
A.	BGR TO RGB 

•	Code-

import cv2 as cv
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
img = cv.imread("/content/img1.png")
from cv2 import cvtColor, COLOR_BGR2RGB
img_rgb = cvtColor(img,COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
print(img_rgb[0,0])


B.	Conversion image to greyscale 
 
•	Code-

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/content/img1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.show()

C.	BGR TO HSV 
 
•	Code-

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/img1.png')
from cv2 import cvtColor, COLOR_BGR2HSV
img_hsv = cvtColor(img,COLOR_BGR2HSV)
plt.imshow(img_hsv)
plt.show()
print(img_hsv[0,0])

1. Splitting image in three channels:
A.	Splitting RGB image into 3 channels
•	Code-
import cv2
import matplotlib.pyplot as plt
img_rgb = cv2.imread('/content/img1.png')
r,g,b = cv2.split(img_rgb)
plt.figure(figsize=(10,10)) #this is used to fix the image size
plt.subplot(221)
plt.title('Original')
plt.imshow(img_rgb)
plt.subplot(222)
 plt.title('Blue Channel')
plt.imshow(b)
plt.subplot(223)
plt.title('Green Channel')
plt.imshow(g)
plt.subplot(224)
plt.title('Red Channel')
plt.imshow(r)

B.	SPLITTING HSV IMAGE INTO 3 CHANNELS
•	Code-
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/content/img1.png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
plt.figure(figsize=(10,10)) #this is used to fix the image size
plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Hue Channel')
plt.imshow(h)
plt.subplot(223)
plt.title('Saturation Channel')
plt.imshow(s)
plt.subplot(224)
plt.title('Value Channel')
plt.imshow(v)

C.	SPLITTING LAB IMAGE INTO 3 CHANNELS
•	Code-
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/content/img1.png')
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(img_lab)
plt.figure(figsize=(10,10)) #this is used to fix the image size
plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Lightness Channel')
plt.imshow(l)
plt.subplot(223)
plt.title('Green-Red Channel')
plt.imshow(a)
plt.subplot(224)
plt.title('Blue-Yellow Channel')
plt.imshow(b)


1. Image Transformation
A.	Translation
•	Code-
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
img = cv.imread('/content/img1.png',0)
rows , cols = img.shape
M = np.float32([[1,0,100],[0,1,100]])
dst = cv.warpAffine (img, M , (cols,rows))
cv2_imshow(dst)
cv2.waitKey(0)
cv2.destroyAllWindows

We can see that the image is cropped, to get the full image, we can use the following code-

•	Code-
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
img = cv.imread('/content/img1.png',0)
rows , cols = img.shape
M = np.float32([[1,0,100],[0,1,100]])
dst = cv.warpAffine (img, M , (cols+100,rows+100)) 
cv2_imshow(dst)
cv2.waitKey(0)
cv2.destroyAllWindows

B.	Rotation along X,Y axis

•	Code-
import cv2
from google.colab import files
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()

file_name = next(iter(uploaded))
img = cv2.imread(file_name)

rotated_x = cv2.flip(img, 0)

rotated_y = cv2.flip(img, 1)

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title('Rotated along X axis')
plt.imshow(cv2.cvtColor(rotated_x, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.title('Rotated along Y axis')
plt.imshow(cv2.cvtColor(rotated_y, cv2.COLOR_BGR2RGB))
plt.show()

C.	Rotation at a particular angle

•	Code-
import cv2
from google.colab import files
import numpy as np
from matplotlib import pyplot as plt

uploaded = files.upload()
file_name = next(iter(uploaded))
img = cv2.imread(file_name)
angle = 45

height, width = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Rotated Image')
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.show()

D.	Cropping

•	Code-
import cv2
from google.colab import files
import numpy as np
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = next(iter(uploaded))
img = cv2.imread(file_name)
x, y, w, h = 800, 500, 500, 400 
cropped_img = img[y:y+h, x:x+w]
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Cropped Image')
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.show()

1. Image Transformation
A.	Scaling by shrinking and enlarging an image
•	Code-
import cv2
from google.colab import files
from google.colab.patches import cv2_imshow
uploaded = files.upload()
file_name = next(iter(uploaded))
img = cv2.imread(file_name)
shrink_scale_percent = 10
enlarge_scale_percent = 120
shrink_width = int(img.shape[1] * shrink_scale_percent / 100)
shrink_height = int(img.shape[0] * shrink_scale_percent / 100)
enlarge_width = int(img.shape[1] * enlarge_scale_percent / 100)
enlarge_height = int(img.shape[0] * enlarge_scale_percent / 100)
shrinked_img = cv2.resize(img, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)
enlarged_img = cv2.resize(img, (enlarge_width, enlarge_height), interpolation=cv2.INTER_CUBIC)
cv2_imshow(shrinked_img)
cv2_imshow(enlarged_img)


B.	Shearing along X and Y Axis
•	Code-
import cv2
from google.colab import files
import numpy as np
from matplotlib import pyplot as plt
uploaded = files.upload()
file_name = next(iter(uploaded))
img = cv2.imread(file_name)
shear_factor_x = 0.2
shear_factor_y = 0.1
height, width = img.shape[:2]
M_shear_x = np.float32([[1, shear_factor_x, 0],
                        [0, 1, 0]])
M_shear_y = np.float32([[1, 0, 0],
                        [shear_factor_y, 1, 0]])

sheared_img_x = cv2.warpAffine(img, M_shear_x, (width, height))
sheared_img_y = cv2.warpAffine(img, M_shear_y, (width, height))
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1, 3, 2)
plt.title('Sheared along X axis')
plt.imshow(cv2.cvtColor(sheared_img_x, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.title('Sheared along Y axis')
plt.imshow(cv2.cvtColor(sheared_img_y, cv2.COLOR_BGR2RGB))\
plt.show()

1. Linear Image Filtering
•	Code-
import numpy as np
import cv2
import matplotlib.pyplot as plt
def point_operation(img,k,l):
    img = np.asarray(img,dtype = float)
    img = img*k + l
    img[img>255] = 255
    img[img<0] = 0
    return np.asarray(img,dtype=int)
def main():
    img = cv2.imread('download.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out1 = point_operation(gray,0.5,0)
    out2 = point_operation(gray,1,10)
    out3 = point_operation(gray,0.7,25)
    res = np.hstack([gray,out1,out2,out3])
    plt.imshow(res, cmap = "gray")
    plt.axis("off")
 
main()

2. 2D Linear Image Filtering
C.	Using Custom Kernel
•	Code-
import numpy as np
import cv2
from matplotlib import pyplot as plt
image = cv2.imread("download.jpg")
kernel1 = np.ones((5,5),np.float64)/25
img = cv2.filter2D(src = image, ddepth=-1,kernel=kernel1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image')
plt.axis('off')

plt.show()

D.	Using cv2.blur
•	Code-
import numpy as np
import cv2
def plot_cv_img(input_image,output_image):
    fig,ax = plt.subplots(nrows=1,ncols=2)
    ax[0].imshow(cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB))
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB))
    ax[1].set_title('Box Filter 5x5')
    ax[1].axis('off')
    plt.show()
def main():
    img =cv2.imread("download.jpg")
    Kernel_size = (5,5)
    blur = cv2.blur(img,Kernel_size)
    plot_cv_img(img,blur)
main()

1. Log Transformation
•	Code-
import cv2
import numpy as np
image = cv2.imread('download.jpg', cv2.IMREAD_GRAYSCALE)
c = 255 / np.log(1 + np.max(image))
log_transformed = c * np.log(1 + image)
log_transformed = np.uint8(log_transformed)
cv2.imshow('Original Image', image)
cv2.imshow('Log Transformed Image', log_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()

2. Inverse Log Transformation
•	Code-
import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow
uploaded = files.upload()
file_name = next(iter(uploaded))
image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_GRAYSCALE)
print("Image shape:", image.shape)
max_intensity = np.max(image)
c = 255 / np.log(1 + max_intensity)
inverse_log_transformed = np.exp(image / c) - 1
inverse_log_transformed = np.uint8(inverse_log_transformed)
cv2_imshow(image)
cv2_imshow(inverse_log_transformed)

3. Power law Transformation
•	Code-
import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow
def power_law_transform(image, gamma):
    image_normalized = image.astype('float32') / 255.0
    transformed_image = np.power(image_normalized, gamma)
    transformed_image = np.uint8(transformed_image * 255)
    return transformed_image
uploaded = files.upload()
file_name = next(iter(uploaded))
image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_GRAYSCALE)
gamma = 0.5
transformed_image = power_law_transform(image, gamma)
cv2_imshow(image)
cv2_imshow(transformed_image)

1. Robert Operator
•	Code-

import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)
roberts_x = np.array([[1, 0],
                      [0, -1]])
roberts_y = np.array([[0, 1],
                      [-1, 0]])
gradient_x = cv2.filter2D(image, -1, roberts_x)
gradient_y = cv2.filter2D(image, -1, roberts_y)
gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Edge Detection (Roberts Operator)')
plt.axis('off')
plt.show()

2. Sobel Operator
•	Code-

import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Edge Detection (Sobel Operator)')
plt.axis('off')
plt.show()

3. Canny Edge Detection
•	Code-
mport cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)  # You can adjust the thresholds here
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

1. Harris Corner Detection
•	Code-
import cv2
from google.colab import files
from matplotlib import pyplot as plt
uploaded = files.upload()
for filename in uploaded.keys():
    img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show() 

1. Line Detection using Hough transformation
•	Code-
import cv2
import numpy as np
from google.colab import files
uploaded = files.upload()
for filename in uploaded.keys():
    img = cv2.imread(filename)
    edges = cv2.Canny(img, 50, 100)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        for r_thetha in lines:
            arr = np.array(r_thetha[0], dtype=np.float64)
            r, thetha = arr
            a = np.cos(thetha)
            b = np.sin(thetha)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('/content/linesdetected_img.jpg', img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

