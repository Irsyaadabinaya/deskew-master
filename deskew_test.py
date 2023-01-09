import math
import cv2
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
from pickle import TRUE
from typing import Tuple, Union
from PIL import Image
from cv2 import INTER_AREA
from matplotlib import image
from deskew import determine_skew
from typing import List

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def deskew(file_name, csv_file, iteration):

    # ppm_file = [cv2.imread(file) for file in glob.glob('deskew-master/*.ppm')]
    image = cv2.imread('gambar_source/'+ file_name +'.ppm')
    # cv2.imwrite('ready2skew.png',ppm_file)

    # image = cv2.imread('ready2skew.png')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)

    rotated = rotate(image, angle, (0, 0, 0))
    cv2.imwrite('output.png', rotated)

    hasil = cv2.imread('output.png')
    #cv2.imshow("hasil", hasil)

    #cek kemiringan
    new_grayscale = cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)
    new_angle = determine_skew(new_grayscale)
    akurasi = angle+new_angle
    print(file_name)
    print ("original angle =", angle)
    print ("new angle = ", new_angle)
    print (akurasi)

    # save accuracy data 
    # csv_file = open('akurasi.csv', 'w', newline='')
    with csv_file:
        header = ['Number', 'File name', 'Old Angle', 'New Angle']
        writer = csv.DictWriter(csv_file, fieldnames= header)

        writer.writeheader()
        writer.writerow({'Number' : iteration+1, 'File name' : file_name,
        'Old Angle' : angle, 'New Angle' : new_angle})

    # Resizing
    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    src = cv2.resize(image, dim, interpolation=INTER_AREA)

    # Custom window
    # cv2.namedWindow('Source', cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('Hasil setelah Skew Corrected', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Source', image)
    # cv2.imshow('Hasil setelah Skew Corrected', hasil)
    # cv2.resizeWindow('Hasil setelah Skew Corrected', 600, 600)
    # cv2.resizeWindow('Source', 600, 600)

    # plt.imshow(hasil)
    # plt.show()

    # save all images
    cv2.imwrite('gambar_hasil/'+file_name+'/' + file_name +'_hasil.png', hasil)
    cv2.imwrite('gambar_hasil/'+file_name+'/' + file_name +'_source.png', image)

    #plotting
    edges = canny(grayscale)
    h, a, d = hough_line(edges)
    _, ap, _ = hough_line_peaks(h, a, d, num_peaks=20)

    def display_hough(h, a, d):
        plt.imshow(
            np.log(1 + h),
            extent=[np.rad2deg(a[-1]), np.rad2deg(a[0]), d[-1], d[0]],
            cmap=plt.cm.gray,
            aspect=1.0 / 90)
        plt.savefig('gambar_hasil/'+file_name+'/' + file_name +'_plot.png')
        # plt.show()

    display_hough(h, a, d)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "gambar_source"  #insert image path 
dir_list = os.listdir(path) #make path directory as path
 
# print("Files and directories in '", path, "' :")
 
# prints all files
# print(dir_list[1])
# print(len(dir_list))

for i in range(len(dir_list)):
    csv_file = open('akurasi.csv', 'a', newline='')
    file_name = dir_list[i]
    deskew(file_name[:-4], csv_file, i)
    # print(file_name[:-4])


