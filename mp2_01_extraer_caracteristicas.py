import cv2
import numpy as np
import sys
import math
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
import os
from collections import deque

def processImages(files):
    vectors = deque()
    for fname in files:
        file = f'{sys.argv[1]}/{fname}'
        image = im = cv2.imread(file) 
        pix_h = len(image)
        pix_w = len(image[0])
        blur = cv2.medianBlur(image, 15)

        tapon = blur[int(pix_h*0.0):int(pix_h*0.3), int(pix_w/2 - 50):int(pix_w/2 + 50)]

        edges_tapon = cv2.Canny(image=tapon, threshold1=35, threshold2=35)
        kernel = np.ones((1,3),np.uint8)
        er_tapon = cv2.erode(edges_tapon, kernel)
        M_tapon = cv2.moments(er_tapon, True)
        if not M_tapon["m00"] == 0:
            cX_tapon = int(M_tapon["m10"] / M_tapon["m00"])
            cY_tapon = int(M_tapon["m01"] / M_tapon["m00"])
            #tapon = tapon[cY_tapon - 40 if cY_tapon - 40 > 0 else 0:cY_tapon + 40, cX_tapon - 40 if cX_tapon - 40 > 0 else 0: cX_tapon + 40]
        hist_tapon = cv2.calcHist([tapon], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256])

        etiquetas = blur[int(pix_h*0.3):int(pix_h*0.8), int(pix_w/2 - 100):int(pix_w/2 + 100)]
        
        edges_et = cv2.Canny(image=etiquetas, threshold1=35, threshold2=35)
        kernel = np.ones((1,3),np.uint8)
        er_et = cv2.erode(edges_et, kernel)
        M_et = cv2.moments(er_et, True)
        cX_et = int(M_et["m10"] / M_et["m00"])
        cY_et = int(M_et["m01"] / M_et["m00"])

        size = 100
        #etiquetas = etiquetas[int(cY_et - size if cY_et - size > 0 else 0):int(cY_et + size),int(cX_et - size if cX_et - size > 0 else 0):int(cX_et + size)]
        hist_et = cv2.calcHist([etiquetas], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256])
        vector = np.append([f'{fname}'], [hist_tapon, hist_et])
        vectors.append(vector)
    return np.array(vectors)

def main():
    if not len(sys.argv) == 3:
        print(f"usage: {sys.argv[0]} <in_img_dir> <out_features_write_file>", file=sys.stderr)
        exit(1)
    np.set_printoptions(threshold=sys.maxsize) 
    np.set_printoptions(precision=2)
    directory = sys.argv[1]
    pool = Pool()
    for _, _, files in os.walk(directory):
        archivos = np.array(files)
        vectors = np.concatenate(pool.map(processImages, archivos.reshape((1, int(len(archivos)/1)))))
        np.savetxt(sys.argv[2], vectors, delimiter=',', fmt="%s")


if __name__ == "__main__":
    main()
