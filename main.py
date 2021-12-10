import cv2
import numpy as np
import sys
import math


def main():
    image = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(image, 15)
    
    edgesXY = cv2.Sobel(blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    edges = cv2.Canny(image=blur, threshold1=100, threshold2=100)
    kernel = np.ones((1,2),np.uint8)
    er = cv2.erode(edges, kernel)

    interval = 50

    pix_h = len(image)
    pix_w = len(image[0])

    blocks_h = int(pix_h/interval)
    blocks_w = int(pix_w/interval)

    block_w_midpoint = blocks_w / 2
    block_h_midpoint = blocks_h / 2

    for i in [block_h_midpoint]:
        for j in [block_w_midpoint]:
            block = blur[\
                    int(i * interval - interval):int(i * interval + interval),\
                    int(j * interval - interval):int(j * interval + interval)\
                    ]
            cv2.imshow("block", block)
            #cv2.imwrite("test.png", edges)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
