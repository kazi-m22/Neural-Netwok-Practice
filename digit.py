import cv2
import numpy as np
from PIL import Image
import pandas
import os

i0 = cv2.imread('./images/0_r.jpg')
# i1 = cv2.imread('./images/2.jpg',0)
# i2 = cv2.imread('./images/3.jpg',0)

# i0 = cv2.resize(i0, (28,28))
i0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(i0,230,255,cv2.THRESH_BINARY)
cv2.imshow('image', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()