import time

import cv2

s_time = time.time()
a = cv2.imread(r'C:\Users\jdah5454\PycharmProjects\classification\animals\animals\antelope\0a37838e99.jpg')
e_time = time.time()

print(e_time - s_time)