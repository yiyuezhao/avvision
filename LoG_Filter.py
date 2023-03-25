import cv2 
import numpy as np 
from numpy import random

# LoG滤波函数
def LoG_filter(img, K_size=3, K_sigma=3):
	
	H, W, C = img.shape

	# zero_padding
	pad = K_size // 2 
	out = np.zeros((H+2*pad, W+2*pad, C), dtype=np.float)
	out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

	# parameter settings
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad+K_size):
		for y in range(-pad, -pad+K_size):
			K[y + pad, x + pad] = (x ** 2 + y ** 2 - K_sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (K_sigma ** 2)))

	K /= (2 * np.pi * (K_sigma ** 6))
	# normalization
	K /= K.sum()

	tmp = out.copy()

	# filter
	for h in range(H):
		for w in range(W):
			out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

	out = np.clip(out, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out

# input_image
path = 'C:/Users/Administrator/Desktop/image/'


file_in = path + 'cake.jpg' 
file_out = path + 'LoG_filter.jpg' 
img = cv2.imread(file_in)

# obtain the size of the image
size = img.shape

# create the same noise integrated with the initial image
img_noise = 20 * random.standard_normal(size) #make the noise visible
img = img + img_noise

cv2.imwrite(path+'cake_noise.jpg', img)

out = LoG_filter(img, K_size=3, K_sigma=3)
# save the image
cv2.imwrite(file_out, out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
