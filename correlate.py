import numpy as np, PIL as img
from PIL import Image
import gc, random
import matplotlib.pyplot as plt

moves = np.array([[1,0], [0,1], [-1,0], [0,-1]])

rad = 7
im1 = Image.open("C:/Users/RedRussianBear/Downloads/DICT_Samples_Slides/Correlation_Example_Images/PIC00300.Tif").convert('L')
#im1 = im1.resize(map(lambda a: int(a/5), im1.size))
im2 = Image.open("C:/Users/RedRussianBear/Downloads/DICT_Samples_Slides/Correlation_Example_Images/PIC00310.Tif").convert('L')
#im2 = im2.resize(map(lambda a: int(a/5), im2.size))

gc.collect()

def correlatef(image1, image2):
	if (not (image1.size == image2.size)):
		return False
	size = image1.size
	w = size[0]
	h = size[1]

	a1 = np.array(image1)
	a2 = np.array(image2)

	f1 = np.fft.fft2(a1)
	f2 = np.fft.fft2(a2)

	inc = np.zeros(shape = (h, w), dtype=[('r', '>i4'), ('c', '>i4')])

	for r in  range(rad, h, 2*rad):
		for c in range(rad, w, 2*rad):

			mask = np.zeros(shape = (h, w), dtype = bool)
			mask[r - rad : r + rad + 1, c - rad : c + rad + 1] = True

			temp = np.vectorize(lambda a, m: a if m else 0)(a1, mask)
			ftemp = np.fft.fft2(temp)

			R = np.conj(ftemp) * f2
			cor = np.fft.ifft2(R)

			m = cor.argmax()

			inc[r,c] = (int(m%w) - r, int(m/h) - c)

	return inc

def correlates(image1, image2):
	if (not (image1.size == image2.size)):
		return False
	size = image1.size
	w = size[0]
	h = size[1]

	a1 = np.array(image1)
	a2 = np.array(image2)

	inc = np.zeros(shape = (h, w), dtype = [('r', '>i4'), ('c', '>i4'), ('cor', '>i4')])


	for r in range(rad, h - rad, 2*rad):
		for c in range(rad, w - rad, 2*rad):
			a = a1[r - rad : r + rad + 1, c - rad : c + rad + 1]
			b = a2[r - rad : r + rad + 1, c - rad : c + rad + 1]
			curcor = ((a - b)*(a - b)).sum()
			if(curcor == 0):
				inc[r,c] = np.array([0,0,0])
				continue
			
			maxima = np.zeros(shape = (60), dtype = [('r', '>i4'), ('c', '>i4'), ('cor', '>i4')])

			for i in range(0, 60):
				d0 = np.array([random.randint(rad, h - 1 - rad), random.randint(rad, w - 1 - rad)]) - np.array([r, c])
				d = d0
				dr = d[0]
				dc = d[1]
				b = a2[r - rad + dr : r + rad + 1 + dr, c - rad  + dc : c + rad + 1 + dc]
				curcor = ((a - b)*(a - b)).sum()

				while(True):
					d0 = d

					for step in moves:
						new = step + d0
						dr = new[0]
						dc = new[1]
						b = a2[r - rad + dr : r + rad + 1 + dr, c - rad  + dc : c + rad + 1 + dc]
						if(a.shape != b.shape):
							continue
						cor = ((a - b)*(a - b)).sum()
						if(cor > curcor):
							d = new
							curcor = cor
					if(np.array_equal(d, d0)):
						break

				maxima[i] = (d[0], d[1], curcor)

			mincor = 20000000
			maxd = 0
			for maximum in maxima:
				if(maximum[2] < mincor):
					mincor = maximum[2]
					maxd = maximum
			inc[r,c] = maxd

	return inc

x = correlates(im1, im2)

for i in range(0, x.shape[0]):
	for j in range(0, x.shape[1]):
		c = x[i,j]
		if(c[2] > rad*rad*2048):
			x[i,j] = (0, 0, 0)

X = im1.size[1] - np.nonzero(x)[0]
Y = np.nonzero(x)[1]
U = -1 * np.vectorize(lambda f: f[0])(x[np.nonzero(x)])
V = np.vectorize(lambda f: f[1])(x[np.nonzero(x)])

plt.quiver(Y, X, V, U)
plt.show()
