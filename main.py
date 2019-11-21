import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

from scipy import ndimage as nd
#from scipy import misc

import skimage
from skimage.metrics import peak_signal_noise_ratio

#from SSIM_PIL import compare_ssim

img = Image.open("gorilla.jpg")

#print ((img.histogram()))  if detecting what kind of the image has asked.

#pix_val = list(img.getdata())
#print(pix_val) #visits each pixel

iar = np.array(img)
#print(iar)

def noise_generator(noise_type, image):
    """
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255  #might add tuple here
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0      #might add tuple here
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        return image

'''
plt.figure(1)
sp_im = noise_generator('poisson', iar)
plt.imshow(sp_im)
plt.axis('off')
plt.show()
plt.close(1)
#print sp_im
'''

def signaltonoise(a, axis=0, ddof=0):

    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def psnr(img1, img2):                   #PSNR high means good quality
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    Pixel_max = 255.0
    return 20 * math.log10(Pixel_max / math.sqrt(mse))

'''
plt.figure(figsize=(18,24))
plt.subplot(421)
plt.imshow(iar)
plt.title('Original Image')
plt.axis("off")

gauss_im = noise_generator('gaussian', iar)
plt.subplot(422)
SNR = peak_signal_noise_ratio(iar, gauss_im)
print(SNR)
Psnr = psnr(iar, gauss_im)
print(Psnr)

plt.imshow(gauss_im)
plt.title('Gaussian Noise')
plt.axis("off")

sp_im = noise_generator('s&p', iar)
plt.subplot(423)
SNR = peak_signal_noise_ratio(iar, sp_im)
print(SNR)
plt.title('Salt & Pepper Noise')
plt.imshow(sp_im)
plt.axis("off")
plt.show()
'''

def plotnoise(iar, mode, r, c, i):     #Uses Skimage
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(iar, mode=mode)  #produced noise
        SNR = peak_signal_noise_ratio(iar, gimg)
        PSNR = psnr(iar, gimg)
        plt.imshow(gimg)
        print(SNR)
        print(PSNR)
    else:
        plt.imshow(iar)
    plt.title(mode)
    plt.axis("off")


#SSIM Structural Similarity:Used for measuring the similarity between two images.

'''
plt.figure(figsize=(18,24))
r=4
c=2
plotnoise(iar, "gaussian", r,c,1)
plotnoise(iar, "localvar", r,c,2)
plotnoise(iar, "poisson", r,c,3)
plotnoise(iar, "salt", r,c,4)
plotnoise(iar, "pepper", r,c,5)
plotnoise(iar, "s&p", r,c,6)
plotnoise(iar, "speckle", r,c,7)
plotnoise(iar, None, r,c,8)
plt.show()
'''

gaussian_img = skimage.util.random_noise(iar, mode="gaussian")
plt.imsave("gaussian.jpg",gaussian_img)
psnrNoise = psnr(iar, gaussian_img)

gaussian_filter = nd.gaussian_filter(gaussian_img, sigma=3)   #filters works as, it averages the values in image and replaces all the pixel with it.Gaussian filter does not preserve edges
plt.imsave("gaussian_filter.jpg", gaussian_filter)
psnrGaussian = psnr(gaussian_img, gaussian_filter)


median_filter = nd.median_filter(gaussian_img, size=3)
plt.imsave("median_filter.jpg", median_filter)
psnrMedian = psnr(gaussian_img, median_filter)


from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_bilateral, denoise_tv_chambolle, denoise_wavelet

sigma_est = np.mean(estimate_sigma(gaussian_img, multichannel=True))   #To be used in nonlocal_means filter.
nonlocal_means = denoise_nl_means(gaussian_img, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)
plt.imsave("nonlocal_means.jpg", nonlocal_means)
psnrNonlocal = psnr(gaussian_img, nonlocal_means)

tv_chambolle_filter= denoise_tv_chambolle(gaussian_img, weight=0.1, multichannel=True)
plt.imsave("tv_chambolle.jpg", tv_chambolle_filter)
psnrTv=psnr(gaussian_img,tv_chambolle_filter)

#bilateral_filter=denoise_bilateral(gaussian_img, sigma_color=0.05, sigma_spatial=15, multichannel=True)   #Runs too slow.
#plt.imsave("bilateral.jpg", bilateral_fiter)

wavelet_filter=denoise_wavelet(gaussian_img, multichannel=True, rescale_sigma=True)
plt.imsave("wavelet.jpg", wavelet_filter)
psnrWavelet = psnr(gaussian_img, wavelet_filter)     #How noiser than original

sp_img = skimage.util.random_noise(iar, mode="s&p")
plt.imsave("sp.jpg",sp_img)

print(psnrNoise)
print(psnrGaussian)
print(psnrMedian)
print(sigma_est)
print(psnrNonlocal)
print(psnrTv)
print(psnrWavelet)


#value = compare_ssim(iar, np.uint8(gimg))
#print(value)

'''
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(img, gimg, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
'''