import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

from scipy import ndimage as nd
#from scipy import misc

import skimage
from skimage.metrics import peak_signal_noise_ratio
from skimage import io
from skimage.color import rgb2grey



#from SSIM_PIL import compare_ssim

img = io.imread("gorilla.jpg")
print(img)
#greyimg = Image.open("grayscale.png").convert("L")
#greyimg = np.array(greyimg)
#print(np.shape(greyimg))

#print ((img.histogram()))  if detecting what kind of the image has asked.

#pix_val = list(img.getdata())
#print(pix_val) #visits each pixel

#iar = np.array(img)
#print(iar)



def convolve2d(image, kernel):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):  # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()
    return output


def median_FilterByHand(kernel, image):
        denoised_Image = np.empty(np.shape(image))
        mean = 0
        for i in range(len(image)):
            for y in range(len(image[0])):
                if i + len(kernel) <= len(image) and y + len(kernel) <= len(image[0]):
                    tempMatrix = np.matmul(image[i:i+len(kernel),y:y+len(kernel)],kernel)
                    tempMatrix = np.reshape(tempMatrix,(1, len(kernel) ** 2))
                    if len(kernel) % 2 == 0:
                        mean = (tempMatrix[0][int(math.pow(len(kernel), 2) / 2) - 1] + tempMatrix[0][
                            int(math.pow(len(kernel), 2) / 2)]) * 1 / 2
                    else:
                        mean = tempMatrix[0][math.ceil((math.pow(len(kernel), 2) / 2)) - 1]
                    denoised_Image[i][y] = mean
        return denoised_Image

def median_Filter(kernel,image):
    denoised_Image = np.asarray(image)
    mean = 0
    startP = len(kernel) / 2
    endPoint = math.ceil(len(kernel) / 2)
    for i in range(len(image)):
        for y in range(len(image[0])):
            if i+len(kernel) <= len(image) and y + len(kernel) <= len(image[0]) and (i-startP>0 and y-startP>0) and((image[i][y] == 0 or image[i][y] == 1)):
                tempMatrix = np.matmul(denoised_Image[i:i+len(kernel),y:y+len(kernel)],kernel)
                tempMatrix = np.reshape(tempMatrix,(1,len(kernel)**2))
                mean = np.mean(tempMatrix[0])
                denoised_Image[i][y] = mean
    return denoised_Image

def median_Filter2(kernel,image):
    denoised_Image = np.asarray(image)
    mean = 0
    startP = math.ceil(len(kernel))/2
    endPoint = math.ceil(len(kernel)/2)
    for i in range(len(image)):
        for y in range(len(image[0])):
            if i+len(kernel) < len(image) and y + len(kernel) < len(image[0]) and (i-startP>0 and y-startP>0):
                tempMatrix = np.matmul(image[i-startP:i+endPoint,y-startP:y+endPoint],kernel)
                tempMatrix = np.reshape(tempMatrix,(1,len(kernel)**2))
                mean = np.mean(tempMatrix[0])
                denoised_Image[i][y] = mean
    return denoised_Image



'''def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
'''

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


gaussian_greyimg = skimage.util.random_noise(img, mode="gaussian")
plt.imsave("gaussian_grey.png",gaussian_greyimg)
print(gaussian_greyimg)
psnrNoise = psnr(img, gaussian_greyimg)




gaussian_img = skimage.util.random_noise(img, mode="gaussian")
plt.imsave("gaussian.jpg",gaussian_img)
psnrNoise = psnr(img, gaussian_img)

#gaussian_filter = nd.gaussian_filter(gaussian_greyimg, sigma=3)   #filters works as, it averages the values in image and replaces all the pixel with it.Gaussian filter does not preserve edges
#plt.imsave("gaussian_filter.png", gaussian_filter,cmap="gray")
#psnrGaussian = psnr(gaussian_greyimg, gaussian_filter)


#median_filter = nd.median_filter(gaussian_greyimg, size=3)
#plt.imsave("median_filter.png", median_filter)
#psnrMedian = psnr(gaussian_greyimg, median_filter)


#from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_bilateral, denoise_tv_chambolle, denoise_wavelet

#sigma_est = np.mean(estimate_sigma(gaussian_greyimg, multichannel=False))   #To be used in nonlocal_means filter.  Multichannel is to determine if pic is RGB. Noise standart derivation.
'''
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

sp_img = skimage.util.random_noise(img, mode="s&p")
plt.imsave("sp.jpg",sp_img)
plt.imsave("greyimg.jpg",greyimg)
'''
#print(psnrNoise)
#print(psnrGaussian)
#print(psnrMedian)
#print(sigma_est)
#print(psnrNonlocal)
#print(psnrTv)
#print(psnrWavelet)

'''
kernel = np.array([[1/16,2/16,1/16],[2/16,4/16, 2/16],[1/16,2/16,1/16]])
image_gaussianblur = convolve2d(gaussian_greyimg,kernel)
print ('\n First 5 columns and rows of the image_gaussianblur matrix: \n', image_gaussianblur[:5,:5]*255)
plt.imsave("gaussianblurbyhand.png", image_gaussianblur)
psnrGaussBlurByHand = psnr(img, gaussian_greyimg)
#print(psnrGaussBlurByHand)

kernelMedian = np.identity(3)
image_medianblure = median_Filter(kernelMedian, gaussian_greyimg)
plt.imsave("medianfilterbyhand.jpg", image_medianblure)
psnrmedianbyhand = psnr(gaussian_greyimg, image_medianblure)
print(psnrmedianbyhand)

kernelMedian3 = np.identity(3)
image_medianblure3 = median_Filter(kernelMedian3, gaussian_greyimg)
plt.imsave("medianfilterbyhand2.jpg", image_medianblure3)
psnrmedianbyhand3 = psnr(gaussian_greyimg, image_medianblure3)
print(psnrmedianbyhand3)

kernelMedian2 = np.identity(250)
image_medianblure2 = median_FilterByHand(kernelMedian2, gaussian_greyimg)
plt.imsave("medianfilterbyhandmistake.jpg", image_medianblure2)
#psnrmedianmistake = psnr(gaussian_greyimg, image_medianblure2)
#print(psnrmedianmistake)

median_filterbyhand = median_filter(greyimg, 3)
median_filterbyhand2 = Image.fromarray(median_filterbyhand)
plt.imsave("medianfilterbyhand.jpg", median_filterbyhand2)
psnrmedianbyhand = psnr(greyimg, median_filterbyhand2)
print(psnrmedianbyhand)
'''

#value = compare_ssim(iar, np.uint8(gimg))
#print(value)

'''
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(img, gimg, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
'''



from functools import reduce, partial
from sklearn.decomposition import PCA
from sklearn.neighbors.ball_tree import BallTree
from skimage import data_dir
import skimage.color as color
from skimage.io import imread_collection
from skimage.transform import resize



'''
def nonlocalmeans(img, algorithm="clustered", **kwargs):
    if algorithm == "naive":
        return _nonlocalmeans_naive(img, **kwargs)
    if algorithm == "clustered":
        return _nonlocalmeans_clustered(img, **kwargs)


def _distance(values, pixel_window, h2, Nw):
    patch_window, central_diff = values

    diff = np.sum((pixel_window - patch_window) ** 2)
    # remove the central distance from the computation
    diff -= central_diff

    w = np.exp(-diff / (h2 * Nw))

    # return the color of the pixel and the weight associated with the patch
    nr, nc = patch_window.shape
    return w * patch_window[int(nr / 2), int(nc / 2)], w


def _nonlocalmeans_naive(img, n_big=20, n_small=5, h=10):
    new_n = np.zeros_like(img)

    Nw = (2 * n_small + 1) ** 2
    h2 = h * h
    n_rows = len(img)
    n_cols = len(img[0])

    # precompute the coordinate difference for the big patch
    D = range(-n_big, n_big + 1)
    big_diff = [(r, c) for r in D for c in D if not (r == 0 and c == 0)]

    # precompute coordinate difference for the small patch
    small_rows, small_cols = np.indices((2 * n_small + 1, 2 * n_small + 1)) - n_small

    padding = n_big + n_small
    n_padded = np.pad(img, padding, mode='reflect')

    for r in range(padding, padding + n_rows):
        for c in range(padding, padding + n_cols):
            pixel_window = n_padded[small_rows + r, small_cols + c]

            # construct a list of patch_windows
            windows = [n_padded[small_rows + r + d[0], small_cols + c + d[1]] for d in big_diff]

            # construct a list of central differences
            central_diffs = [(n_padded[r, c] - n_padded[r + d[0], c + d[1]]) for d in big_diff]

            distance_map = partial(_distance, pixel_window=pixel_window, h2=h2, Nw=Nw)
            distances = map(distance_map, zip(windows, central_diffs))

            total_c, total_w = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), distances)
            new_n[r - padding, c - padding] = total_c / total_w
        print(r)

    return new_n

'''
def nonlocalmeans_clustered(img, n_small=5, n_components=9, n_neighbors=30, h=10):

    Nw = (2 * n_small + 1) ** 2
    h2 = h * h
    n_rows, n_cols = img.shape

    # precompute the coordinate difference for the big patch
    small_rows, small_cols = np.indices(((2 * n_small + 1), (2 * n_small + 1))) - n_small

    # put all patches so we can cluster them
    n_padded = np.pad(img, n_small, mode='reflect')
    patches = np.zeros((n_rows * n_cols, Nw))

    n = 0
    for r in range(n_small, n_small + n_rows):
        for c in range(n_small, n_small + n_cols):
            window = n_padded[r + small_rows, c + small_cols].flatten()
            patches[n, :] = window
            n += 1

    transformed = PCA(n_components=n_components).fit_transform(patches)
    # index the patches into a tree
    tree = BallTree(transformed, leaf_size=2)

    new_img = np.zeros_like(img)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            _, ind = tree.query(transformed[[idx]], k=n_neighbors)
            ridx = np.array([(int(i / n_cols), int(i % n_cols)) for i in ind[0, 1:]])
            colors = img[ridx[:, 0], ridx[:, 1]]
            # compute real patch distances
            dist = [np.mean((patches[i] - patches[idx])**2) for i in ind[0, 1:]]
            w = np.exp(-np.array(dist) / h2)
            new_img[r, c] = np.sum(w * colors) / np.sum(w)

    return new_img


noise_var = np.logspace(-4, -1, 5)

col_dir = "lena.png"
lena=imread_collection(col_dir)[0].astype(np.float)/255



def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)

noise_var = np.logspace(-4, -1, 5)

lena = resize(lena, (lena.shape[0]/2, lena.shape[1]/2))
lena = color.rgb2lab(lena)
lena = lena[:,:,0]

noisy = []
for sigma2 in noise_var:
    noise = np.random.normal(0, np.sqrt(sigma2), lena.shape)
    n = lena + noise
    # avoid going over bounds
    n[n > 100] = 100
    n[n < 0] = 0
    noisy.append(n)

#img = noisy[3][:,:,0]
# img = np.zeros((100, 100))
# img[:50, :] = 1
#original = lena[:, :, 0]
#img += np.random.normal(0, 0.1, img.shape)

def estimate_noise(img):
    upper = img[:-2, 1:-1].flatten()
    lower = img[2:, 1:-1].flatten()
    left = img[1:-1, :-2].flatten()
    right = img[1:-1, 2:].flatten()
    central = img[1:-1, 1:-1].flatten()
    U = np.column_stack((upper, lower, left, right))
    c_estimated = np.dot(U, np.dot(np.linalg.pinv(U), central))
    error = np.mean((central - c_estimated)**2)
    sigma = np.sqrt(error)
    return sigma

sigmas = []
for i,n in enumerate(noisy):
    sigma = estimate_noise(n)
    sigmas.append(sigma)
    print("Estimated noise is {0:.4f}, real noise is {1:.4f}".format(sigma, np.sqrt(noise_var[i])))

denoised_clustered = []

plt.imsave("noisy.jpg", noisy[2])
for i,n in enumerate(noisy):
    denoised = nonlocalmeans_clustered(noisy[2], n_neighbors=30, n_small=1, h=2*sigmas[i])
    denoised_clustered.append(denoised)

    plt.imsave("denoisedbyhand.jpg",denoised,cmap='gray')


