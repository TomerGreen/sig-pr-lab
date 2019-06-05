import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from scipy.signal import fftconvolve, convolve2d
from skimage.color import rgb2gray


# The value of an endline pixel in the signal
ENDLINE_VAL = 0
# The value of a black pixel in the image
BLACK_VAL = 0.5
# The number of values that represent each pixel in the array.
PIXEL_LEN = 1000
# The wavenumber for the sending frequency.
CARRIER_WAVENUMBER = 1
# The sent signal will start with this many pixels of endline_val.
START_SIG_LEN = 3
# Amplitude in volts of the signal sent.
AMPLITUDE = 0.05
# Voltage representing 0 in the sent signal.
OFFSET = 0.1


PYRAMID = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0],
    [0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2,   0],
    [0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.4, 0.2,   0],
    [0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2,   0],
    [0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.4, 0.2,   0],
    [0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2,   0],
    [0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0]
])



def read_image(filename):
    """Reads an image from a file into a 3-dimensional tensor (height, width, channel)."""
    im = rgb2gray(imread(filename))
    im = im.astype(np.float64) / 255
    return im


def flatten_image(im):
    im = im + BLACK_VAL
    #This part adds a column to the end of the image.
    im_with_endline = np.full((im.shape[0], im.shape[1] + 1), ENDLINE_VAL).astype(np.float64)
    im_with_endline[:, :-1] = im
    flattened = im_with_endline.flatten()
    # Adding the start signal. One more pixel is added because of linearlization of the signal.
    flattened = np.concatenate([np.full((START_SIG_LEN + 1,), ENDLINE_VAL), flattened])
    return flattened


def prep_im_to_send(sig):
    # The start signal
    send_sig = np.full((PIXEL_LEN * START_SIG_LEN,), ENDLINE_VAL).astype(np.float64)
    for i in range(1, sig.shape[0]):
        # Adding data for the pixel value and a linear increase to from the previous pixel
        pixel_line = np.linspace(sig[i-1], sig[i], PIXEL_LEN)
        pixel_plateau = np.full((PIXEL_LEN,), sig[i])
        send_sig = np.concatenate([send_sig, pixel_line])
        send_sig = np.concatenate([send_sig, pixel_plateau])
    return send_sig


def modulate(sig, carrier_freq):
    t = np.arange(sig.shape[0])
    wave_sig = np.cos(carrier_freq * t)
    sig = sig * wave_sig
    sig = sig * AMPLITUDE + OFFSET
    return sig


def send_im(im, carrier_freq, show_sig=False):
    plt.imshow(im, 'gray')
    plt.show()
    flattened = flatten_image(im)
    sig_to_send = prep_im_to_send(flattened)
    send_x = np.arange(sig_to_send.shape[0])
    mod_sig = modulate(sig_to_send, carrier_freq)
    mod_x = np.arange(mod_sig.shape[0])
    if show_sig:
        plt.plot(send_x, sig_to_send)
        plt.show()
        plt.plot(mod_x, mod_sig)
        plt.show()
    return sig_to_send


def demodulate(sig, carrier_freq):
    fourier_sig = np.fft.fft(sig)
    freq_interval = 2 * np.pi / sig.shape[0]
    plt.plot(fourier_sig)
    plt.show()
    am_shift = round(carrier_freq / freq_interval)
    # Shifts the signal in frequency space according to the carrier frequency.
    fourier_sig = np.roll(fourier_sig, am_shift)
    plt.plot(fourier_sig)
    plt.show()
    demod_sig = np.fft.ifft(fourier_sig)
    plt.plot(demod_sig)
    plt.show()


if __name__ == "__main__":
    # im = read_image('images/sine.jpg')
    im = PYRAMID
    freq = 2 * np.pi * CARRIER_WAVENUMBER / PIXEL_LEN
    print(freq)
    sig_to_send = send_im(im, freq, False)
    demodulated = demodulate(sig_to_send, freq)
