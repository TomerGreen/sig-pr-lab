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
    fourier_sig = np.fft.fftshift(fourier_sig)
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


def test_simple_demodulation():
    x = np.arange(0, 30, 0.01)
    sine_sig = np.sin(x)
    freq_sig = np.fft.fft(sine_sig)
    original = np.fft.ifft(freq_sig)
    plt.plot(sine_sig)
    plt.show()
    plt.scatter(x, np.abs(freq_sig))
    plt.show()
    plt.plot(original)
    plt.show()
    # a = np.zeros(1000)
    # a[100] = 1
    # b = np.fft.ifft(a)
    # plt.plot(a)
    # plt.show()
    # plt.plot(b)
    # plt.show()


def test_demodulation():
    x = np.arange(0, 10, 0.01)
    sine_sig = np.sin(x)
    carrier_sig = np.cos(10*x)
    send_sig = sine_sig * carrier_sig
    fourier_sig = np.fft.fft(send_sig)
    # not sure about this part
    freq_interval = 2 * np.pi / 10
    freq_shift = round(freq_interval * 10)
    print(freq_shift)
    shifted_fourier_sig = np.roll(fourier_sig, -freq_shift)
    demodulated = np.fft.ifft(shifted_fourier_sig)
    plt.plot(send_sig)
    plt.show()
    plt.scatter(x, fourier_sig, s=0.1)
    plt.show()
    plt.plot(shifted_fourier_sig)
    plt.show()
    plt.plot(demodulated)
    plt.show()


if __name__ == "__main__":
    # im = read_image('images/sine.jpg')
    # im = PYRAMID
    # freq = 2 * np.pi * CARRIER_WAVENUMBER / PIXEL_LEN
    # print(freq)
    # sig_to_send = send_im(im, freq, True)
    # demodulated = demodulate(sig_to_send, freq)
    # test_demodulation()
    test_demodulation()
