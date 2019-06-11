import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from scipy.signal import fftconvolve, convolve2d
from skimage.color import rgb2gray


# The value of an endline pixel in the signal
ENDLINE_VAL = 0.1
# The value of a black pixel in the image
BLACK_VAL = 0.2
# The length of the signal in seconds
SIGNAL_LEN = 10
# The wavenumber for the sending frequency.
CARRIER_FREQ = 10
# Signal that marks the start and end of the image.
START_END_SIG = np.array([ENDLINE_VAL, BLACK_VAL, ENDLINE_VAL])
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


def create_wavelets_im(size, waves):
    im = np.zeros((size, size))
    cen = size/2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - cen) ** 2 + (j - cen) ** 2)
            im[i, j] = np.cos((waves * np.pi * dist) / cen)
    im = (im + 1) / 2
    return im


def read_image(filename):
    """Reads an image from a file into a 3-dimensional tensor (height, width, channel)."""
    im = rgb2gray(imread(filename))
    im = im.astype(np.float64) / 255
    return im


def flatten_image(im):
    """
    Returns a 1D array where each value is the value of a pixel or endline.
    """
    im = im + BLACK_VAL
    #This part adds a column to the end of the image.
    im_with_endline = np.full((im.shape[0], im.shape[1] + 1), ENDLINE_VAL).astype(np.float64)
    im_with_endline[:, :-1] = im
    flattened = im_with_endline.flatten()
    flattened = np.concatenate([START_END_SIG, flattened, START_END_SIG])
    return flattened


# def prep_im_to_send(sig):
#     """
#     Duplicates the signal such that each pixel is represented by a certain number of array members.
#     """
#     sig_to_send = np.empty(0)
#     for i in range(0, sig.shape[0]):
#         pixel_sig = np.full((PIXEL_LEN,), sig[i])
#         sig_to_send = np.concatenate([sig_to_send, pixel_sig])
#     return sig_to_send


# Deprecated for now.
"""
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
"""


def modulate(sig, carrier_freq):
    t = np.arange(0, SIGNAL_LEN, SIGNAL_LEN/sig.shape[0])
    wave_sig = np.sin(carrier_freq * t)
    sig = sig * wave_sig
    sig = sig * AMPLITUDE + OFFSET
    return sig


def send_im(im, carrier_freq, show_sig=False):
    sig_to_send = flatten_image(im)
    send_x = np.arange(sig_to_send.shape[0])
    mod_sig = modulate(sig_to_send, carrier_freq)
    mod_x = np.arange(mod_sig.shape[0])
    if show_sig:
        plt.imshow(im, 'gray')
        plt.show()
        plt.plot(send_x, sig_to_send)
        plt.show()
        plt.plot(mod_x, mod_sig)
        plt.show()
        fourier_sig = np.fft.fft(mod_sig)
        plt.plot(np.real(fourier_sig))
        plt.plot(np.imag(fourier_sig), c='red')
        plt.show()
    return sig_to_send


def demodulate(sig, carrier_freq, plot=False):
    fourier_sig = np.fft.fft(sig)
    pos_fourier_sig = np.concatenate([fourier_sig[0:round(sig.shape[0]/2)], np.zeros(round(sig.shape[0]/2))])
    shifted_fourier_sig = np.roll(pos_fourier_sig, round(-2 * carrier_freq))
    demod_sig = np.fft.ifft(shifted_fourier_sig) * 2j
    if plot:
        plt.plot(np.real(fourier_sig))
        plt.plot(np.imag(fourier_sig), c='red')
        plt.show()
        plt.plot(np.real(shifted_fourier_sig))
        plt.plot(np.imag(shifted_fourier_sig), c='red')
        plt.show()
        plt.plot(np.real(demod_sig))
        plt.plot(np.imag(demod_sig), c='red')
        plt.show()
    return demod_sig


    # freq_interval = 2 * np.pi / sig.shape[0]
    # plt.plot(fourier_sig)
    # plt.show()
    # am_shift = round(carrier_freq / freq_interval)
    # # Shifts the signal in frequency space according to the carrier frequency.
    # fourier_sig = np.roll(fourier_sig, am_shift)
    # plt.plot(fourier_sig)
    # plt.show()
    # demod_sig = np.fft.ifft(fourier_sig)
    # plt.plot(demod_sig)
    # plt.show()


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


def test_demodulation():
    carrier_freq = 50
    sig_freq = 1
    samples = 1000
    T = 4 * np.pi
    x = np.arange(0, T, T/samples)
    orig_sig = np.array(x)
    fourier_orig = np.fft.fft(orig_sig)
    carrier_sig = np.sin(carrier_freq * x)
    send_sig = orig_sig * carrier_sig
    fourier_sig = np.fft.fft(send_sig)
    # The hard part
    freq_shift = 2 * carrier_freq
    print(freq_shift)
    # The positive half of the fourier signal.
    pos_fourier_sig = np.concatenate([fourier_sig[0:round(samples/2)], np.zeros(round(samples/2))])
    shifted_fourier_sig = np.roll(pos_fourier_sig, -freq_shift)
    demodulated = 2j * np.fft.ifft(shifted_fourier_sig)
    plt.plot(np.real(orig_sig))
    plt.plot(np.imag(orig_sig), c='red')
    plt.show()
    plt.plot(np.real(send_sig))
    plt.plot(np.imag(send_sig), c='red')
    plt.show()
    plt.plot(np.real(fourier_sig))
    plt.plot(np.imag(fourier_sig), c='red')
    plt.show()
    plt.plot(np.real(shifted_fourier_sig))
    plt.plot(np.imag(shifted_fourier_sig), c='red')
    plt.show()
    plt.plot(np.real(demodulated))
    plt.plot(np.imag(demodulated), c='red')
    plt.show()


if __name__ == "__main__":
    # im = read_image('images/sine.jpg')
    im = create_wavelets_im(50, 5)
    sig_to_send = send_im(im, CARRIER_FREQ, True)
    demodulated = demodulate(sig_to_send, CARRIER_FREQ, True)
