import numpy as np
from scipy import signal


def butter_bandpass(low_cut, high_cut, fs, order=2):
    """
    Define butterworth with bandpass filter
    """
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    wn = [low, high]
    b, a = signal.butter(N=order, Wn=wn, btype='bandpass', output='ba')
    return b, a


def butter_lowpass(cut, fs, order=4):
    """
    Define butterworth with lowpass filter
    """
    nyq = 0.5 * fs
    low = cut / nyq
    b, a = signal.butter(N=order, Wn=low, btype='lowpass', output='ba')
    return b, a


def butter_highpass(cut, fs, order=4):
    """
    Define butterworth with highpass filter
    """
    nyq = 0.5 * fs
    high = cut / nyq
    b, a = signal.butter(N=order, Wn=high, btype='highpass', output='ba')
    return b, a


def butterworth_filter(x, low_cut=1, high_cut=20, fs=1000, order=4, mode='bandpass'):
    """
    Define butterworth filter
    """
    a = 0
    b = 0
    if mode == 'bandpass':
        b, a = butter_bandpass(low_cut, high_cut, fs, order)
    elif mode == 'lowpass':
        b, a = butter_lowpass(low_cut, fs, order)
    elif mode == 'highpass':
        b, a = butter_highpass(high_cut, fs, order)
    else:
        assert 'Not found \'{}\' mode!'.format(mode)
    return signal.lfilter(b, a, x)


def downsample_signal(x, kernel_size=3):
    num_signal = x.shape[0]
    x = x.reshape(num_signal, -1, int(kernel_size))
    return x.mean(axis=2)


def filter1d(x, low_cut=1, high_cut=20, order=4, mode='highpass'):
    """
    Filter 1-channel signal using bandpass butterworth filter
    """
    shifted = x - x.mean()    # Mean corrected or mean shift
    return butterworth_filter(shifted, low_cut=low_cut, high_cut=high_cut, order=order, mode=mode)


def filter2d(x, kernel_size=3, low_cut=1, high_cut=20, order=4, mode='highpass'):
    """
    Filter 4-channel signal using bandpass butterworth filter
    """
    if kernel_size != 0:
        x = downsample_signal(x, kernel_size)
    num_channels = x.shape[0]
    return np.array([filter1d(x[i], low_cut, high_cut, order, mode) for i in range(num_channels)])


def overlap(x, window_size, window_step):
    window_size, window_step = map(int, (window_size, window_step))
    if window_size % 2 != 0:
        raise ValueError('Window size must be even!')
    # Make sure there are an even number of windows before stride tricks
    pad = np.zeros((window_size - len(x) % window_size))
    x = np.hstack((x, pad))
    valid = len(x) - window_size
    n_windows = valid // window_step
    output = np.ndarray((n_windows, window_size), dtype=x.dtype)
    for i in range(n_windows):
        # Slide the window along the signal
        start = i * window_step
        stop = start + window_size
        output[i] = x[start: stop]
    return output


def normalize(x):
    """
    Normalize 4-channel signal. We rectify the signal before dividing to signal peak
    """
    rectified = abs(x)
    n_channels = x.shape[0]
    return np.array([rectified[i] / rectified[i].max() for i in range(n_channels)])


def stft1d(x, fft_size=128, step=65, real=False, compute_onesided=True):
    """
    Short-Time Fourier Transform for 1-channel signal
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fft_size // 2
    x = overlap(x, fft_size, step)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(fft_size) / (fft_size - 1))
    x = x * win[None]
    return local_fft(x)[:, :cut]


def spectrogram1d(x, log=True, thresh=5, fft_size=512, step_size=64):
    """
    Generate spectrogram for 1-channel signal
    """
    spectrogram = np.abs(stft1d(x, fft_size, step_size, real=False, compute_onesided=True))
    if log:
        spectrogram /= spectrogram.max()
        spectrogram = np.log10(spectrogram)
        spectrogram[spectrogram < -thresh] = -thresh
    else:
        spectrogram[spectrogram < thresh] = thresh
    return spectrogram


def spectrogram2d(x, fft_size=200, step_size=5):
    """
    Generate spectrogram for 4-channel signal
    """
    n_channels = x.shape[0]
    spec = np.array([spectrogram1d(x[i], fft_size=fft_size, step_size=step_size) for i in range(n_channels)])
    return np.transpose(spec, (0, 2, 1))  # channel x time_frame_window x freq
