import numpy as np
import matplotlib.pyplot as plt

def blackman_harris_window(N):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def stft(x, w, h, win_fn=blackman_harris_window):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((w, nwin), dtype=np.complex)
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[h*j:h*j+w]
        # Zeropad if necessary
        if len(xj) < w:
            xj = np.concatenate((xj, np.zeros(w-len(xj))))
        # Apply window function
        xj = win_fn(w)*xj
        # Put the fourier transform into S
        S[:, j] = np.fft.fft(xj)
    return S

def amplitude_to_db(S, amin=1e-10, ref=1):
    """
    Convert an amplitude spectrogram to be expressed in decibels
    
    Parameters
    ----------
    S: ndarray(win, T)
        Amplitude spectrogram
    amin: float
        Minimum accepted value for the spectrogram
    ref: int
        0dB reference amplitude
        
    Returns
    -------
    ndarray(win, T)
        The dB spectrogram
    """
    SLog = 20.0*np.log10(np.maximum(amin, S))
    SLog -= 20.0*np.log10(np.maximum(amin, ref))
    return SLog

def istft(S, w, h):
    """
    Compute the complex inverse Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    S: ndarray(w, nwindows, dtype=np.complex)
        Complex spectrogram
    w: int
        Window length
    h: int
        Hop length
    
    Returns
    -------
    y: ndarray(N)
        Audio samples of the inverted STFT
    """
    N = (S.shape[1]-1)*h + w # Number of samples in result
    y = np.zeros(N)
    for j in range(S.shape[1]):
        xj = np.fft.ifft(S[:, j])
        y[j*h:j*h+w] += np.real(xj)
    y /= (w/h/2)
    return y
