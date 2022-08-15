"""
Script that contains various basic python functions for daily calculations
"""
import numpy as np
import scipy.io as sio
from obspy.core import UTCDateTime, Stream
import pandas as pd
import pyasdf


def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n


def amp_fft(signal, sampling_rate, pad=1, window=False, resample_log=False):
    """ Function to get single sided fft"""
    signal = signal - np.mean(signal)  # detrend
    hann = np.hanning(len(signal))
    total_length_signal = nextpow2(len(signal) * pad)

    if window is True:
        signal_fft = np.fft.fft(signal * hann, n=total_length_signal)
    elif window is False:
        signal_fft = np.fft.fft(signal, n=total_length_signal)

    signal_fft = signal_fft[0:int(total_length_signal / 2 + 1)]
    signal_fft = signal_fft / len(signal)  # normalise
    signal_fft[1:-1] = signal_fft[1:-1] * 2  # single sided, that is why times two
    freq = np.arange(0, sampling_rate / 2 + sampling_rate / total_length_signal, sampling_rate / total_length_signal)
    res = freq[1:2][0]

    if resample_log:
        freq_int = np.logspace(0.1, 5, num=10000)
        signal_fft_interp = np.interp(freq_int, freq, signal_fft, left=None, right=None, period=None)
        return signal_fft_interp, freq_int, res
    else:
        return signal_fft, freq, res


def test_signal():
    start = 0
    stop = 3
    fs = 1e2
    step = 1/fs
    time = np.arange(start, stop, step)

    y = .5 * np.cos(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 40 * time)
    y = y - np.mean(y)
    return time, y, fs


def load_Grimsel_catalog(path_to_catalog):
    # load catalogue
    mat_contents = sio.loadmat(path_to_catalog)
    eve = mat_contents['eve']

    # Change time to UTC time
    for i in range(len(eve[0, :])):
        time_mu = str(eve[0, i]['time_mu'][0])
        time_mu = time_mu.replace('_', '.')
        time_mu = UTCDateTime('2017-02-09T' + time_mu)
        eve[0, i]['time_mu'] = time_mu
    eve_df = pd.DataFrame.from_dict(eve[0])  # transform to pandas dataframe
    return eve_df


def load_stream_Grimsel(start_time, end_time, asdf_ini, stanos=None):
    stream_1 = Stream()
    if stanos:
        for stano in stanos:
            stream_1 += asdf_ini.get_waveforms(network='GRM',
                                               station=str(stano).zfill(3),
                                               location="001", channel="001",
                                               starttime=start_time,
                                               endtime=end_time,
                                               tag="raw_recording")
    else:
        for station_nr in enumerate(asdf_ini.waveforms.list()):
            stream_1 += asdf_ini.get_waveforms(network=station_nr[1][:station_nr[1].rfind('.')],
                                               station=station_nr[1][station_nr[1].rfind('.')+1:],
                                               location="001", channel="001",
                                               starttime=start_time,
                                               endtime=end_time,
                                               tag="raw_recording")
    return stream_1


def flatten_list_of_list(regular_list):
    flat_list = list(np.concatenate(regular_list).flat)
    return flat_list


def maybeidx(vector, v):
    """ Function to see at what index value is in list"""
    try:
        return vector.index(v)
    except ValueError:
        return False
