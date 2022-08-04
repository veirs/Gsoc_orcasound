import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import helpers as h
################## functions for data processing





###############################################

specduration = 3   # make 3 second spectrograms
Ntimes       = 256  # number of power spectra to calculate for the spectrogram
Nfreqs       = 255
f_low = 200
f_high = 10000
logFrequency = True

wavStartSecs  = 23    # index to move along wav file grabbing blocks of data for psds


wavfile = "../OS_9_27_2017_08_25_00__0002.wav"
#wavfile = "../Monika_Track L.wav"
with sf.SoundFile(wavfile) as f:
    blocksize = int(specduration * f.samplerate // Ntimes)  # this is one bin in the final spectrogram
    samplerate = f.samplerate
    if f_high >= samplerate / 2:
        print("Selected high frequency is at or above Nyquist frequency which is too high")
        print("Setting f_high to 90% of Nyquist (1/2 samplerate")
        f_high = 0.9 * samplerate/2.0
    samples = h.getSamples(wavStartSecs, int(specduration * f.samplerate), wavfile)
    ##samples is the wav file samples selected for the specified spectrogram duration
    spectrogram = h.getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logFrequency, samplerate, samples)

plt.imshow(spectrogram)
title = "Unfiltered and un-normalized\n{} at {} s".format(wavfile, wavStartSecs)
if logFrequency:
    title += "\n log of frequency scale"
plt.title(title)
plt.show()
plt.close()