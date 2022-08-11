import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import helpers as h
"""
 functions for data processing are in the helpers.py file
"""
def getSignalToNoise(callSpec, denoiseSpec, bkgndSpec):
    Nfreqs = callSpec.shape[0]
    Ntimes = callSpec.shape[1]
    aveBackgnd = np.mean(np.abs(bkgndSpec))
    S2N_1 = np.max(np.abs(callSpec))/aveBackgnd
    S2N_2 = np.max(np.abs(callSpec)) / np.mean(np.abs(callSpec))
    S2N_3 = np.max(np.abs(denoiseSpec)) / np.mean(np.abs(denoiseSpec))
    return S2N_1, S2N_2, S2N_3


def getSpectrogram(wavfile, wavStartSecs, Ntimes, Nfreqs, specduration, f_low, f_high, logFrequency):
    with sf.SoundFile(wavfile) as f:
        blocksize = int(specduration * f.samplerate // Ntimes)  # this is one bin in the final spectrogram
        samplerate = f.samplerate
        if f_high >= samplerate / 2:
            print("Selected high frequency is at or above Nyquist frequency which is too high")
            print("Setting f_high to 90% of Nyquist (1/2 samplerate")
            f_high = 0.9 * samplerate / 2.0
        samples = h.getSamples(wavStartSecs, int(specduration * f.samplerate), wavfile)
        ##samples is the wav file samples selected for the specified spectrogram duration
        spectrogram = h.getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logFrequency, samplerate, samples)
    return spectrogram

##########################################################################
specduration = 3   # make 3 second spectrograms
Ntimes       = 256  # number of power spectra to calculate for the spectrogram
Nfreqs       = 255  # will be 1 frequency bin less than spectrogram
                    # this is because the bottom row contains the total psd in each time slice
f_low = 200
f_high = 10000
logFrequency = True

callStartSecs  = 23    # time (s) of start of a call of specduration seconds

bkgndStartSecs = 15    # time (s) of start of a 'background' period with no call

wavfile = "../OS_9_27_2017_08_25_00__0002.wav"

callSpec = getSpectrogram(wavfile, callStartSecs, Ntimes, Nfreqs, specduration, f_low, f_high, logFrequency)
bkgndSpec = getSpectrogram(wavfile, bkgndStartSecs, Ntimes, Nfreqs, specduration, f_low, f_high, logFrequency)


denoise_1 = np.abs(callSpec - bkgndSpec)
sig2n_1, sig2n_2, sig2n_3 = getSignalToNoise(callSpec, denoise_1, bkgndSpec)
print("maximum of spectrogram/ mean of background =",sig2n_1)
print("max of abs spectrogram / mean of abs spectrogram", sig2n_2)
print("max of abs denoise_1/mean of abs denoise_1", sig2n_3)

fig, axes = plt.subplots(2, 2, figsize=(12, 13))  # plt.figure(8,3,figsize=(4, 3))
fig.suptitle("Spectrograms from {}".format( wavfile), fontsize = 16)
axes[0, 0].set_title("The orca call", fontsize = 16)
axes[0, 0].imshow(callSpec)
axes[0, 0].xaxis.set_visible(False)
axes[0, 0].yaxis.set_visible(False)
axes[0, 1].set_title("The background", fontsize = 16)
axes[0, 1].imshow(bkgndSpec)
axes[0, 1].xaxis.set_visible(False)
axes[0, 1].yaxis.set_visible(False)
axes[1, 0].set_title("Absolute value of difference", fontsize = 16)
axes[1, 0].imshow(denoise_1)
axes[1, 0].xaxis.set_visible(False)
axes[1, 0].yaxis.set_visible(False)

fig.tight_layout()
plt.show()
plt.savefig("plots/simpleDenoise.jpg")
plt.close()
