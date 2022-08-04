import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

DEBUG = 0
def convertToNumpy(f, typedict, data, channelchoice = -1):
                 #call with channelchoice or default to  -1 to pick channel with higher amplitude
    if f.channels == 2:
        if channelchoice == -1:
            ch0 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[0::2]))
            ch1 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[1::2]))
            if ch0 > ch1:
                channelchoice = 0
            else:
                channelchoice = 1
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])[channelchoice::2]
    else:
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])
    return npdata

def getSamples(startsecs, Nsamples, WAV):
    # need to get Ntimes blocks of time series data
    channelchoice = -1    # pick channel with higher amplitude
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'

    NsamplesNeeded = Nsamples
    npsamples = []
    while NsamplesNeeded > 0:
        with sf.SoundFile(WAV) as f:
#            data = f.buffer_read(1000, dtype=typedict[f.subtype])
            availableSamples = f.seek(0, sf.SEEK_END) - int(startsecs*f.samplerate)
            f.seek(startsecs*f.samplerate)
            while availableSamples > 0 and NsamplesNeeded > 0:
                if availableSamples >= NsamplesNeeded:
                    data = f.buffer_read(NsamplesNeeded,  dtype=typedict[f.subtype])
                    npdata = convertToNumpy(f, typedict, data)
                    NsamplesNeeded = 0
                else:
                    data = f.buffer_read(availableSamples, dtype=typedict[f.subtype])
                    npdata = convertToNumpy(f, typedict, data)
                    NsamplesNeeded  -= availableSamples
                    startsecs = 0
                    availableSamples = 0

                if len(npsamples) == 0:
                    npsamples = npdata
                else:
                    npsamples = np.append(npsamples, npdata)
            f.close()
#    print("n samples", len(npsamples))
    return npsamples

def setupFreqBands(flow, fhigh, nbands, doLogs):
    df = (fhigh - flow) / nbands
    fbands = np.zeros(nbands)
    if not doLogs:
        for i in range(nbands):
            fbands[i] = flow + i*df
    else:
        dlogf = (np.log10(fhigh) - np.log10(flow)) / (nbands - 0)
        fbands[0] = flow
        for i in range(1, nbands):
            if DEBUG > 0:
                print("np.power(10,(i * dlogf))", np.power(10,(i * dlogf)))
            fbands[i] = np.power(10,np.log10(flow) + (i * dlogf))
        if DEBUG > 0:
            print("flow,fbands,fhigh",flow,fbands,fhigh)
    return fbands

"""
   Average and interpolate psds into nbands frequency bins
"""

def compressPsdSliceLog(freqs, psds, flow, fhigh, nbands, doLogs):
    compressedSlice = np.zeros(nbands + 1)  # totPwr in [0] and frequency of bands is flow -> fhigh in nBands steps
    #    print("Num freqs", len(freqs))
    idxPsd = 0
    idxCompressed = 0
    fbands = setupFreqBands(flow, fhigh, nbands, doLogs)
    dfbands = []
    for i in range(len(fbands)-1):
        df = fbands[i+1] - fbands[i]
        dfbands.append(df)
    dfbands.append(df)   # add one more to have 1 for 1 with fbands
    # integrate psds into fbands
    df = freqs[1] - freqs[0]   # this freq scale is linear as it comes from the wav samplerate
    totPwr = 0
    while freqs[idxPsd] <= fhigh and idxCompressed < nbands:
        # find index in freqs for the first fband
        inNewBand = False
        if DEBUG == 10:
            print(idxPsd, freqs[idxPsd] , fbands[idxCompressed])
        while freqs[idxPsd] < fbands[idxCompressed]:  # step through psd frequencies until greater than this fband
            idxPsd += 1
            inNewBand = True
        deltaf = freqs[idxPsd] - fbands[idxCompressed]  # distance of this psd frequency into this fband
        if DEBUG == 10: print(deltaf)
        if deltaf > dfbands[idxCompressed]:  # have jumped an entire band
            compressedSlice[idxCompressed + 1] += psds[idxPsd]*dfbands[idxCompressed]/df  # frac of psds = slice
            idxCompressed += 1
        else:
            pfrac = deltaf / df
            compressedSlice[idxCompressed + 1] += pfrac * psds[idxPsd]  # put frac of first pwr in psd
            if inNewBand:
                idxCompressed += 1
            idxPsd += 1
        if DEBUG == 10: print(idxPsd, idxCompressed, deltaf, inNewBand)
        if DEBUG == 10: print("")
    compressedSlice[0] = np.sum(compressedSlice)
    return compressedSlice
"""
     Ntimes is number of time bins, Nfreqs is number of frequency bins between f_low and f_high
     a spectrogram is calculated for all the samples in a time bin
     These spectrograms are interpolated (compressed) into the time and frequency bins
     The bottom row contains the total psd for each time slice
"""
def getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logFrequency, samplerate, samples):

    specGram = []
    samplesPerBin = len(samples) // Ntimes
    for i in range(Ntimes):
        data = samples[i*samplesPerBin: (i+1)*samplesPerBin]
        data = data * np.hamming(len(data))
#        data = data * np.blackman(len(data))
        spec = np.abs(np.fft.rfft(data))  #, 4096)) #Nfft))
        f_values = np.fft.fftfreq(len(data), d=1. / samplerate)
        spec = compressPsdSliceLog(f_values, spec, f_low, f_high, Nfreqs, logFrequency)
        specGram.append(spec)  # flip to put low frequencies at 'bottom' of array as displayed on screen

    #  transform array
    specGram = np.log10(np.flip(specGram) + 0.001)  # to avoid log(0)
    specGram = np.rot90(specGram, 3)  # this rotates the spectrogram so the bottom left is t_min and f_low
        ###COULD use  square roots etc to bring lower peaks up  i.e.  0.36  -> 0.6  -> 0.77
    return specGram

def getNorm(ary):  # normalizes array to the array mean +- 4 standard deviations
    nrows = ary.shape[1]
    bbmax = np.max(ary[nrows-1, :])    # get max of bottom row
    bbmin = np.min(ary[nrows - 1, :])
    ary[nrows-1, :] = (ary[nrows-1, :] - bbmin)/(bbmax - bbmin)  # normalize the bottom row to 0 -> 1
    aryMean = np.mean(ary[0:nrows - 1, :])
    aryStd  = np.std(ary[0:nrows - 1, :])
    ary[0:nrows - 1, :] = (ary[0:nrows - 1, :] - aryMean)/(4 * aryStd)
    ary[ary<-1] = -1
    ary[ary>1] = 1
    ary = ary/2.0 + 0.5  # I don't remember why this is in here
    return ary

