# Here are some thoughts about noise reduction (signal enhancement) in frequency space.

The goal:  Improve the signal to noise of orca vocalizations as background noise changes due to various ephemeral background noise source possibilities:  Wind, Waves, Boats, Ships, Other




Install these libraries using your IDE or python -m pip install numpy  etc.:

numpy
matplotlib
soundfile

1_simpleSpectrogram.py  shows how to calculate the individual psd elements and save
them in an array of prescribed size.  Here there are 256 x 256 psd values between specified frequency
limits and specified time limits.

2_simpleSpectrogramDenoise.py  shows a very simple way to subtract background noise from
a section of the wav file which does not have a call from the spectrogram of the call

Sample plot:
![Simple denoiser](plots/simpleDenoiser.jpg "Simple denoiser")
