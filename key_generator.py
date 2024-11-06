import numpy as np
from scipy.io.wavfile import write

# Parameters
sample_rate = 44100  # 44.1 kHz sample rate
duration = 5  # 2 seconds duration
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Frequency components
frequency1 = 1000  # 1000 Hz
frequency2 = 2000  # 2000 Hz
frequency3 = 3000  # 3000 Hz

# Signal creation
signal = (4 * np.sin(2 * np.pi * frequency1 * t) +
          2 * np.sin(2 * np.pi * frequency2 * t) +
          np.sin(2 * np.pi * frequency3 * t))

# Normalize signal to prevent clipping
signal = signal / np.max(np.abs(signal))

# Convert to 16-bit PCM format
signal_pcm = np.int16(signal * 32767)

# Write to a .wav file
write("key.wav", sample_rate, signal_pcm)
