import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 44100  # in Hz
duration = 5  # in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Frequencies
frequencies = [1000, 2000, 4000]  # in Hz

# Generate combined sine wave
combined_wave = sum(0.5 * np.sin(2 * np.pi * f * t) for f in frequencies)  # amplitude reduced to avoid clipping

# Save to .wav file
filename = "key.wav"
write(filename, sampling_rate, np.int16(combined_wave * 32767))  # Convert to 16-bit PCM format
print(f"Generated {filename}")

# Optional: Plot combined sine wave
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], combined_wave[:1000])  # Plot a small segment for clarity
plt.title("Combined Sine Wave (1000 Hz + 2000 Hz + 4000 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
