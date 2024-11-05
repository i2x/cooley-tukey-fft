import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import windows

# Audio settings
fs = 44100  # Sampling rate
n_fft = 2048  # Number of FFT points
window = windows.hann(n_fft)  # Apply a Hann window to the segment

# Set up real-time plotting with subplots
plt.ion()  # Turn on interactive mode
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# FFT frequency bins up to Nyquist frequency
freqs = np.fft.rfftfreq(n_fft, 1/fs)

# Shared data structure for live plotting
audio_data = np.zeros(n_fft)

# Audio callback function for real-time input
def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(status)
    audio_data = indata[:, 0]  # Take the first channel

# Start the audio stream for real-time input
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=n_fft)
stream.start()

try:
    while True:
        # Apply window function and FFT
        segment = audio_data * window
        fft_values = np.abs(fft(segment)[:n_fft // 2 + 1])  # Include only positive frequencies
        
        # Apply smoothing (moving average) for visualization
        smoothed_fft_values = np.convolve(fft_values, np.ones(5)/5, mode='same')
        
        # Focus on the 60 Hz to 6000 Hz range
        valid_indices = (freqs >= 60) & (freqs <= 6000)
        freqs_focus = freqs[valid_indices]
        fft_values_focus = smoothed_fft_values[valid_indices]
        
        # Reduce data to 100 points for better visualization
        indices = np.linspace(0, len(freqs_focus) - 1, 100, dtype=int)
        reduced_freqs = freqs_focus[indices]
        reduced_fft_values = fft_values_focus[indices]
        
        # Find the top 4 frequency-amplitude pairs
        top_indices = np.argsort(reduced_fft_values)[-4:][::-1]
        top_freqs = reduced_freqs[top_indices]
        top_amps = reduced_fft_values[top_indices]
        
        # Calculate the percentage of each amplitude relative to the maximum amplitude
        max_amp = np.max(reduced_fft_values)
        top_percentages = (top_amps / max_amp) * 100
        
        # Create data for the table
        table_data = [[f"{freq:.2f} Hz", f"{amp:.2f}", f"{percent:.2f} %"]
                      for freq, amp, percent in zip(top_freqs, top_amps, top_percentages)]
        
        # Clear previous plots and tables
        ax.clear()
        ax_table.clear()
        
        # Plot the FFT spectrum with reduced data
        ax.bar(reduced_freqs, reduced_fft_values, width=(reduced_freqs[1] - reduced_freqs[0]), align='center')
        ax.set_xlim(60, 6000)  # Display only 60 Hz to 6000 Hz range
        ax.set_ylim(0, max(reduced_fft_values) * 1.1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Real-Time FFT Spectrum (60 Hz to 6000 Hz) with 100 Bars')
        
        # Display the table with the top 4 frequency-amplitude pairs
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, colLabels=["Frequency", "Amplitude", "% of Max"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust table size
        
        plt.pause(0.05)  # Reduce pause time for smoother updates

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    stream.stop()
    plt.ioff()
    plt.show()
