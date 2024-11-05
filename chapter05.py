import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import windows

# Custom Cooley–Tukey FFT function
def cooley_tukey_fft(x):
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2")

    # Divide: even and odd indexed elements
    even_fft = cooley_tukey_fft(x[::2])
    odd_fft = cooley_tukey_fft(x[1::2])
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even_fft + factor[:N // 2] * odd_fft,
                           even_fft - factor[:N // 2] * odd_fft])

# Function to pad the signal to the nearest power of 2
def pad_to_power_of_two(signal):
    N = len(signal)
    next_power_of_two = 2**int(np.ceil(np.log2(N)))
    padded_signal = np.zeros(next_power_of_two)
    padded_signal[:N] = signal
    return padded_signal

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
        # Apply window function and pad the segment to the nearest power of 2
        segment = audio_data * window
        padded_segment = pad_to_power_of_two(segment)
        
        # Apply custom Cooley–Tukey FFT
        fft_values = np.abs(cooley_tukey_fft(padded_segment)[:len(padded_segment) // 2 + 1])
        
        # Apply smoothing (moving average) for visualization
        smoothed_fft_values = np.convolve(fft_values, np.ones(5)/5, mode='same')
        
        # Focus on the 0 Hz to 20,000 Hz range
        valid_indices = (freqs >= 0) & (freqs <= 20000)
        freqs_focus = freqs[valid_indices]
        fft_values_focus = smoothed_fft_values[valid_indices]
        
        # Reduce data to 200 points for better visualization
        indices = np.linspace(0, len(freqs_focus) - 1, 200, dtype=int)
        reduced_freqs = freqs_focus[indices]
        reduced_fft_values = fft_values_focus[indices]
        
        # Find the top 4 frequency-amplitude pairs
        top_indices = np.argsort(reduced_fft_values)[-4:][::-1]
        top_freqs = reduced_freqs[top_indices]
        top_amps = reduced_fft_values[top_indices]
        
        # Calculate the percentage of each amplitude relative to the maximum amplitude
        max_amp = np.max(reduced_fft_values)
        if max_amp > 0:
            top_percentages = (top_amps / max_amp) * 100
        else:
            top_percentages = [0] * len(top_amps)
        
        # Create data for the table
        table_data = [[f"{freq:.2f} Hz", f"{amp:.2f}", f"{percent:.2f} %"]
                      for freq, amp, percent in zip(top_freqs, top_amps, top_percentages)]
        
        # Clear previous plots and tables
        ax.clear()
        ax_table.clear()
        
        # Plot the FFT spectrum with reduced data
        ax.bar(reduced_freqs, reduced_fft_values, width=(reduced_freqs[1] - reduced_freqs[0]), align='center')
        ax.set_xlim(0, 20000)  # Display only 0 Hz to 20,000 Hz range
        ax.set_ylim(0, 2.5)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Real-Time FFT Spectrum (0 Hz to 20,000 Hz) with 200 Bars')
        
        # Display the table with the top 4 frequency-amplitude pairs
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, colLabels=["Frequency", "Amplitude", "% of Max"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust table size
        
        plt.pause(0.02)  # Reduce pause time for smoother updates

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    stream.stop()
    plt.ioff()
    plt.show()
