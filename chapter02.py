import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import windows  # Corrected import

# Parameters for the sweep
fs = 44100  # Sampling frequency
duration = 5  # Duration of the sweep in seconds
start_freq = 100  # Start frequency in Hz
end_freq = 2000  # End frequency in Hz

# Generate the sweep signal
t = np.linspace(0, duration, int(fs * duration))
sweep_signal = np.sin(2 * np.pi * (start_freq + (end_freq - start_freq) * t / duration) * t)

# Set up real-time plotting with subplots
plt.ion()  # Turn on interactive mode
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# FFT settings
n_fft = 2048
freqs = np.fft.rfftfreq(n_fft, 1/fs)  # Only positive frequencies up to Nyquist
window = windows.hann(n_fft)  # Apply a Hann window to the segment

# Play the sweep signal in a non-blocking way
sd.play(sweep_signal, fs)

# Plot the FFT in real-time as a bar chart
for i in range(0, len(sweep_signal) - n_fft, fs // 10):  # Update every 0.1 second
    if not sd.get_stream().active:  # Check if the sound has stopped playing
        break

    segment = sweep_signal[i:i + n_fft] * window  # Apply window function
    fft_values = np.abs(fft(segment)[:n_fft // 2 + 1])  # Include up to Nyquist frequency
    
    # Find the top 4 frequency-amplitude pairs
    top_indices = np.argsort(fft_values)[-4:][::-1]  # Indices of top 4 amplitudes
    top_freqs = freqs[top_indices]
    top_amps = fft_values[top_indices]
    
    # Calculate the percentage of each amplitude relative to the maximum amplitude
    max_amp = np.max(fft_values)
    top_percentages = (top_amps / max_amp) * 100
    
    # Create data for the table
    table_data = [[f"{freq:.2f} Hz", f"{amp:.2f}", f"{percent:.2f} %"]
                  for freq, amp, percent in zip(top_freqs, top_amps, top_percentages)]
    
    # Clear previous plots and tables
    ax.clear()
    ax_table.clear()
    
    # Plot the FFT spectrum
    bars = ax.bar(freqs, fft_values, width=freqs[1] - freqs[0], align='center')
    ax.set_xlim(0, 5000)  # Display up to 5000 Hz
    ax.set_ylim(0, max(fft_values) * 1.1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Real-Time FFT Spectrum (Bar Chart)')
    
    # Display the table with the top 4 frequency-amplitude pairs
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data, colLabels=["Frequency", "Amplitude", "% of Max"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust table size
    
    plt.pause(0.1)

# Stop the sound and turn off interactive mode
sd.stop()
plt.ioff()
plt.show()
