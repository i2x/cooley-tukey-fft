import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import windows  # Corrected import
import soundfile as sf

# Load the .wav file
filename = 'input.wav'  # Replace with your .wav file
sweep_signal, fs = sf.read(filename)

# Ensure the audio is mono for simplicity
if len(sweep_signal.shape) > 1:
    sweep_signal = sweep_signal[:, 0]  # Take the first channel if stereo

# Set up real-time plotting with subplots
plt.ion()  # Turn on interactive mode
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# FFT settings
n_fft = 2048
freqs = np.fft.rfftfreq(n_fft, 1/fs)  # Only positive frequencies up to Nyquist
window = windows.hann(n_fft)  # Apply a Hann window to the segment

# Play the audio in a non-blocking way
sd.play(sweep_signal, fs)

# Plot the FFT in real-time as a bar chart
update_step = fs // 50  # Increase the update rate for smoother visualization
for i in range(0, len(sweep_signal) - n_fft, update_step):  # Update more frequently
    if not sd.get_stream().active:  # Check if the sound has stopped playing
        break

    segment = sweep_signal[i:i + n_fft] * window  # Apply window function
    fft_values = np.abs(fft(segment)[:n_fft // 2 + 1])  # Include up to Nyquist frequency
    
    # Apply smoothing (moving average) for visualization
    smoothed_fft_values = np.convolve(fft_values, np.ones(5)/5, mode='same')
    
    # Reduce data to 50 points for better visualization
    indices = np.linspace(0, len(freqs) - 1, 50, dtype=int)
    reduced_freqs = freqs[indices]
    reduced_fft_values = smoothed_fft_values[indices]
    
    # Find the top 4 frequency-amplitude pairs
    top_indices = np.argsort(reduced_fft_values)[-4:][::-1]  # Indices of top 4 amplitudes
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
    bars = ax.bar(reduced_freqs, reduced_fft_values, width=(reduced_freqs[1] - reduced_freqs[0]), align='center')
    ax.set_xlim(0, 20000)  # Display up to 20,000 Hz
    ax.set_ylim(0, max(reduced_fft_values) * 1.1)
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
    
    plt.pause(0.05)  # Reduce pause time for smoother updates

# Stop the sound and turn off interactive mode
sd.stop()
plt.ioff()
plt.show()
