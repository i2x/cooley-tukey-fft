import numpy as np
import serial  # For reading from COM port
import matplotlib.pyplot as plt
from scipy.signal import windows

# Custom Cooley–Tukey FFT function
def cooley_tukey_fft(x):
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2")

    even_fft = cooley_tukey_fft(x[::2])
    odd_fft = cooley_tukey_fft(x[1::2])

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

# Serial port settings
port = 'COM3'
baudrate = 115200
n_fft = 2048
window = windows.hann(n_fft)

# Set up real-time plotting
plt.ion()
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
freqs = np.fft.rfftfreq(n_fft, 1/44100)

try:
    # Initialize the serial port
    ser = serial.Serial(port, baudrate, timeout=1)

    while True:
        # Read a block of data from the serial port
        raw_data = ser.read(n_fft)
        if len(raw_data) == n_fft:
            # Convert raw bytes to NumPy array and normalize
            audio_data = np.frombuffer(raw_data, dtype=np.int16) / 32768.0
            
            # Pad audio_data to match n_fft
            padded_audio_data = np.zeros(n_fft)
            padded_audio_data[:len(audio_data)] = audio_data
            
            # Apply the window function to the padded data
            segment = padded_audio_data * window
            padded_segment = pad_to_power_of_two(segment)
            
            # Apply custom Cooley–Tukey FFT
            fft_values = np.abs(cooley_tukey_fft(padded_segment)[:len(padded_segment) // 2 + 1])
            smoothed_fft_values = np.convolve(fft_values, np.ones(5)/5, mode='same')

            valid_indices = (freqs >= 0) & (freqs <= 6000)
            freqs_focus = freqs[valid_indices]
            fft_values_focus = smoothed_fft_values[valid_indices]

            indices = np.linspace(0, len(freqs_focus) - 1, 61, dtype=int)
            reduced_freqs = freqs_focus[indices]
            reduced_fft_values = fft_values_focus[indices]

            top_indices = np.argsort(reduced_fft_values)[-4:][::-1]
            top_freqs = reduced_freqs[top_indices]
            top_amps = reduced_fft_values[top_indices]
            max_amp = np.max(reduced_fft_values)
            top_percentages = (top_amps / max_amp) * 100 if max_amp > 0 else [0] * len(top_amps)

            table_data = [[f"{freq:.2f} Hz", f"{amp:.2f}", f"{percent:.2f} %"]
                          for freq, amp, percent in zip(top_freqs, top_amps, top_percentages)]

            ax.clear()
            ax_table.clear()

            ax.bar(reduced_freqs, reduced_fft_values, width=(reduced_freqs[1] - reduced_freqs[0]), align='center')
            ax.set_xlim(0, 6000)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Real-Time FFT Spectrum (0 Hz to 6000 Hz) with 61 Bars')

            ax_table.axis('tight')
            ax_table.axis('off')
            table = ax_table.table(cellText=table_data, colLabels=["Frequency", "Amplitude", "% of Max"], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            plt.pause(0.02)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    ser.close()
    plt.ioff()
    plt.show()
