import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import serial
import struct
import time

# Function to map ADC value (0-4095) to voltage (0-5V)
def map_adc_value_to_voltage(adc_value):
    max_adc_value = 4095
    max_voltage = 5
    return (adc_value / max_adc_value) * max_voltage

# Function to pad the signal to the nearest power of 2
def pad_to_power_of_two(signal):
    N = len(signal)
    next_power_of_two = 2**int(np.ceil(np.log2(N)))
    padded_signal = np.zeros(next_power_of_two)
    padded_signal[:N] = signal
    return padded_signal

# Serial port settings
ser = serial.Serial('COM3', baudrate=115200, timeout=1)

# Signal and FFT settings
fs = 44100  # Sampling rate (set as needed for your ADC)
n_fft = 2048
window = windows.hann(n_fft)

# Real-time plot setup
plt.ion()
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
freqs = np.fft.rfftfreq(n_fft, 1/fs)
data_buffer = np.zeros(n_fft)

try:
    while True:
        # Collect data from serial port until buffer is filled
        for i in range(n_fft):
            raw_data = ser.read(2)
            if len(raw_data) == 2:
                # Convert the 2-byte data into a 16-bit integer
                adc_value = struct.unpack('>H', raw_data)[0]
                voltage = map_adc_value_to_voltage(adc_value)  # Map ADC value to voltage
                data_buffer[i] = voltage

                print(voltage)
            else:
                print("Data not available. Check connection.")
                break

        # Apply window function and pad the segment
        segment = data_buffer * window
        padded_segment = pad_to_power_of_two(segment)
        
        # FFT processing with numpy's rfft for real values
        fft_values = np.abs(np.fft.rfft(padded_segment))
        smoothed_fft_values = np.convolve(fft_values, np.ones(5) / 5, mode='same')

        # Focus on 0 Hz to 6000 Hz
        valid_indices = (freqs >= 0) & (freqs <= 6000)
        freqs_focus = freqs[valid_indices]
        fft_values_focus = smoothed_fft_values[valid_indices]
        
        # Reduce to 61 points
        indices = np.linspace(0, len(freqs_focus) - 1, 61, dtype=int)
        reduced_freqs = freqs_focus[indices]
        reduced_fft_values = fft_values_focus[indices]

        # Top 4 frequency-amplitude pairs
        top_indices = np.argsort(reduced_fft_values)[-4:][::-1]
        top_freqs = reduced_freqs[top_indices]
        top_amps = reduced_fft_values[top_indices]

        max_amp = np.max(reduced_fft_values)
        top_percentages = (top_amps / max_amp) * 100 if max_amp > 0 else [0] * len(top_amps)

        # Table data
        table_data = [[f"{freq:.2f} Hz", f"{amp:.2f}", f"{percent:.2f} %"]
                      for freq, amp, percent in zip(top_freqs, top_amps, top_percentages)]

        # Update plots and tables
        ax.clear()
        ax_table.clear()
        ax.bar(reduced_freqs, reduced_fft_values, width=(reduced_freqs[1] - reduced_freqs[0]), align='center')
        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 5000)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Real-Time FFT Spectrum (0 Hz to 6000 Hz) with 61 Bars')

        # Display table
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, colLabels=["Frequency", "Amplitude", "% of Max"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.pause(0.04)  # Smooth update interval

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    ser.close()
    plt.ioff()
    plt.show()
