import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

    # Combine with normalization to match np.fft.fft
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    combined_fft = np.concatenate([
        even_fft + factor[:N // 2] * odd_fft,
        even_fft - factor[:N // 2] * odd_fft
    ])

    return combined_fft

# Function to pad the signal to the nearest power of 2
def pad_to_power_of_two(signal):
    N = len(signal)
    next_power_of_two = 2**int(np.ceil(np.log2(N)))
    padded_signal = np.zeros(next_power_of_two)
    padded_signal[:N] = signal
    return padded_signal

# Initial plotting function
def plot_signals(Hz=1):
    T = 1        # Total time in seconds
    fs = 120      # Increased sampling frequency in Hz to avoid aliasing
    t_continuous = np.linspace(0, T, fs)  # Time vector for continuous signal
    t_discrete = np.arange(0, T, 1/fs)    # Time vector for discrete signal

    # Create initial figure and subplots with a smaller size
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3, hspace=0.6)  # Increase bottom space and spacing between subplots

    # Initial signals
    continuous_signal = np.sin(2 * np.pi * Hz * t_continuous)
    discrete_signal = np.sin(2 * np.pi * Hz * t_discrete)

    # Plot initial signals
    line1, = axs[0].plot(t_continuous, continuous_signal, label='Continuous Signal', color='blue')
    axs[0].set_title(f'Continuous Sine Wave at {Hz} Hz')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid()
    axs[0].legend()

    markerline, stemlines, baseline = axs[1].stem(t_discrete, discrete_signal, label='Discrete Signal', basefmt=" ", linefmt='orange', markerfmt='ro')
    axs[1].set_title(f'Discrete Sine Wave at {Hz} Hz')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()
    axs[1].legend()

    # Initial FFT (NumPy's FFT)
    padded_signal = pad_to_power_of_two(discrete_signal)
    N = len(padded_signal)
    fft_result = np.fft.fft(padded_signal)
    fft_magnitude = np.abs(fft_result)[:N // 2] / (N // 2)  # Only positive frequencies
    fft_frequency = np.fft.fftfreq(N, 1/fs)[:N // 2]

    axs[2].stem(fft_frequency, fft_magnitude, linefmt='green', markerfmt='go', basefmt=" ")
    axs[2].set_title('FFT of Discrete Signal (Positive Frequencies)')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Magnitude')
    axs[2].grid()
    axs[2].set_xlim(0, 10)  # Set x-axis limits

    # Custom FFT (Cooley–Tukey)
    custom_fft_result = cooley_tukey_fft(padded_signal)
    custom_fft_magnitude = np.abs(custom_fft_result)[:N // 2] / (N // 2)  # Normalize and take positive frequencies

    axs[3].stem(fft_frequency, custom_fft_magnitude, linefmt='purple', markerfmt='mo', basefmt=" ")
    axs[3].set_title('Cooley–Tukey FFT (Positive Frequencies)')
    axs[3].set_xlabel('Frequency (Hz)')
    axs[3].set_ylabel('Magnitude')
    axs[3].grid()
    axs[3].set_xlim(0, 10)  # Set x-axis limits same as subplot 3

    # Add slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Hz', 0, 10.0, valinit=Hz, valstep=0.2)

    # Update function for the slider
    def update(val):
        new_Hz = slider.val
        continuous_signal = np.sin(2 * np.pi * new_Hz * t_continuous)
        discrete_signal = np.sin(2 * np.pi * new_Hz * t_discrete)

        # Update continuous signal plot
        line1.set_ydata(continuous_signal)
        axs[0].set_title(f'Continuous Sine Wave at {new_Hz} Hz')

        # Clear and redraw the discrete signal plot
        axs[1].cla()
        axs[1].stem(t_discrete, discrete_signal, label='Discrete Signal', basefmt=" ", linefmt='orange', markerfmt='ro')
        axs[1].set_title(f'Discrete Sine Wave at {new_Hz} Hz')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid()
        axs[1].legend()

        # Update NumPy FFT
        padded_signal = pad_to_power_of_two(discrete_signal)
        fft_result = np.fft.fft(padded_signal)
        fft_magnitude = np.abs(fft_result)[:N // 2] / (N // 2)

        axs[2].cla()
        axs[2].stem(fft_frequency, fft_magnitude, linefmt='green', markerfmt='go', basefmt=" ")
        axs[2].set_title('FFT of Discrete Signal (Positive Frequencies)')
        axs[2].set_xlabel('Frequency (Hz)')
        axs[2].set_ylabel('Magnitude')
        axs[2].grid()
        axs[2].set_xlim(0, 10)  # Set x-axis limits

        # Update Cooley–Tukey FFT
        custom_fft_result = cooley_tukey_fft(padded_signal)
        custom_fft_magnitude = np.abs(custom_fft_result)[:N // 2] / (N // 2)

        axs[3].cla()
        axs[3].stem(fft_frequency, custom_fft_magnitude, linefmt='purple', markerfmt='mo', basefmt=" ")
        axs[3].set_title('Cooley–Tukey FFT (Positive Frequencies)')
        axs[3].set_xlabel('Frequency (Hz)')
        axs[3].set_ylabel('Magnitude')
        axs[3].grid()
        axs[3].set_xlim(0, 10)  # Set x-axis limits

        fig.canvas.draw_idle()  # Redraw the canvas

    # Connect the slider to the update function
    slider.on_changed(update)

    # Apply tight layout with padding
    plt.tight_layout(pad=2.5)
    plt.show()

# Initial plot
plot_signals(Hz=1)
