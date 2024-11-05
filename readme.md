# Custom Cooley–Tukey FFT Visualization Script


## Requirements

Install them with:
```bash
pip install numpy matplotlib sounddevice scipy soundfile

```

### Contents
- chapter01  FFT concept
- chapter02  FFT vs SWEEP generator
- chapter03 - FFT vs .wave
- chapter04 - FFT vs input from microphone

## Overview
This Python script visualizes and compares continuous and discrete sine waves, their Fast Fourier Transform (FFT) using NumPy, and a custom implementation of the Cooley–Tukey FFT algorithm. It allows users to explore the frequency domain representation of sine waves and interactively adjust the frequency using a slider.

### Features
- Visualization of continuous and discrete sine waves.
- Comparison between NumPy's FFT and a custom Cooley–Tukey FFT.
- Interactive slider to change the signal frequency and observe its impact on the FFT.
- Well-labeled plots for easy interpretation.

## Purpose
The script serves as an educational tool to demonstrate the working of FFT and provides a side-by-side comparison of NumPy's FFT with a custom implementation of the Cooley–Tukey algorithm. It is ideal for:
- Learning signal processing fundamentals.
- Understanding how FFT algorithms work.
- Comparing custom and standard FFT implementations.

