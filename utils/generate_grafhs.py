import numpy as np
import matplotlib.pyplot as plt

pad = 2
line_width = 12


def plot_original_signal(y, save_file_location, x=None):
    plt.figure(figsize=(10, 4))

    # Plot with thicker line
    if x is not None:
        plt.plot(x, y, color='blue', linewidth=line_width)  # Set line thickness here
        plt.xlim(x[0], x[-1])  # Set x-axis limits to remove extra space
    else:
        plt.plot(y, color='blue', linewidth=line_width)  # Set line thickness here
        plt.xlim(0, len(y) - 1)  # Set x-axis limits to remove extra space

    # Set background to white
    plt.gca().set_facecolor('white')

    # Remove both x-axis and y-axis ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Thicken the border around the figure
    for spine in plt.gca().spines.values():
        spine.set_linewidth(line_width)  # Set border thickness

    # Adjust layout to remove the left margin (0 value sets the margin to the edge)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.tight_layout(pad=pad)

    # Save the figure
    plt.savefig(f'{save_file_location}.png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Saved file to {save_file_location}")


def plot_fft_box(y, save_file_location, x=None):
    plt.figure(figsize=(10, 4))

    # Plot the FFT data with thicker lines and larger red markers
    if x is not None:
        markerline, stemlines, baseline = plt.stem(x, y, linefmt='r-', basefmt=" ", markerfmt="ro")
    else:
        markerline, stemlines, baseline = plt.stem(y, linefmt='r-', basefmt=" ", markerfmt="ro")

    # Adjust the marker and line sizes
    plt.setp(markerline, markersize=line_width*1.5)  # Set marker size
    plt.setp(stemlines, linewidth=line_width)     # Set line width

    # Set background to white, remove ticks
    plt.gca().set_facecolor('white')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Thicken the border around the figure (black frame)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)  # Set border thickness to be visible

    # Use tight layout to force a better fit
    plt.tight_layout(pad=pad)

    # Save the figure with tight bounding box and some minimal padding (to preserve the black frame)
    plt.savefig(f'{save_file_location}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.show()
    print(f"Saved file to {save_file_location}.png")


def plot_box(y, save_file_location, x=None):
    plt.figure(figsize=(10, 4))

    # Plot with thicker line
    if x is not None:
        plt.plot(x, y, color='blue', linewidth=line_width)  # Set line thickness here
    else:
        plt.plot(y, color='blue', linewidth=line_width)  # Set line thickness here

    # Set background to white, remove ticks
    plt.gca().set_facecolor('white')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Thicken the border around the figure
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # Set border thickness
    plt.tight_layout(pad=pad)

    plt.savefig(f'{save_file_location}.png', bbox_inches='tight', dpi=300)  # Save as PNG
    plt.show()
    print(f"saved file to {save_file_location}")


# Step 1: Generate a signal of length 1000
np.random.seed(0)  # For reproducibility
signal_length = 1000
t = np.linspace(0, 1, signal_length)
original_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)+1.5*t*t  # Combination of two sine waves


plot_original_signal(original_signal,"original_signal",t)

# Step 3: Split the signal into windows of size 100 with overlap of 50
window_size = 100
overlap = 50
num_windows = (signal_length - overlap) // (window_size - overlap)

windows = [original_signal[i * (window_size - overlap):i * (window_size - overlap) + window_size] for i in range(num_windows)]

for i, window in enumerate(windows):
    plot_box(window,f"window_{i}",x=np.arange(100))

# Step 5: Perform FFT and plot the magnitude for each window
for i, window in enumerate(windows):
    fft_window = np.fft.fft(window)
    fft_magnitude = np.abs(fft_window)
    frequencies = np.fft.fftfreq(len(windows), 1 / 1)  # Frequency axis
    plot_fft_box(fft_magnitude[:len(frequencies)//2],f"mag_of_window_{i}",x=frequencies[:len(frequencies)//2])


# Step 6: Add noise to the original signal and plot it
noise = 0.1 * np.sin(2 * np.pi * 40 * t) + 0.3* np.cos(2 * np.pi * 10 * t)  # Small sine wave with low amplitude
noisy_signal = original_signal + noise

plot_box(noisy_signal,"noise_signal")


def plot_combined_signals(original, noisy, start, end, save_file_location, t=None):
    plt.figure(figsize=(10, 4))

    # Plot the original and noisy signals with thicker lines
    plt.plot(t, original, label="Original Signal", color='blue', linewidth=2)  # Thicker line for the original signal
    plt.plot(t[start:end], noisy[start:end], label="Noisy Signal", color='red', alpha=0.7,
             linewidth=line_width/2)  # Thicker line for the noisy signal

    # Set background to white, remove ticks
    plt.gca().set_facecolor('white')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Thicken the border around the figure
    for spine in plt.gca().spines.values():
        spine.set_linewidth(line_width)  # Set border thickness
    plt.tight_layout(pad=pad)

    plt.savefig(f'{save_file_location}.png', bbox_inches='tight', dpi=300)  # Save as PNG
    plt.show()
    print(f"Saved plot to {save_file_location}.png")


# Step 3: Plot the original and noisy signals from location 300 to 1000
plot_combined_signals(original_signal, noisy_signal, 380, 1000, "combined_signal", t=t)
