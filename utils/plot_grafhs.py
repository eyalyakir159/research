import matplotlib.pyplot as plt
import numpy as np



def plot_graphs_four_inputs(x, y1, y2, y3, y4,ytitle="",plt_name=False):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot y1 and y2 on the left y-axis
    ax1.plot(x, y1, 'bs-', label='L = 96')  # Blue squares line for y1
    ax1.plot(x, y2, 'g^-', label='L = 192')         # Green triangles line for y2
    ax1.set_xlabel('Sequence Length (L)')
    ax1.set_ylabel(ytitle, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.plot(x, y3, 'rd-', label='L = 336')   # Red diamonds line for y3
    ax1.plot(x, y4, 'ko-', label='L = 720')


    # Combine the legends from both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1 , labels_1 , loc='upper right')

    # Show grid and plot
    ax1.grid(True)
    plt.show()
    if plt_name:
        plt.savefig(plt_name, format="pdf")  #
def plot_metrics_nfft(x, rmse, mae,plt_name=False):
    # Replace 25 with infinity in the x-ticks, but keep 25 in the data
    x = [25 if val == 0 else val for val in x]  # Replace 0 with 25 in the data

    x_sorted, y1_sorted, y2_sorted = zip(*sorted(zip(x, rmse, mae)))

    # Convert back to lists (if needed)
    x = list(x_sorted)
    rmse = list(y1_sorted)
    mae = list(y2_sorted)

    fig, ax1 = plt.subplots()

    # Plot RMSE on the left y-axis
    ax1.set_xlabel(r'$N_{\mathrm{FFT}}$ - Window size / FFT resolution')
    ax1.set_ylabel('RMSE', color='blue')  # Change to match left axis RMSE
    ax1.plot(x, rmse, 'bs-', label='RMSE')  # Blue squares with lines for RMSE
    ax1.tick_params(axis='y', labelcolor='blue')  # Set ticks to blue

    # Adjusting x-axis ticks to show powers of 2 explicitly
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Change tick labels, replace 25 with infinity symbol
    x_labels = [str(int(val)) for val in x]
    ax1.set_xticklabels(x_labels)

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE', color='red')  # Change to match right axis MAE
    ax2.plot(x, [m + 0.0005 for m in mae], 'rd-', label='MAE')  # Red diamonds with lines for MAE, adding a small gap
    ax2.tick_params(axis='y', labelcolor='red')  # Set ticks to red

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))  # Adjust this value to move left

    #plt.title('RMSE and MAE vs $N_{\mathrm{FFT}}$')
    fig.tight_layout()
    if plt_name:
        plt.savefig(plt_name, format="pdf")  # This saves the plot as a PDF file
    plt.show()

import matplotlib.pyplot as plt

def plot_metrics_seqlen(x, rmse, mae, plt_name=False):
    # Replace 0 with 25 in the x-ticks, but keep 0 in the data if required
    x = [25 if val == 0 else val for val in x]

    # Sorting the data by x-values
    x_sorted, y1_sorted, y2_sorted = zip(*sorted(zip(x, rmse, mae)))

    # Convert back to lists if needed
    x = list(x_sorted)
    rmse = list(y1_sorted)
    mae = list(y2_sorted)

    # Remove '48' from the x-ticks
    x_ticks = [val for val in x if val != 48]

    fig, ax1 = plt.subplots()

    # Plot RMSE on the left y-axis
    ax1.set_xlabel("Lookback Window Size (L)")
    ax1.set_ylabel('RMSE', color='blue')  # Change to match left axis RMSE
    ax1.plot(x, rmse, 'bs-', label='RMSE')  # Blue squares with lines for RMSE
    ax1.tick_params(axis='y', labelcolor='blue')  # Set ticks to blue

    # Set the x-ticks without '48'
    ax1.set_xticks(x_ticks)

    # Update the x-tick labels to match the values (replace 25 with the infinity symbol)
    x_labels = [str(int(val)) if val != 25 else r'$\infty$' for val in x_ticks]
    ax1.set_xticklabels(x_labels)

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE', color='red')  # Change to match right axis MAE
    ax2.plot(x, [m + 0.0005 for m in mae], 'rd-', label='MAE')  # Red diamonds with lines for MAE, adding a small gap
    ax2.tick_params(axis='y', labelcolor='red')  # Set ticks to red

    # Add a vertical dotted line at N_fft = 48
    #ax1.axvline(x=48, color='black', linestyle=':', linewidth=1)  # Black dotted line

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))  # Adjust this value to move left

    # Adjust layout for better fit
    fig.tight_layout()

    # Save the plot as a PDF if a filename is provided
    if plt_name:
        plt.savefig(plt_name, format="pdf")

    # Show the plot
    plt.show()

def plot_metrics_hoplen(x, rmse, mae,plt_name=False):
    # Replace 25 with infinity in the x-ticks, but keep 25 in the data
    x = [25 if val == 0 else val for val in x]  # Replace 0 with 25 in the data

    x_sorted, y1_sorted, y2_sorted = zip(*sorted(zip(x, rmse, mae)))

    # Convert back to lists (if needed)
    x = list(x_sorted)
    rmse = list(y1_sorted)
    mae = list(y2_sorted)

    fig, ax1 = plt.subplots()

    # Plot RMSE on the left y-axis
    ax1.set_xlabel(r'number of STFT context windows (p)')
    ax1.set_ylabel('RMSE', color='blue')  # Change to match left axis RMSE
    ax1.plot(x, rmse, 'bs-', label='RMSE')  # Blue squares with lines for RMSE
    ax1.tick_params(axis='y', labelcolor='blue')  # Set ticks to blue

    # Adjusting x-axis ticks to show powers of 2 explicitly
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Change tick labels, replace 25 with infinity symbol
    x_labels = [str(int(val)) for val in x]
    ax1.set_xticklabels(x_labels)

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE', color='red')  # Change to match right axis MAE
    ax2.plot(x, [m + 0.0005 for m in mae], 'rd-', label='MAE')  # Red diamonds with lines for MAE, adding a small gap
    ax2.tick_params(axis='y', labelcolor='red')  # Set ticks to red

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))  # Adjust this value to move left

    #plt.title('RMSE and MAE vs $N_{\mathrm{FFT}}$')
    fig.tight_layout()
    if plt_name:
        plt.savefig(plt_name, format="pdf")  # This saves the plot as a PDF file
    plt.show()
def plot_metrics_embed(x, rmse, mae,plt_name=False):
    x_sorted, y1_sorted, y2_sorted = zip(*sorted(zip(x, rmse, mae)))

    # Convert back to lists (if needed)
    x = list(x_sorted)
    rmse = list(y1_sorted)
    mae = list(y2_sorted)

    fig, ax1 = plt.subplots()

    # Plot RMSE on the left y-axis
    ax1.set_xlabel('Embed size (E)')  # Changing x-label to match the image
    ax1.set_ylabel('RMSE', color='blue')  # Change to match left axis RMSE
    ax1.plot(x, rmse, 'bs-', label='RMSE')  # Blue squares with lines for RMSE
    ax1.tick_params(axis='y', labelcolor='blue')  # Set ticks to blue
    ax1.set_xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2

    # Adjusting x-axis ticks to show powers of 2 explicitly
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE', color='red')  # Change to match right axis MAE
    ax2.plot(x, [m + 0.00005 for m in mae], 'rd-', label='MAE')  # Red diamonds with lines for MAE, adding a small gap
    ax2.tick_params(axis='y', labelcolor='red')  # Set ticks to red

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))  # Adjust this value to move left

    #plt.title('RMSE and MAE vs Embed Size')
    fig.tight_layout()
    if plt_name:
        plt.savefig(plt_name, format="pdf")  # This saves the plot as a PDF file
    plt.show()

def plot_metrics_TOPM(x, rmse, mae,plt_name=False):
    # Replace 25 with infinity in the x-ticks, but keep 25 in the data
    x = [25 if val == 0 else val for val in x]  # Replace 0 with 25 in the data

    x_sorted, y1_sorted, y2_sorted = zip(*sorted(zip(x, rmse, mae)))

    # Convert back to lists (if needed)
    x = list(x_sorted)
    rmse = list(y1_sorted)
    mae = list(y2_sorted)

    fig, ax1 = plt.subplots()

    # Plot RMSE on the left y-axis
    ax1.set_xlabel('M - Selected quantity of frequencies')  # Changing x-label to match the image
    ax1.set_ylabel('RMSE', color='blue')  # Change to match left axis RMSE
    ax1.plot(x, rmse, 'bs-', label='RMSE')  # Blue squares with lines for RMSE
    ax1.tick_params(axis='y', labelcolor='blue')  # Set ticks to blue

    # Adjusting x-axis ticks to show powers of 2 explicitly
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Change tick labels, replace 25 with infinity symbol
    x_labels = [r'$\infty$' if val == 25 else str(int(val)) for val in x]
    ax1.set_xticklabels(x_labels)

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE', color='red')  # Change to match right axis MAE
    ax2.plot(x, [m + 0.00005 for m in mae], 'rd-', label='MAE')  # Red diamonds with lines for MAE, adding a small gap
    ax2.tick_params(axis='y', labelcolor='red')  # Set ticks to red

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))  # Adjust this value to move left

    #plt.title('RMSE and MAE vs TopM')
    fig.tight_layout()
    if plt_name:
        plt.savefig(plt_name, format="pdf")  # This saves the plot as a PDF file
    plt.show()




###################### Embed sweep ###########################





#### ETT Embed 96 ###
embed = [64, 8, 512, 4, 256, 128, 1024, 2, 16, 32, 1]
mae = [0.058709, 0.059416, 0.058732, 0.060142, 0.058462, 0.058605, 0.060258, 0.060662, 0.059608, 0.058952, 0.076604]
rmse = [0.08793, 0.088403, 0.088283, 0.089133, 0.087797, 0.087827, 0.090156, 0.089649, 0.088709, 0.088195, 0.1109]

#### ETT Embed 336 ###
embed = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
mae = [0.099534, 0.06955, 0.068684, 0.069702, 0.068247, 0.068436, 0.068732, 0.068325, 0.071039, 0.070423, 0.069117]
rmse = [0.12898, 0.10041, 0.099666, 0.10051, 0.099464, 0.09967, 0.09996, 0.09967, 0.10219, 0.10181, 0.10086]


#Electricity Embed 96 ###
embed = [256, 128, 64, 32, 16, 8, 4, 2, 1]
rmse = [0.070391, 0.070217, 0.070332, 0.070455, 0.071001, 0.071359, 0.072018, 0.072559, 0.080329]
mae = [0.042141, 0.041923, 0.042019, 0.04209, 0.042547, 0.042977, 0.043541, 0.044044, 0.051989]


#Electricity Embed 336 ###
embed = [256, 128, 64, 32, 16, 8, 4, 2, 1]
rmse = [0.073761, 0.073378, 0.073247, 0.07357, 0.073776, 0.074111, 0.074623, 0.075103, 0.082553]
mae = [0.046482, 0.046067, 0.04593, 0.04629, 0.04649, 0.046836, 0.047267, 0.047707, 0.054957]


###########################NFFT experiment###########################
#ETT 96 NFFT
n_fft = [8, 12, 16, 24, 32, 48]
mae = [0.058609, 0.058444, 0.058459, 0.058624, 0.05859, 0.058431]
rmse = [0.087909, 0.08776, 0.087816, 0.087985, 0.087898, 0.087827]
#ETT 336 NFFT#
n_fft = [8, 12, 16, 24, 32, 48]
mae = [0.069549, 0.068457, 0.068326, 0.068028, 0.068113, 0.068034]
rmse = [0.10066, 0.099817, 0.09973, 0.099486, 0.099637, 0.099582]

#electricity 96 nfft
n_fft = [48, 32, 24, 16, 12, 12, 8, 8]
rmse = [0.070848, 0.070783, 0.070852, 0.070866, 0.070862, 0.070401, 0.070772, 0.070183]
mae = [0.042515, 0.042489, 0.042483, 0.042537, 0.042565, 0.042087, 0.042482, 0.042015]

#electricity 336 nfft
n_fft = [48, 32, 24, 16, 12, 8, 8]
rmse = [0.073724, 0.073738, 0.073729, 0.073712, 0.073752, 0.073702, 0.073293]
mae = [0.046448, 0.046442, 0.046438, 0.046434, 0.046476, 0.046485, 0.046027]

#################### M experiment ################################

#ETT 96 M ##
TopM = [1, 2, 4, 8, 12, 16, 20,25]
mae = [ 0.058423, 0.058248, 0.057973, 0.058187, 0.058163, 0.05819, 0.058205,0.058436]
rmse = [ 0.087794, 0.087634, 0.087198, 0.087312, 0.0873, 0.087329, 0.087356,0.087844]

#ETT 336 M ##
# Arrays for TopM, MAE, and RMSE
TopM = [ 1, 2, 4, 8, 12, 16, 20 , 25]
mae = [ 0.067991, 0.067796, 0.067423, 0.067489, 0.067558, 0.067578, 0.067536,0.067987]
rmse = [ 0.09943, 0.09919, 0.098622, 0.098662, 0.098874, 0.098877, 0.098716,0.099508]


#electricity 96 M#
TopM = [1, 2, 4, 8, 12, 16, 20, 25]
mae = [0.042028, 0.041882, 0.041852, 0.04193, 0.041997, 0.042067, 0.04201, 0.042024]
rmse = [0.070242, 0.069908, 0.069814, 0.069976, 0.070081, 0.070154, 0.07011, 0.070187]

#electricity 336 M#

TopM = [1, 2, 4, 8, 12, 16, 20, 25]
mae = [0.045967, 0.045771, 0.045825, 0.045934, 0.045912, 0.045974, 0.045986, 0.046013]
rmse = [0.073267, 0.07305, 0.073064, 0.073169, 0.073154, 0.073206, 0.073216, 0.073353]


#################### hop lenght experiment ################################

# ETT 96 hop lenght
hop_array = np.array([3, 4, 6, 8, 12, 16, 24, 32])
window_size = (96/hop_array +1).astype(int)
rmse = [0.087369, 0.087493, 0.08776, 0.087803, 0.08752, 0.087951, 0.087505, 0.087729]
mae = [0.05781, 0.057995, 0.058444, 0.05859, 0.058189, 0.058593, 0.058113, 0.058423]


# ETT 336 hop lenght
hop_array = np.array([3, 4, 6, 8, 12, 16, 24, 32])
window_size = (96/hop_array +1).astype(int)
rmse = [0.099814, 0.099666, 0.09992, 0.09967, 0.099622, 0.099482, 0.099812, 0.09953]
mae = [0.068411, 0.068424, 0.068528, 0.068325, 0.068244, 0.068034, 0.06851, 0.068138]


#Electricity 96 hop lenght
# Arrays for MAE, RMSE, and Hop Length
mae = [0.042421, 0.042378, 0.042512, 0.042564, 0.042538, 0.042597, 0.042502, 0.042009, 0.042405]
rmse = [0.07076, 0.070861, 0.070887, 0.071021, 0.071023, 0.070918, 0.070864, 0.070326, 0.070788]
hop_length = [32, 24, 16, 12, 8, 6, 4, 3, 3]
window_size = (96/hop_array +1).astype(int)


#Electricity 336 hop lenght
# Arrays for MAE, RMSE, and Hop Length
mae = [0.046442, 0.0466, 0.046525, 0.046514, 0.046503, 0.0465, 0.046663, 0.046361, 0.046581]
rmse = [0.073722, 0.073873, 0.073803, 0.0738, 0.073772, 0.073731, 0.074003, 0.073625, 0.073873]
hop_length = [32, 24, 16, 12, 8, 6, 4, 3, 3]
window_size = (96/hop_array +1).astype(int)





#seq len ETTh1 96

seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae = [0.060739, 0.061315, 0.059554, 0.05834, 0.058191, 0.058183, 0.05874, 0.061532]
rmse = [0.088263, 0.089049, 0.087443, 0.087, 0.087253, 0.087514, 0.08844, 0.092196]


#seq len ETTh1 192
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae = [0.067089, 0.063978, 0.064424, 0.062458, 0.063616, 0.06323, 0.065122, 0.068316]
rmse = [0.095692, 0.092631, 0.093329, 0.092007, 0.093693, 0.093913, 0.095686, 0.099432]

#seq len ETTh1 336
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae = [0.069953, 0.068544, 0.068324, 0.066703, 0.066633, 0.068237, 0.070693, 0.073796]
rmse = [0.098895, 0.097651, 0.097469, 0.096522, 0.097337, 0.099624, 0.10188, 0.10515]

#seq len ETTh1 720
# Arrays based on the new image data
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae = [0.075962, 0.074808, 0.076089, 0.075732, 0.075058, 0.076579, 0.080044, 0.082438]
rmse = [0.1034, 0.10232, 0.10391, 0.10344, 0.1028, 0.1046, 0.10789, 0.11069]


#seq len ETTm1 96
rmse = [0.075295, 0.075396, 0.075023, 0.074201, 0.075482, 0.079429, 0.092737, 0.10538]
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae4 = [0.048974, 0.049203, 0.048966, 0.048474, 0.049217, 0.052401, 0.060466, 0.068874]

#seq len ETTm1 192

rmse = [0.079775, 0.080316, 0.080223, 0.079562, 0.080109, 0.084289, 0.097592, 0.10909]
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae3 = [0.052274, 0.052434, 0.052753, 0.052403, 0.052773, 0.055694, 0.064905, 0.073547]

#seq len ETTm1 336
rmse = [0.083534, 0.084458, 0.083745, 0.083786, 0.084037, 0.08796, 0.10187, 0.11671]
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae1 = [0.055222, 0.05582, 0.05545, 0.055852, 0.05591, 0.059273, 0.068855, 0.082871]

#seq len ETTm1 720
rmse = [0.089925, 0.090161, 0.090331, 0.091175, 0.092103, 0.097163, 0.10622, 0.11841]
seqlen = [672, 576, 480, 288, 192, 96, 48, 24]
mae2 = [0.060162, 0.060326, 0.060368, 0.061533, 0.062611, 0.067467, 0.073655, 0.085195]


#seq len Traffic 96
rmse = [0.06863, 0.069012, 0.070909, 0.072764, 0.082115, 0.097723, 0.10285]
seqlen = [576, 480, 288, 192, 96, 48, 24]
mae = [0.031226, 0.031444, 0.032803, 0.033766, 0.040886, 0.050985, 0.054392]


#seq len Traffic 192
rmse = [0.0699, 0.07066, 0.072558, 0.074384, 0.081542, 0.092799, 0.099041]
seqlen = [576, 480, 288, 192, 96, 48, 24]
mae = [0.032318, 0.032842, 0.034019, 0.035079, 0.040272, 0.047011, 0.051299]



#seq len Traffic 336
# Arrays for MAE, RMSE, Hop Length, and Sequence Length
mae = [0.033573, 0.034004, 0.035139, 0.036297, 0.041112, 0.048028, 0.052798]
rmse = [0.070965, 0.071511, 0.073648, 0.075654, 0.08265, 0.09422, 0.10111]
seqlen = [576, 480, 288, 192, 96, 48, 24]


#seq len Traffic 720

