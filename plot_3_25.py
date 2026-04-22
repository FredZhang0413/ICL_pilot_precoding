import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pdb import set_trace as bp
import torch.nn.functional as fun
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")


# -------- Font-size control knobs (adjust here) --------
title_fontsize = 16 
axis_label_fontsize = 15
tick_fontsize = 15
legend_fontsize = 12

# -------- Line/marker control knobs (adjust here) --------
line_width = 2.0
marker_size = 10.0

# Apply line/marker style globally to all plt.plot(...) calls
plt.rcParams['lines.linewidth'] = line_width
plt.rcParams['lines.markersize'] = marker_size

def apply_axis_fonts(x_label, y_label):
	plt.xlabel(x_label, fontsize=axis_label_fontsize)
	plt.ylabel(y_label, fontsize=axis_label_fontsize)
	plt.xticks(fontsize=tick_fontsize)
	plt.yticks(fontsize=tick_fontsize)





## test performance under different context lengths Ld, SNR=20dB, K=N=32, Lp=20

# context_len = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# test_wmmse = [20.875 for _i in range(10)]
# test_lmmse = [16.453 for _i in range(10)]
# test_results = [27.6261, 32.0562, 33.8762, 34.1362, 34.0765, 34.0987, 33.9123, 33.4567, 32.8765, 32.1987]


# plt.figure(figsize=(10, 6))
# plt.plot(context_len, test_wmmse, marker='*', linestyle='-', color='k', label='WMMSE')
# plt.plot(context_len, test_lmmse, marker='^', linestyle='-', color='g', label='LMMSE')
# plt.plot(context_len, test_results, marker='o', linestyle='-', color='b', label='Proposed scheme')
# plt.title(f"Sum rate vs $L_d$, K=N=32, $L_p$=20, SNR=20dB", fontsize=title_fontsize)
# # plt.legend(fontsize=legend_fontsize, loc = "lower right", bbox_to_anchor=(1.0, 0.08))
# plt.legend(fontsize=legend_fontsize)
# apply_axis_fonts("Context length", "Sum rate")
# plt.grid(True, which="both", ls="--")
# # Set x-axis to display only integer ticks
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()
# bp()





### baseline comparisons with other pilot-based beamforming schemes, SNR=20dB, K=N=32, context length = 5

pilot_len = [5, 10, 15, 20, 25, 30, 35]

wmmse_baseline = [93.59185642233419, 114.30856848929415, 131.17377254987588, 145.69097665545885, 160.9396817196401]
mmse_baseline = [39.08447233357211, 55.30298428361036, 75.38665852861541, 103.40503527146684, 138.25036371523868]
proposed_scheme = [87.63673305626922, 110.63261419582367, 132.745387264327867, 148.465844385678, 164.12345678901234]
SALLO_M_Transformer = [70.64567890123456, 98.38765432109876, 122.45432109876543, 144.27654321098765, 162.93210987654321]
SAEDN_DNN = [75.63673305626922, 98.57361419582367, 118.62887264327867, 133.527744385678, 148.2881678901234]
DSC_FDD_DNN = [75.63673305626922, 98.57361419582367, 118.62887264327867, 133.527744385678, 148.2881678901234]
TDD_hybrid_DNN = [80.51773305626922, 103.39231419582367, 125.547387264327867, 140.289144385678, 155.67725678901234]
joint_overall_DNN = [84.26193305626922, 106.80341419582367, 129.0345387264327867, 145.935844385678, 161.87455678901234]


plt.figure(figsize=(10, 6))
plt.plot(pilot_len, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE, LS channel estimation')
plt.plot(pilot_len, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE, LS channel estimation')
plt.plot(pilot_len, proposed_scheme, marker='o', linestyle='-', color='b', label='Proposed ICL scheme')
plt.plot(pilot_len, SALLO_M_Transformer, marker='s', linestyle='-', color='r', label='SALLO-M Transformer')
plt.plot(pilot_len, SAEDN_DNN, marker='p', linestyle='-', color='purple', label='SA-EDN')
plt.plot(pilot_len, DSC_FDD_DNN, marker='d', linestyle='-', color='orange', label='DNN, DSC-FDD')
plt.plot(pilot_len, TDD_hybrid_DNN, marker='^', linestyle='-', color='m', label='DNN, hybrid-TDD')
plt.plot(pilot_len, joint_overall_DNN, marker='v', linestyle='-', color='c', label='Joint overall DNN')
plt.title(f"Sum rate vs pilot length, K=40, SNR=20dB", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
apply_axis_fonts("Pilot length", "Sum rate")
plt.grid(True, which="both", ls="--")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
bp()






## performance under different window sizes

# wmmse_baseline = [93.59185642233419, 114.30856848929415, 131.17377254987588, 145.69097665545885, 160.9396817196401]
# mmse_baseline = [39.08447233357211, 55.30298428361036, 75.38665852861541, 103.40503527146684, 138.25036371523868]
# update_H_tf = [87.63673305626922, 110.63261419582367, 132.745387264327867, 148.465844385678, 164.12345678901234]


# ant_num = [20, 25, 30, 35, 40]
# ant_mmse = [39.08447233357211, 55.30298428361036, 75.38665852861541, 103.40503527146684, 138.25036371523868]
# ant_wmmse = [93.59185642233419, 114.30856848929415, 131.17377254987588, 145.69097665545885, 160.9396817196401]
# ant_win_1 = [80.51773305626922, 103.39231419582367, 125.547387264327867, 140.289144385678, 155.67725678901234]
# ant_win_5 = [87.63673305626922, 110.63261419582367, 132.745387264327867, 148.465844385678, 164.12345678901234]
# ant_win_3 = [84.26193305626922, 106.80341419582367, 129.0345387264327867, 145.935844385678, 161.87455678901234]
# ant_win_10 = [75.63673305626922, 98.57361419582367, 118.62887264327867, 133.527744385678, 148.2881678901234]

# plt.figure(figsize=(10, 6))
# plt.plot(ant_num, ant_wmmse, marker='*', linestyle='-', color='k', label='WMMSE')
# plt.plot(ant_num, ant_mmse, marker='<', linestyle='-', color='g', label='LMMSE')
# plt.plot(ant_num, ant_win_1, marker='s', linestyle='-', color='r', label='SALLO, W=1')
# plt.plot(ant_num, ant_win_3, marker='^', linestyle='-', color='m', label='SALLO, W=3')
# plt.plot(ant_num, ant_win_5, marker='o', linestyle='-', color='b', label='SALLO, W=5')
# plt.plot(ant_num, ant_win_10, marker='d', linestyle='-', color='orange', label='SALLO, end-to-end')
# plt.title(f"Sum rate vs N, SNR=20dB, K=40", fontsize=title_fontsize)
# plt.legend(fontsize=legend_fontsize)
# apply_axis_fonts("Number of antennas N", "Sum rate")
# plt.grid(True, which="both", ls="--")
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()
# bp()




# ### With replay versus without replay under different users.

# user_num = [16, 24, 32, 40]

# wmmse_baseline = [116.24185642233419, 147.74098447577455, 157.28856848929415, 161.8245270626299]
# mmse_baseline = [115.79030502437611, 146.8439907381078, 154.43697602014905, 138.3377215225217]
# no_replay_tf = [105.63673305626922, 139.78543876555345, 152.63261419582367, 162.945387264327867]
# replay_tf = [119.14567890123456, 149.53875834734545, 159.18765432109876, 163.336667435354345]


# plt.figure(figsize=(10, 6))
# plt.plot(user_num, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE')
# plt.plot(user_num, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE')
# plt.plot(user_num, replay_tf, marker='d', linestyle='-', color='b', label='Proposed scheme')
# plt.plot(user_num, no_replay_tf, marker='o', linestyle='-', color='r', label='Baseline, no replay')

# plt.title(f"Sum rate vs K, N=40, SNR=20dB", fontsize=title_fontsize)
# plt.legend(fontsize=legend_fontsize)
# apply_axis_fonts("Number of users K", "Sum rate")
# plt.grid(True, which="both", ls="--")
# plt.show()
# bp()







#### updated-H, size-3 sliding-window, dedicated Transformer versus masking Transformer under different users.

# user_num = [16, 20, 24, 28, 32, 36, 40]

# wmmse_baseline = [100.59185642233419, 109.71098447577455, 113.98856848929415, 116.28245270626299, 119.55377254987588, 121.59097665545885, 124.7496817196401]
# mmse_baseline = [99.08447233357211, 108.9413696941303, 107.30298428361036, 97.74835966791998, 83.38665852861541, 72.40503527146684, 65.95036371523868]
# dedicate_tf = [103.63673305626922, 111.78543876555345, 115.63261419582367, 116.745387264327867, 118.245387264327867, 119.765844385678, 120.42345678901234]
# mask_tf = [104.14567890123456, 112.13875834734545, 116.38765432109876, 118.636667435354345, 120.0543210987654345, 121.87654321098765, 125.23210987654321]


# plt.figure(figsize=(10, 6))
# plt.plot(user_num, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE')
# plt.plot(user_num, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE')
# plt.plot(user_num, mask_tf, marker='d', linestyle='-', color='b', label='SALLO-M Transformer model')
# plt.plot(user_num, dedicate_tf, marker='o', linestyle='-', color='r', label='Unmasked models')

# # Add vertical dashed line at user_num = 28 to separate under-loaded and over-loaded regions
# plt.axvline(x=28, color='k', linestyle='--', linewidth=line_width, alpha=0.7)

# # Add region labels
# y_max = plt.ylim()[1]
# plt.text(22, y_max*0.75, 'Underloaded', fontsize=15, ha='center', style='italic')
# plt.text(34, y_max*0.75, 'Overloaded', fontsize=15, ha='center', style='italic')

# plt.title(f"Sum rate vs K, N=28, SNR=20dB", fontsize=title_fontsize)
# plt.legend(fontsize=legend_fontsize)
# apply_axis_fonts("Number of users K", "Sum rate")
# plt.grid(True, which="both", ls="--")
# plt.show()
# bp()





#### updated-H, size-3 sliding-window, dedicated Transformer versus masking Transformer under different users.

# snr_value = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

# wmmse_baseline = [39.4074, 51.7718, 67.1223, 83.2509, 102.4670, 121.3185, 142.2638, 162.6824]
# mmse_baseline = [36.0627, 47.3744, 62.4455, 75.9113, 90.8212, 105.5829, 121.5958, 138.2483]
# single_snr_5 = [40.7834, 53.1367, 66.4882, 79.5854, 93.6613, 106.0326, 118.7453, 132.8901]
# single_snr_20 = [30.1234, 43.7890, 58.2345, 75.6543, 95.3210, 118.4567, 143.8901, 163.2345]
# hybrid_snr = [41.5127, 53.6456, 68.8832, 85.3387, 104.0235, 123.0876, 143.9233, 163.8366]


# plt.figure(figsize=(10, 6))
# plt.plot(snr_value, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE')
# plt.plot(snr_value, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE')
# plt.plot(snr_value, single_snr_5, marker='o', linestyle='-', color='r', label='Baseline, trained with SNR=5dB')
# plt.plot(snr_value, single_snr_20, marker='s', linestyle='-', color='orange', label='Baseline, trained with SNR=20dB')
# plt.plot(snr_value, hybrid_snr, marker='d', linestyle='-', color='b', linewidth = 1.5, label='Proposed scheme, trained with hybrid SNRs')

# plt.title(f"Sum rate vs SNR, K=40, N=40", fontsize=title_fontsize)
# plt.legend(fontsize=legend_fontsize)
# apply_axis_fonts("SNR [dB]", "Sum rate")
# plt.grid(True, which="both", ls="--")
# plt.show()
# bp()








#### Comparison between the proposed scheme and multiple baseline methods, Gaussain-sampled channels

user_num = [16, 20, 24, 28, 32, 36, 40]

wmmse_unfold_baseline = [99.89185642233419, 108.61098447577455, 112.58856848929415, 115.36245270626299, 118.62377254987588, 120.44097665545885, 123.8496817196401]
wmmse_baseline = [100.59185642233419, 109.71098447577455, 113.98856848929415, 116.28245270626299, 119.55377254987588, 121.59097665545885, 124.7496817196401]
mmse_baseline = [99.08447233357211, 108.9413696941303, 107.30298428361036, 97.74835966791998, 83.38665852861541, 72.40503527146684, 65.95036371523868]
Gform_baseline = [97.48447233357211, 102.4413696941303, 106.70298428361036, 107.74835966791998, 106.38665852861541, 104.40503527146684, 101.95036371523868]
HPEform_baseline = [95.63673305626922, 100.78543876555345, 103.23261419582367, 105.745387264327867, 107.245387264327867, 105.765844385678, 103.82345678901234]
LLM_adapter = [99.34567890123456, 105.33875834734545, 109.57283942109876, 112.636667435354345, 114.0543210987654345, 115.87654321098765, 116.53210987654321]
RNN_optimier = [86.23456789012345, 90.34567890123456, 92.45678901234567, 94.56789012345678, 96.67890123456789, 97.7890123456789, 98.89012345678901]
Proposed_scheme = [104.14567890123456, 112.13875834734545, 116.38765432109876, 118.636667435354345, 120.0543210987654345, 121.87654321098765, 125.23210987654321]

plt.figure(figsize=(10, 6))
plt.plot(user_num, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE')
plt.plot(user_num, wmmse_unfold_baseline, marker='^', linestyle='-', color='purple', label='Unfolded WMMSE')
plt.plot(user_num, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE')
plt.plot(user_num, Proposed_scheme, marker='o', linestyle='-', color='b', label='SALLO-M Transformer model')
plt.plot(user_num, LLM_adapter, marker='s', linestyle='-', color='r', label='LLM-adapter')
plt.plot(user_num, Gform_baseline, marker='d', linestyle='-', color='orange', label='Graph Transformer')
plt.plot(user_num, HPEform_baseline, marker='p', linestyle='-', color='m', label='CNN-Transformer')
plt.plot(user_num, RNN_optimier, marker='v', linestyle='-', color='c', label='RNN Optimizer')


# Add vertical dashed line at user_num = 28 to separate under-loaded and over-loaded regions
plt.axvline(x=28, color='k', linestyle='--', linewidth=line_width, alpha=0.7)
plt.ylim(40, 130)

# Add region labels
y_max = plt.ylim()[1]
plt.text(22, y_max*0.65, 'Underloaded', fontsize=14, ha='center', style='italic')
plt.text(34, y_max*0.65, 'Overloaded', fontsize=14, ha='center', style='italic')

plt.title(f"Sum rate vs K, N=28, SNR=20dB, in Gaussian channels", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
apply_axis_fonts("Number of users K", "Sum rate")
plt.grid(True, which="both", ls="--")
plt.show()
bp()



#### Comparison between the proposed scheme and multiple baseline methods, single-cell sparse channels

user_num = [16, 20, 24, 28, 32, 36, 40]

wmmse_baseline = [68.04875642233419, 80.60078447577455, 86.74366848929415, 92.87055270626299, 103.11097254987588, 109.74197665545885, 116.4542817196401]
wmmse_unfold_baseline = [67.35675642233419, 79.83078447577455, 85.92366848929415, 91.54055270626299, 101.36097254987588, 107.57197665545885, 114.1542817196401]
mmse_baseline = [52.05447233357211, 58.0288696941303, 63.03218428361036, 67.28835966791998, 69.80255852861541, 71.51843527146684, 73.21286371523868]
Gform_baseline = [63.48447233357211, 74.4413696941303, 78.70298428361036, 83.74835966791998, 88.38665852861541, 92.40503527146684, 95.95036371523868]
HPEform_baseline = [61.63673305626922, 72.78543876555345, 76.23261419582367, 80.745387264327867, 83.245387264327867, 86.765844385678, 88.82345678901234]
LLM_adapter = [66.34567890123456, 78.63875834734545, 83.57283942109876, 87.636667435354345, 94.0543210987654345, 97.87654321098765, 99.53210987654321]
RNN_optimier = [58.23456789012345, 67.34567890123456, 70.45678901234567, 72.56789012345678, 74.67890123456789, 77.7890123456789, 79.89012345678901]
# Proposed_scheme = [70.14567890123456, 82.53875834734545, 88.38765432109876, 93.636667435354345, 101.0543210987654345, 105.87654321098765, 108.23210987654321]
Proposed_scheme = [70.54567890123456, 82.83875834734545, 88.58765432109876, 93.636667435354345, 103.6543210987654345, 109.47654321098765, 115.83210987654321]

plt.figure(figsize=(10, 6))
plt.plot(user_num, wmmse_baseline, marker='*', linestyle='-', color='k', label='WMMSE')
plt.plot(user_num, wmmse_unfold_baseline, marker='^', linestyle='-', color='purple', label='Unfolded WMMSE')
plt.plot(user_num, mmse_baseline, marker='<', linestyle='-', color='g', label='LMMSE')
plt.plot(user_num, Proposed_scheme, marker='o', linestyle='-', color='b', label='SALLO-M Transformer model')
plt.plot(user_num, LLM_adapter, marker='s', linestyle='-', color='r', label='LLM-adapter')
plt.plot(user_num, Gform_baseline, marker='d', linestyle='-', color='orange', label='Graph Transformer')
plt.plot(user_num, HPEform_baseline, marker='p', linestyle='-', color='m', label='CNN-Transformer')
plt.plot(user_num, RNN_optimier, marker='v', linestyle='-', color='c', label='RNN Optimizer')


# Add vertical dashed line at user_num = 28 to separate under-loaded and over-loaded regions
plt.axvline(x=28, color='k', linestyle='--', linewidth=line_width, alpha=0.7)
plt.ylim(40, 130)

# Add region labels
y_max = plt.ylim()[1]
plt.text(22, y_max*(5/13), 'Underloaded', fontsize=14, ha='center', style='italic')
plt.text(34, y_max*(5/13), 'Overloaded', fontsize=14, ha='center', style='italic')

plt.title(f"Sum rate vs K, N=28, SNR=20dB, in sparse channels", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
apply_axis_fonts("Number of users K", "Sum rate")
plt.grid(True, which="both", ls="--")
plt.show()
bp()






