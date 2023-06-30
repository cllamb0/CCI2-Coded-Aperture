import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import os
import sys


direc = sys.argv[1]

final_data, files_list = {}, []
data_dir = 'Optimizations/{}/'.format(direc)
for f in os.listdir(data_dir):
    if f.endswith('.npy') and f.startswith('data_'):
        files_list.append([int(f.split('_')[1].split('-')[0]), f])
files_list.sort()

for i, fd in enumerate(files_list):
    files_list[i] = np.load(data_dir+fd[1], allow_pickle=True).item()

for key in files_list[0]:
    final_data[key] = np.concatenate(list(d[key] for d in files_list))
del files_list
print('----------------------------------------------------')
print('Water level is currently at: {}'.format(final_data['Water Levels'][-1]))
print('Best metric so far at: {}, {}/{} iterations in'.format(
    min(final_data['Metrics']), final_data['Iterations'][np.argmin(final_data['Metrics'])]+1, final_data['Iterations'][-1]+1))
print('Longest in a row without improvement: {}'.format(max(final_data['Stopping Iterations'])))
print('----------------------------------------------------')

plt.figure(dpi=300)
plt.plot(final_data['Iterations'], final_data['Metrics'], label='Metrics', alpha=0.5, color='grey')
plt.plot(final_data['Iterations'], final_data['Water Levels'], label='Water Level', linewidth=0.5, color='black')
plt.legend()
plt.xlabel('# Iterations')
plt.savefig(data_dir+'Plots/MIDRUN_Water_Level_Evolution.png',
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved midrun water level evolution plot')

plt.figure(dpi=300)
plt.plot(final_data['Iterations'], final_data['Stopping Iterations'], color='maroon')
plt.xlabel('# Iterations')
plt.ylabel('# Iterations without improvement')
plt.savefig(data_dir+'Plots/MIDRUN_stopping_iterations_Evolution_test.png',
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved midrun stopping iterations evolution')
