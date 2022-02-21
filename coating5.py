"""
Code created by Alexis MELOT (UdeS.nano) inspired by Pierre Yger's (Institut Vision) code.
Date : 20/02/2022
cmd : python coating5.py 

"""

import h5py
import scipy
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges

from circus.shared.probes import *


# Global Booleans
unwhiten = False
save_svg = True
use_monopole = True

# Global Variables
all_channels = np.arange(60)
label = ['100/0', '70/30', '50/50','40/60','30/70']
coating_name =['edot','edotG4_70_30','edotG4_50_50','edotG4_40_60','edotG4_30_70']
coating_conf = { 
    'edot' : [ 1,  6,  16, 21, 26, 31, 46, 56],
    'edotG4_70_30' : [ 2, 7, 12, 17, 22, 27, 32, 37, 47, 52, 57],
    'edotG4_50_50' : [ 3, 8, 13, 18, 28, 33, 38, 43, 53],
    'edotG4_40_60' : [ 5, 10, 15, 20, 25, 30, 35, 50],
    'edotG4_30_70' : [ 4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
    }
coating = {'g4_mea' : {'channels' : [1,2,3,4,5,6,7,8,9,10,12,13,
                            14,15,16,17,18,19,20,21,22,24,
                            25,26,27,28,29,30,31,32,33,34,
                            35,37,38,39,43,44,46,47,49,50,
                            52,53,56,57],
                      'mapping' : 'mapping_G4.txt',
                      'file' : 'G4_MEA.h5'}}
      
mcs_mapping = h5py.File('G4_MEA.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['Label'].astype('int')
mcs_factor = h5py.File('G4_MEA.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['ConversionFactor'][0] * 1e-6

def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um=250):
    # constant for initial guess and bounds
    initial_z = 20

    ind_max = np.argmax(wf_ptp)
    max_ptp = wf_ptp[ind_max]
    max_alpha = max_ptp * max_distance_um

    # initial guess is the center of mass
    com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
    x0 = np.zeros(4, dtype='float32')
    x0[:2] = com
    x0[2] = initial_z
    initial_alpha = np.sqrt(np.sum((com - local_contact_locations[ind_max, :])**2) + initial_z**2) * max_ptp
    x0[3] = initial_alpha

    # bounds depend on initial guess
    bounds = ([x0[0] - max_distance_um, x0[1] - max_distance_um, 1, 0],
              [x0[0] + max_distance_um,  x0[1] + max_distance_um, max_distance_um*10, max_alpha])

    return x0, bounds

def add_label(violin, label,violin_labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    violin_labels.append((mpatches.Patch(color=color), label))

def estimate_distance_error(vec, wf_ptp, local_contact_locations):
        # vec dims ar (x, y, z amplitude_factor)
        # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
        dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
        ptp_estimated = vec[3] / dist
        err = wf_ptp - ptp_estimated
        return err


#################
# Retrieve data #
#################

mapping = np.loadtxt(coating['g4_mea']['mapping'])
params = CircusParser(coating['g4_mea']['file'])

if unwhiten:
    params.write('data', 'suffix', '-raw')
    params.write('whitening', 'spatial', 'False')
else:
    params.write('data', 'suffix', '')
    params.write('whitening', 'spatial', 'True')

params = CircusParser(coating['g4_mea']['file'])
params.get_data_file()


mads = load_data(params, 'mads')
thresholds = load_data(params, 'thresholds')

if unwhiten:
    mads *= mcs_factor
    thresholds *= mcs_factor

# get the coated channels
coated_all = []
coated_channels = {'edot':[],
            'edotG4_70_30':[],'edotG4_50_50':[],
            'edotG4_40_60':[],'edotG4_30_70':[]}

for i in mapping[coating['g4_mea']['channels'],1]:
    coated_all += [np.where(mcs_mapping == i)[0]]
coated_all = np.array(coated_all).flatten()
for conf,chans in coating_conf.items():
    for i in mapping[chans,1]:
        coated_channels[conf] += [np.where(mcs_mapping == i)[0]]
    coated_channels[conf] = np.array(coated_channels[conf]).flatten()
        

nodes, edges = get_nodes_and_edges(params)

inv_nodes = np.zeros(60, dtype=np.int32)
inv_nodes[nodes] = np.arange(len(nodes))

non_coated_channels = all_channels[~np.in1d(all_channels, coated_all)]

########### Plots ###########

fig = plt.figure(figsize=(15,10), constrained_layout=True)
spec = fig.add_gridspec(3, 3)

#Noise Level
ax0 = fig.add_subplot(spec[0, 0])
plt_legend=[]
for i,name in enumerate(coating_name):
    add_label(ax0.violinplot([mads[inv_nodes[coated_channels[name]]]], [i], showmeans=True),label[i],plt_legend)
#add_label(ax[0, 0].violinplot([mads[inv_nodes[non_coated_channels]]], [i+1], showmeans=True),'not coated',plt_legend)

if unwhiten:
    ax0.set_ylabel('Noise level ($\mathrm{\mu}V)$')
else:
    ax0.set_ylabel('Noise level')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_xticks([])
#ax0.legend(*zip(*plt_legend), loc='lower right',prop={'size': 6})
#-----------------------
#Peak amplitude + SNR
res = load_data(params, 'mua')
coated_amplitudes = {
    'edot': np.zeros(0, dtype=np.float32),
    'edotG4_70_30': np.zeros(0, dtype=np.float32),
    'edotG4_50_50':np.zeros(0, dtype=np.float32),
    'edotG4_40_60':np.zeros(0, dtype=np.float32),
    'edotG4_30_70':np.zeros(0, dtype=np.float32)}
coated_snrs = {
    'edot': np.zeros(0, dtype=np.float32),
    'edotG4_70_30': np.zeros(0, dtype=np.float32),
    'edotG4_50_50':np.zeros(0, dtype=np.float32),
    'edotG4_40_60':np.zeros(0, dtype=np.float32),
    'edotG4_30_70':np.zeros(0, dtype=np.float32)}

for conf,chans in coating_conf.items():
    for a in inv_nodes[chans]:
        coated_amplitudes[conf] = np.concatenate((coated_amplitudes[conf], res['amplitudes']['elec_%d' %a]/thresholds[a]))
        coated_snrs[conf] = np.concatenate((coated_snrs[conf], [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))
       
if unwhiten: #TODO: need to be fixed
    coated_amplitudes *= mcs_factor
    coated_snrs *= mcs_factor

non_coated_amplitudes = np.zeros(0, dtype=np.float32)
non_coated_snrs = np.zeros(0, dtype=np.float32)

for a in inv_nodes[non_coated_channels]:
    non_coated_amplitudes = np.concatenate((non_coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))
    non_coated_snrs = np.concatenate((non_coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))

if unwhiten:#TODO: need to be fixed
    non_coated_amplitudes *= mcs_factor
    non_coated_snrs *= mcs_factor

## Peak amplitude
#plt_legend=[]
# for i,name in enumerate(coating_name):
#     add_label(ax[0, 1].violinplot([coated_amplitudes[conf]], [0], showmeans=True),label[i],plt_legend)
# add_label(ax[0, 1].violinplot(20*np.log10(non_coated_amplitudes), [i+1], showmeans=True),'not coated',plt_legend)
# ax[0, 1].set_ylabel('normalized peak amplitude')
# ax[0, 1].spines['top'].set_visible(False)
# ax[0, 1].spines['right'].set_visible(False)
# ax[0, 1].set_xticks([])
# ax[0, 1].legend(*zip(*plt_legend), loc='lower right',prop={'size': 6})

# SNR
ax1 = fig.add_subplot(spec[0, 1])
plt_legend=[]
for i,name in enumerate(coating_name):
    add_label(ax1.violinplot(20*np.log10(coated_snrs[name]), [i], showmeans=True),label[i],plt_legend)
#add_label(ax1.violinplot(20*np.log10(non_coated_snrs), [i+1], showmeans=True),'not coated',plt_legend)
ax1.set_ylabel('SNR (dB)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks([])
#ax1.legend(*zip(*plt_legend), loc='lower right',prop={'size': 6})
#--------------------
# Energy of templates
templates = load_data(params, 'templates')
nb_templates = templates.shape[1]//2
templates = templates[:,:nb_templates].toarray()
templates = templates.reshape(len(nodes), 31, nb_templates)

nodes, positions = get_nodes_and_positions(params)

norms = numpy.linalg.norm(templates, axis=1)
mask = norms != 0

plt_legend=[]
ax2 = fig.add_subplot(spec[0, 2])
for i,name in enumerate(coating_name):
    add_label(ax2.violinplot(norms[inv_nodes[coated_channels[name]],:][mask[inv_nodes[coated_channels[name]]]], [i], showmeans=True),label[i],plt_legend)
#add_label(ax2.violinplot(norms[inv_nodes[non_coated_channels],:][mask[inv_nodes[non_coated_channels]]], [i+1], showmeans=True),'not coated',plt_legend)
ax2.set_ylabel('Energy of templates')
ax2.set_yscale('log')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks([])
#ax2.legend(*zip(*plt_legend), loc='lower right',prop={'size': 6})
#--------------------
# Putative neuron positions MEA
coms = np.zeros((2, 0))
for idx in range(templates.shape[2]):
    wf = templates[:,:,idx].T
    wf_ptp = wf.ptp(axis=0)

    if use_monopole:
        x0, bounds = make_initial_guess_and_bounds(wf_ptp, positions[:,:2], 1000)
        args = (wf_ptp, positions[:,:2])
        com = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)
    else:
        com = np.sum(wf_ptp[:, np.newaxis] * positions[:,:2], axis=0) / np.sum(wf_ptp)
    coms = np.hstack((coms, com.x[:2,np.newaxis]))

ax_mea = fig.add_subplot(spec[1:3, :])
for i,name in enumerate(coating_name):
    ax_mea.scatter(positions[inv_nodes[coated_channels[name]], 0], positions[inv_nodes[coated_channels[name]], 1], c=f'C{i}',label=label[i])
ax_mea.scatter(positions[inv_nodes[non_coated_channels], 0], positions[inv_nodes[non_coated_channels], 1], c=f'C{i+1}')
ax_mea.scatter(coms[0], coms[1], c='k', alpha=0.5)
ax_mea.spines['top'].set_visible(False)
ax_mea.spines['right'].set_visible(False)
ax_mea.spines['left'].set_visible(False)
ax_mea.spines['bottom'].set_visible(False)
ax_mea.set_xticks([])
ax_mea.set_yticks([])
ax_mea.set_title('MEA layout')

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, 'lower left')

#--------------------
# Peak amplitude

# purity = load_data(params, 'purity')
# cNorm  = Normalize(vmin=0, vmax=1)
# my_cmap = plt.get_cmap('winter')
# scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

# results = load_data(params, 'results')
# for count, spikes in enumerate(results['spiketimes'].values()):
#     colorVal = scalarMap.to_rgba(purity[count])
#     ax[2, 1].scatter(spikes/params.data_file.sampling_rate, count*np.ones(len(spikes)), color=colorVal)
# ax[2, 1].set_xlabel('time (s)')
# ax[2, 1].set_xlim(50, 80)


# ax[2, 1].spines['top'].set_visible(False)
# ax[2, 1].spines['right'].set_visible(False)

# ax[1, 1].spines['top'].set_visible(False)
# ax[1, 1].spines['right'].set_visible(False)

# ax[1, 2].spines['top'].set_visible(False)
# ax[1, 2].spines['right'].set_visible(False)

# for count, e in enumerate(electrodes):
#     if e in coated_channels:
#         ax[1, 1].plot(templates[e,:,count]/thresholds[e], c='C0')
#     else:
#         ax[1, 2].plot(templates[e,:,count]/thresholds[e], c='C1')

# ax[1, 1].plot([0, 31], [-1, -1], 'k--')
# ax[1, 2].plot([0, 31], [-1, -1], 'k--')
# ymin = min(ax[1, 1].get_ylim()[0], ax[1, 2].get_ylim()[0])
# ymax = max(ax[1, 1].get_ylim()[1], ax[1, 2].get_ylim()[1])
# ax[1, 1].set_ylim(ymin, ymax)
# ax[1, 2].set_ylim(ymin, ymax)
# ax[1, 1].set_xlabel('timesteps')
# ax[1, 2].set_xlabel('timesteps')
# ax[1, 1].set_ylabel('normalized amplitude')
# ax[1, 2].set_ylabel('normalized amplitude')



if save_svg:
    fig_name = 'coating_comparison'
    if unwhiten:
        fig_name += "-raw"
    plt.savefig(fig_name + '.svg')

    plt.tight_layout()
    plt.close()
else:
    plt.tight_layout()
    plt.show()