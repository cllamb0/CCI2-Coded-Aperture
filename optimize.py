import numpy as np
import os
import argparse
import time
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
try:
    from distinctipy import distinctipy
except:
    pass
import math
import scipy.signal as signal
from matplotlib.colors import from_levels_and_colors
from tetris_blocks import tetris_blocks, blokus_blocks, total_blocks, block_colors, order_3d, text_place

class OptimizerClass:
    def __init__(self,
                 method        = 'GreatDeluge',
                 stopItr       = 150,
                 data_fname    = None,
                 data_dir      = None,
                 plots_dir     = None,
                 decay_rate    = 0.01,
                 cont          = False,
                 seed          = int(time.time()),
                 save_ev       = 1000,
                 mask_size     = 46,
                 detector_size = 37,
                 magnification = 4,
                 open_frac     = 0.5,
                 hole_limit    = 80,
                 balanced      = True,
                 mask          = None,
                 verbose       = False,
                 section_offset= 0,
                 sectioning    = True,
                 group_indices = None,
                 transmission  = 0.05,
                 corr_weight   = 1,
                 sens_weight   = 1,
                 flat_weight   = 1,
                 dual_corr     = False,
                 initialize    = True,
                 tetrisify     = True,
                 frac_tetris   = 0.5,
                 tetris_blocks = tetris_blocks,
                 blokus_blocks = blokus_blocks,
                 total_blocks  = total_blocks,
                 block_colors  = block_colors,
                 static_blocks = False,
                 label_blocks  = False,
                 bw_tetris     = False):
        """
        Initializer for all class variables and intial
        """

        assert method in ['GreatDeluge']

        if verbose:
            print('Initializing optimization configuration')

        self.method         = method
        self.stopItr        = stopItr
        self.data_fname     = data_fname
        self.decay_rate     = decay_rate
        self.cont           = cont
        self.seed           = seed
        self.save_ev        = save_ev
        self.mask_size      = mask_size
        self.detector_size  = detector_size
        self.magnification  = magnification
        self.sample_size    = math.floor(self.detector_size/self.magnification)
        self.sens_sample    = np.ones((self.sample_size, self.sample_size))
        self.open_frac      = open_frac
        self.hole_limit     = hole_limit
        self.balanced       = balanced
        self.mask           = mask
        self.verbose        = verbose
        self.section_offset = section_offset
        self.section_size   = self.sample_size - self.section_offset
        self.sectioning     = sectioning
        self.group_indices  = group_indices
        self.transmission   = transmission
        self.corr_weight    = corr_weight
        self.sens_weight    = sens_weight
        self.flat_weight    = flat_weight
        self.dual_corr      = dual_corr
        self.initialize     = initialize
        self.tetrisify      = tetrisify # For now, tetrisifying the mask will ignore sectioning
        self.frac_tetris    = frac_tetris
        self.tetris_blocks  = tetris_blocks
        self.blokus_blocks  = blokus_blocks
        self.block_colors   = block_colors
        self.total_blocks   = total_blocks
        self.tetris_needed  = 0
        self.blokus_needed  = 0
        self.static_blocks  = static_blocks
        self.label_blocks   = label_blocks
        self.bw_tetris      = bw_tetris

        if not self.tetrisify:
            self.frac_tetris = 0

        if self.initialize:
            try:
                os.mkdir('Optimizations')
            except:
                pass

            if data_dir is None:
                if not self.tetrisify:
                    ft = 0
                else:
                    ft = self.frac_tetris
                data_dir = 'Optimizations/GD_ms_{}-of_{}-mag_{}-seed_{}-hl_{}-cw_{}-sw_{}-ft_{}/'.format(
                            self.mask_size, str(self.open_frac).split('.')[1], self.magnification, self.seed, self.hole_limit, self.corr_weight, self.sens_weight, ft)
                try:
                    os.mkdir(data_dir)
                    if self.verbose:
                        print('Data directory created')
                except:
                    pass
                self.data_dir = data_dir

            if plots_dir is None:
                plots_dir = data_dir+'Plots/'
                try:
                    os.mkdir(plots_dir)
                    if self.verbose:
                        print('Plots directory created')
                except:
                    pass
                self.plots_dir = plots_dir

            self.cont = len([f for f in os.listdir(self.data_dir) if f.startswith('INCOMPLETE')]) != 0
        else:
            self.cont = False

        np.random.seed(self.seed)
        if self.cont:
            print('\033[36m\033[1mContinuing from previous optimization state\033[0m')

            self.CreateMask(save=False)

            self.LoadRandomState()

            self.init_mask = np.loadtxt(self.data_dir+'Initial_mask.txt')

            self.mask = np.loadtxt(self.data_dir+'INCOMPLETE_mask.txt')
            self.min_mask = np.loadtxt(self.data_dir+'INCOMPLETE_min_mask.txt')
            self.last_imp_mask = np.loadtxt(self.data_dir+'INCOMPLETE_last_imp_mask.txt')

            os.remove(self.data_dir+'INCOMPLETE_mask.txt')
            os.remove(self.data_dir+'INCOMPLETE_min_mask.txt')
            os.remove(self.data_dir+'INCOMPLETE_last_imp_mask.txt')

            if self.tetrisify:
                self.block_assigns = np.load(self.data_dir+'INCOMPLETE_block_assigns.npy', allow_pickle=True).item()
                self.min_block_assigns = np.load(self.data_dir+'INCOMPLETE_min_block_assigns.npy', allow_pickle=True).item()
                self.last_imp_block_assigns = np.load(self.data_dir+'INCOMPLETE_last_imp_block_assigns.npy', allow_pickle=True).item()

                os.remove(self.data_dir+'INCOMPLETE_block_assigns.npy')
                os.remove(self.data_dir+'INCOMPLETE_min_block_assigns.npy')
                os.remove(self.data_dir+'INCOMPLETE_last_imp_block_assigns.npy')
        else:
            if self.initialize:
                self.CreateMask()
                self.init_mask = self.mask.copy()
                if not self.tetrisify:
                    self.VisualizeMask(self.init_mask, 'Initial')
                else:
                    self.VisualizeTetrisMask(self.init_mask, 'Initial', label=self.label_blocks)
                    if self.bw_tetris:
                        self.VisualizeMask(self.init_mask, 'Initial')
                    np.save(self.data_dir+'Initial_block_assigns.npy', self.block_assigns)
                    # self.VisualizeTetrisMask(self.init_mask, 'Initial_labelled', True)
            else:
                self.CreateMask(save=False)


        self.corr_size = signal.correlate2d(self.mask, self.mask[0:self.sample_size, 0:self.sample_size], mode='valid').shape[0]

    def CreateMask(self, save=True):
        """
        Creation of the first mask for the "optimization" process
        """
        temp_flip = False
        if not save and self.balanced:
            self.balanced = False
            temp_flip = True

        hole_checking = True
        while hole_checking:
            if self.tetrisify:
                mask = np.zeros((self.mask_size, self.mask_size))
                block_assigns = {0: 0}

                if self.frac_tetris == 1:
                    self.tetris_needed = round(((1-self.open_frac)*self.mask_size**2)/4)
                elif self.frac_tetris == 0:
                    self.blokus_needed = round(((1-self.open_frac)*self.mask_size**2)/5)
                else:
                    self.tetris_needed, self.blokus_needed = np.linalg.solve(np.array([[4, 5], [1/self.frac_tetris, -1/(1-self.frac_tetris)]]),
                                                               np.array([[(1-self.open_frac)*self.mask_size**2], [0]])).reshape(2,).astype(int)
                    # Adjusting the tetris fraction by a minor amount to achieve near zero filling error
                    element_error = round(((1-self.open_frac)*self.mask_size**2) - (4*self.tetris_needed + 5*self.blokus_needed))
                    self.blokus_needed += element_error % 4
                    self.tetris_needed += (element_error // 4) - (element_error % 4)

                    self.frac_tetris = self.tetris_needed / (self.tetris_needed + self.blokus_needed)

                # Placing tetris blocks into place
                for block_num in range(self.tetris_needed):
                    block_needed = True
                    while block_needed:
                        chosen_block, block_ind = self.choose_block(4, True)
                        c_ind = np.random.randint((np.array(mask.shape) - np.array(chosen_block.shape) + 1))
                        if self.valid_block_placement(mask, chosen_block, c_ind):
                            for row, col in np.ndindex(chosen_block.shape):
                                if chosen_block[row, col] == 1:
                                    mask[c_ind[0]+row, c_ind[1]+col] = block_num+1
                                    block_assigns[block_num+1] = block_ind+1
                            block_needed = False

                # Placing blokus blocks into place
                for block_num_blok in range(self.blokus_needed):
                    block_needed = True
                    while block_needed:
                        chosen_block, block_ind = self.choose_block(5, True)
                        c_ind = np.random.randint((np.array(mask.shape) - np.array(chosen_block.shape) + 1))
                        if self.valid_block_placement(mask, chosen_block, c_ind):
                            for row, col in np.ndindex(chosen_block.shape):
                                if chosen_block[row, col] == 1:
                                    mask[c_ind[0]+row, c_ind[1]+col] = block_num+block_num_blok+2
                                    block_assigns[block_num+block_num_blok+2] = block_ind+8
                            block_needed = False
            else:
                if not self.sectioning:
                    mask = np.zeros(self.mask_size**2)
                    num_open = math.ceil(self.open_frac * (self.mask_size**2))

                    mask[np.random.choice(mask.size, num_open, replace=False)] = 1

                    mask = mask.reshape(self.mask_size, self.mask_size)
                else:
                    num_open = math.ceil(self.open_frac * (self.section_size**2))

                    group_indices = []
                    for col_section in range(math.floor(self.mask_size/self.section_size)):
                        temp_column = np.zeros(self.section_size**2)
                        temp_column[np.random.choice(temp_column.size, num_open, replace=False)] = 1
                        temp_column = temp_column.reshape(self.section_size, self.section_size)
                        group_indices.append([[0, col_section*self.section_size],
                                                  [self.section_size, (col_section+1)*self.section_size]])
                        for row_section in range(math.floor(self.mask_size/self.section_size)-1):
                            temp_mask = np.zeros(self.section_size**2)
                            temp_mask[np.random.choice(temp_mask.size, num_open, replace=False)] = 1
                            temp_mask = temp_mask.reshape(self.section_size, self.section_size)
                            temp_column = np.concatenate((temp_column, temp_mask), axis=0)
                            group_indices.append([[(row_section+1)*self.section_size, (col_section)*self.section_size],
                                                  [(row_section+2)*self.section_size, (col_section+1)*self.section_size]])
                        if col_section == 0:
                            mask = np.copy(temp_column)
                        else:
                            mask = np.concatenate((mask, temp_column), axis=1)

                    if self.mask_size % self.section_size != 0:
                        fix_column = np.zeros((self.mask_size%self.section_size)*(mask.shape[0]))
                        col_fix_open = math.floor(self.open_frac * (fix_column.size))
                        fix_column[np.random.choice(fix_column.size, col_fix_open, replace=False)] = 1
                        fix_column = fix_column.reshape((mask.shape[0]), (self.mask_size%self.section_size))

                        mask = np.concatenate((mask, fix_column), axis=1)
                        group_indices.append([[0, mask.shape[0]], list(mask.shape)])

                        fix_row = np.zeros((self.mask_size%self.section_size)*(mask.shape[1]))
                        row_fix_open = math.floor(self.open_frac * (fix_row.size))
                        fix_row[np.random.choice(fix_row.size, row_fix_open, replace=False)] = 1
                        fix_row = fix_row.reshape((self.mask_size%self.section_size), (mask.shape[1]))

                        sms = mask.shape[0]
                        mask = np.concatenate((mask, fix_row), axis=0)
                        group_indices.append([[sms, 0], list(mask.shape)])

                    self.group_indices = group_indices

            if self.balanced:
                if self.hole_size_checking(mask):
                    hole_checking = False
            else:
                hole_checking = False

                if temp_flip:
                    self.balanced = True

        self.mask = mask.copy()
        if self.tetrisify:
            self.block_assigns = copy.deepcopy(block_assigns)

        if save:
            np.savetxt(self.data_dir+'Initial_mask.txt', mask, fmt='%i')

    def VisualizeMask(self, plot_mask, sname):
        """
        Plotter for way of visualizing a mask in a clean looking way
        """
        if self.tetrisify:
            plot_mask = np.where(plot_mask == 0, 1, 0)
        else:
            plot_mask = plot_mask.reshape(self.mask_size, self.mask_size)

        cmap = mpl.cm.get_cmap("binary").copy()
        cmap.set_bad(color='black')

        fig = plt.figure(dpi=400, facecolor='white')
        ax = fig.gca()
        ax.axis("off")

        patches = []
        for m, row in enumerate(plot_mask):
            for n, col in enumerate(row):
                square = RegularPolygon((n, -m), numVertices=4, radius=0.67,
                                    orientation=np.radians(45), edgecolor='white', linewidth=0.75, facecolor='white')
                patches.append(square)

        collection = PatchCollection(patches, match_original=True, cmap=cmap)
        collection.set_array(np.ma.masked_where(plot_mask.flatten() <= 0, plot_mask.flatten()))
        ax.add_collection(collection)

        plt.plot(np.arange(self.mask_size+1)-0.5, np.ones(self.mask_size+1)*0.5, color='black')
        plt.plot(np.arange(self.mask_size+1)-0.5, -np.ones(self.mask_size+1)*(self.mask_size-0.5), color='black')
        plt.plot(np.ones(self.mask_size+1)-1.5, -np.arange(self.mask_size+1)+0.5, color='black')
        plt.plot(np.ones(self.mask_size+1)+self.mask_size-1.5, -np.arange(self.mask_size+1)+0.5, color='black')

        ax.set_aspect('equal')
        ax.autoscale_view()

        plt.title('Mask Size: {} | Open Fraction: {}'.format(self.mask_size, self.open_frac) +
                  '\nRandom Seed: {}'.format(self.seed), fontsize='small', y = 0.95)

        plt.savefig(self.plots_dir+'{}_coded_mask.png'.format(sname), bbox_inches='tight')

        plt.close()

        print('Saved {} aperture mask image'.format(sname))

    def VisualizeTetrisMask(self, plot_mask, sname, label=False, _block_assigns=None):
        """
        Plotter for a tetrisified mask
        Assumes the defined block_assigns exists
        """
        if _block_assigns is None:
            _block_assigns = self.block_assigns

        mask_3d = np.ndarray(shape=(plot_mask.shape[0], plot_mask.shape[1], 3), dtype=int)
        for i in range(plot_mask.shape[0]):
            for j in range(plot_mask.shape[1]):
                mask_3d[i][j] = self.block_colors[_block_assigns[plot_mask[i][j]]]

        plt.figure(figsize=(7,7), dpi=200, facecolor='white')
        plt.imshow(mask_3d)
        ax = plt.gca()

        if label:
            for i in range(0, plot_mask.shape[0]):
                for j in range(0, plot_mask.shape[1]):
                    c = _block_assigns[plot_mask[j,i]]
                    if c == 0:
                        continue
                    ax.text(i, j, str(int(c)), va='center', ha='center', fontsize=(46*5)/self.mask_size)
        plt.title('{} blocks of 4 elements\n{} blocks of 5 elements'.format(
            self.tetris_needed, self.blokus_needed), y=1.01)

        plt.savefig(self.plots_dir+'{}_tetris_pieces_mask.png'.format(sname), bbox_inches='tight')

        plt.close()

    def createOrderSheet(self, _block_assigns):
        block_quan = {b:list(_block_assigns.values()).count(b) for b in range(1,len(self.total_blocks)+1)}

        # Summing mirror blocks
        block_mirrors = ([5, 7], [12, 13], [14, 15], [16, 17], [19, 20], [22, 23], [24, 25])
        for bm in block_mirrors:
            block_quan[bm[0]] += block_quan[bm[1]]

        plt.figure(figsize=(12,10), facecolor='white', dpi=200)
        ax = plt.gca()
        plt.imshow(order_3d)
        ax.axis('off')
        for key in text_place:
            ax.text(text_place[key][0], text_place[key][1], 'x{}'.format(block_quan[key]), va='center', ha='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.75'))
        plt.title('Quantity of each block to order', fontsize=18, weight='bold')

        plt.savefig(self.plots_dir+'tetris_order_sheet.png', bbox_inches='tight')

        plt.close()

    def VisualizeWaterLevel(self):
        final_data = np.load(self.data_dir+'final_data.npy', allow_pickle=True).item()

        plt.figure(dpi=400, facecolor='white')

        plt.plot(final_data['Iterations'], final_data['Metrics'], label='Metrics', alpha=0.5, color='grey')
        plt.plot(final_data['Iterations'], final_data['Water Levels'], label='Water Level', linewidth=0.5, color='black')
        plt.legend()
        plt.xlabel('# Iterations')
        plt.savefig(self.plots_dir+'Water_Level_Evolution.png', bbox_inches='tight')
        plt.close()

        print('Saved the final water level evolution plot')

    def VisualizeMetricsEvolution(self):
        final_data = np.load(self.data_dir+'final_data.npy', allow_pickle=True).item()

        fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True, dpi=300, figsize=[10,7])

        ax1.plot(final_data['Iterations'], final_data['Q Metrics'], color='red', label='Cross Correlation')
        ax1.legend(loc=1, fontsize=9)
        ax2.plot(final_data['Iterations'], final_data['Sens Metrics'], color='green', label='Sensitivity')
        ax2.legend(loc=1, fontsize=9)
        ax3.plot(final_data['Iterations'], final_data['Flat Metrics'], color='blue', label='Flat Diff')
        ax3.legend(loc=1, fontsize=9)
        ax3.set_xlabel('# of Iterations')
        fig.tight_layout()
        plt.savefig(self.plots_dir+'Metrics_Evolution.png',
                    bbox_inches='tight', facecolor='white')
        plt.close()

        print('Saved the final metrics evolution plot')

    def SaveRandomState(self):
        """
        Saves the current random state to files to be loaded later upon continuation
        """
        state = np.random.get_state()

        simple = [state[0]] + list(state[2:])
        complex = state[1]

        np.save(self.data_dir+'rs_simple.npy', simple)
        np.save(self.data_dir+'rs_complex.npy', complex)

        if self.verbose:
            print('Saved random state')

    def LoadRandomState(self):
        """
        Loads the previously saved random state for the PRNG
        """
        simple = np.load(self.data_dir+'rs_simple.npy')
        complex = np.load(self.data_dir+'rs_complex.npy')

        state = (simple[0], complex, int(simple[1]), int(simple[2]), float(simple[3]))

        np.random.set_state(state)

        os.remove(self.data_dir+'rs_simple.npy')
        os.remove(self.data_dir+'rs_complex.npy')

        if self.verbose:
            print('Loaded random state')

    def CalculateMetric(self, mask, init_metrics=None, return_F=False):
        # Cross correlation metric
        # init_metrics in the form: [init_corr, init_sens]
        if self.tetrisify:
            # Temporarily converting a tetris mask to the standard format
            mask = np.where(mask == 0, 1, 0)

        if self.corr_weight != 0:
            F_matrix = np.array([(signal.correlate2d(mask, mask[m:m+self.sample_size, n:n+self.sample_size], mode='valid')/(self.sample_size**2)).reshape(self.corr_size**2, ) for m in range(0, self.mask_size-self.sample_size+1) for n in range(0, self.mask_size-self.sample_size+1)])
            if return_F:
                return F_matrix
            if self.flat_weight != 0:
                flat_diff = np.max(np.array([np.max(np.abs(np.delete(F_matrix[ind], ind) - (self.open_frac**2))) for ind in range(F_matrix.shape[0])]))
            else:
                flat_diff = 1
            Q1 = (1/self.sample_size)*np.sum((np.diag(F_matrix)-self.open_frac)**4)
            np.fill_diagonal(F_matrix, self.open_frac**2)
            Q2 = (1/(self.sample_size**2-self.sample_size))*np.sum((F_matrix-self.open_frac**2)**4)
            Q = Q1 + Q2
        else:
            Q = 1

        # "Sensitivity" metric
        if self.sens_weight != 0:
            sens_mask = np.copy(mask)
            sens_mask[sens_mask == 0] = self.transmission
            sensitivity_matrix = signal.correlate2d(sens_mask, self.sens_sample, mode='valid')
        else:
            sensitivity_matrix = self.sens_sample
        sens_Q = np.var(sensitivity_matrix)

        if init_metrics is None:
            return Q, sens_Q, flat_diff
        else:
            return self.corr_weight*(Q/init_metrics[0]), self.sens_weight*(sens_Q/init_metrics[1]), self.flat_weight*(flat_diff/init_metrics[2])

    def hole_size_checking(self, mask, plot=False, check_val=1):
        # Returns True if all holes are less than the limit and False otherwise
        # check_val = 1 to check for size of holes (1 is a hole)
        # check_val = 0 to check the size of mask element clumps (0 is a mask element)
        if self.tetrisify:
            mask = np.where(mask == 0, 1, 0)

        # -------- Hole Detection ------------
        row_holes, holes = [], []
        for m, row in enumerate(mask):
            temp_holes = []
            for n, col in enumerate(row):
                if col == check_val:
                    temp_holes.append(n)
            holes.append(temp_holes)

        # -------- Groups Assignment ----------
        groups, change_groups, gn = [], [[],[]], 0
        for j, r in enumerate(holes):
            temp_groups = []
            for k, c in enumerate(r):
                if k != 0:
                    if c-1 != holes[j][k-1]:
                        gn += 1
                temp_groups.append(gn)

            if j != 0:
                sub_groups = [[i for i, x in enumerate(temp_groups) if x == g] for g in set(temp_groups)]

                for sub in sub_groups:
                    group_matched = False
                    for s in sub:
                        if holes[j][s] in holes[j-1]:
                            if group_matched:
                                if len(change_groups[0]) != 0:
                                    if new_sub_group in change_groups[0]:
                                        change_groups[1].append(change_groups[1][change_groups[0].index(new_sub_group)])
                                    else:
                                        change_groups[1].append(new_sub_group)
                                else:
                                    change_groups[1].append(new_sub_group)
                                change_groups[0].append(groups[j-1][holes[j-1].index(holes[j][s])])
                            else:
                                group_matched = True
                                new_sub_group = groups[j-1][holes[j-1].index(holes[j][s])]
                    if group_matched:
                        for s in sub:
                            temp_groups[s] = new_sub_group
            gn += 1
            groups.append(temp_groups)

        change_groups = [[old, new] for new, old in sorted(zip(change_groups[1], change_groups[0]))]
        changed, exceptions = [[], []], [[], []]
        for i, change_pair in enumerate(change_groups):
            change_from, change_to = change_pair[0], change_pair[1]
            if change_from == change_to:
                continue
            for m, row in enumerate(groups):
                if i != 0:
                    if change_to in changed[0]:
                        change_to = changed[1][changed[0].index(change_to)]

                    if change_from in changed[0] and change_to not in changed[1]:
                        change_from = changed[1][changed[0].index(change_from)]
                        exceptions[0].append(change_from)
                        exceptions[1].append(change_to)

                    changed[0].append(change_from)
                    changed[1].append(change_to)
                if change_from in row:
                    change_indices = [j for j, x in enumerate(groups[m]) if x == change_from]
                    for c in change_indices:
                        groups[m][c] = change_to

        double_exceptions = [[], []]
        for ef, except_from in enumerate(exceptions[0]):
            if except_from in exceptions[0][:ef]:
                double_exceptions[0].append(except_from)
                double_exceptions[1].append(exceptions[1][ef])

                double_exceptions[0].append(exceptions[1][ef-1])
                double_exceptions[1].append(exceptions[1][ef])

        for de in range(len(double_exceptions[0])):
            for m, row in enumerate(groups):
                if double_exceptions[0][de] in row:
                    change_indices = [j for j, x in enumerate(groups[m]) if x == double_exceptions[0][de]]
                    for c in change_indices:
                        groups[m][c] = double_exceptions[1][de]

        unique_groups = set(sum(groups, []))

        if plot:
            mask_change = np.copy(mask-2)
            for m, row in enumerate(holes):
                for n, col in enumerate(row):
                    mask_change[m,col] = groups[m][n]

            plt.figure(dpi=200)
            cmap, norm = from_levels_and_colors(sorted([-2]+list(unique_groups)+[list(unique_groups)[-1]+1]),
                                           ['black']+distinctipy.get_colors(len(unique_groups)))
            plt.imshow(mask_change, cmap=cmap, norm=norm)
            plt.show()
        else:
            group_counts = [list(unique_groups),
                    [sum([sum([1 for i, x in enumerate(row) if x == g]) for row in groups]) for g in unique_groups]]

            alarm = [i for i, x in enumerate(group_counts[1]) if x >= self.hole_limit]

            return not alarm

    def choose_block(self, bsize, invert=False):
        if bsize == 4:
            ct = np.random.randint(len(self.tetris_blocks))
            block = np.rot90(self.tetris_blocks[ct], np.random.randint(4))
        elif bsize == 5:
            ct = np.random.randint(len(self.blokus_blocks))
            block = np.rot90(self.blokus_blocks[ct], np.random.randint(4))
        if invert:
            return (~block.astype(bool)).astype(int), ct
        return block, ct

    def valid_block_placement(self, mask, block, ul_corner):
        if (np.where(mask[ul_corner[0]:ul_corner[0]+block.shape[0],
                         ul_corner[1]:ul_corner[1]+block.shape[1]] == 0, 0, 1) & block).astype(bool).any():
            return False
        return True

    def swap_tetris(self):
        # test is the replacement for self for now
        block_swap_ind = np.random.randint(1, list(self.block_assigns.keys())[-1]+1)
        temp_mask = np.where(self.mask == block_swap_ind, 0, self.mask)

        block_size = 4 + int(self.block_assigns[block_swap_ind]-1 > 6)
        if self.static_blocks:
            block_2_place = (~(self.total_blocks[self.block_assigns[block_swap_ind]-1]).astype(bool)).astype(int)
            block_ind = self.block_assigns[block_swap_ind]-1

        block_needed = True
        while block_needed:
            if self.static_blocks:
                block_2_place = np.rot90(block_2_place, np.random.randint(4))
            else:
                block_2_place, block_ind = self.choose_block(block_size, True)
            c_ind = np.random.randint((np.array(temp_mask.shape) - np.array(block_2_place.shape) + 1))
            if self.valid_block_placement(temp_mask, block_2_place, c_ind):
                for row, col in np.ndindex(block_2_place.shape):
                    if block_2_place[row, col] == 1:
                        temp_mask[c_ind[0]+row, c_ind[1]+col] = block_swap_ind
                block_needed = False

        block_ind = block_ind + 7*int(block_size == 5)

        return temp_mask.copy(), block_swap_ind, block_ind

    def remagnify(self):
        # Run this if the magnification has changed
        self.sample_size = math.floor(self.detector_size/self.magnification)
        self.sens_sample = np.ones((self.sample_size, self.sample_size))
        self.corr_size = signal.correlate2d(self.mask, self.mask[0:self.sample_size, 0:self.sample_size], mode='valid').shape[0]

    def Optimize(self):
        if self.method == 'GreatDeluge':
            self.GreatDeluge()
        elif self.method == 'JustRun':
            self.JustRun()

    def GreatDeluge(self):
        if self.cont:
            data = np.load(self.data_dir+'INCOMPLETE_data.npy', allow_pickle=True).item()
            os.remove(self.data_dir+'INCOMPLETE_data.npy')

            Q_init_metric, sens_init_metric, flat_init_metric = self.CalculateMetric(self.init_mask)
            initial_metrics = [Q_init_metric, sens_init_metric, flat_init_metric]
            Q_min_metric, sens_min_metric, flat_min_metric = self.CalculateMetric(self.min_mask, init_metrics=initial_metrics)
            min_metric = Q_min_metric + sens_min_metric + flat_min_metric

            itr = data['Iterations'][-1]
            water_level = data['Water Levels'][-1]
            stop_i = data['Stopping Iterations'][-1]

            try:
                saves = sorted([[int(f.split('_')[1].split('-')[0]), f] for f in os.listdir(self.data_dir) if f.startswith('data_')])[-1][0]//self.save_ev + 1
            except IndexError:
                saves = 0

            print('\033[36m\033[1mA total of {} iterations has been run so far\033[0m'.format(itr))

        else:
            Q_init_metric, sens_init_metric, flat_init_metric = self.CalculateMetric(self.mask)
            initial_metrics = [Q_init_metric, sens_init_metric, flat_init_metric]
            min_metric, water_level = self.corr_weight+self.sens_weight+self.flat_weight, self.corr_weight+self.sens_weight+self.flat_weight
            itr, stop_i = 0, 0
            saves = 0

            data = {}
            data['Metrics'] = [self.corr_weight+self.sens_weight+self.flat_weight]
            data['Q Metrics'] = [self.corr_weight]
            data['Sens Metrics'] = [self.sens_weight]
            data['Flat Metrics'] = [self.flat_weight]
            data['Water Levels'] = [water_level]
            data['Iterations'] = [itr]
            data['Stopping Iterations'] = [stop_i]

            self.last_imp_mask = self.mask.copy()
            self.min_mask = self.mask.copy()

            if self.tetrisify:
                self.last_imp_block_assigns = copy.deepcopy(self.block_assigns)
                self.min_block_assigns = copy.deepcopy(self.block_assigns)

        try:
            while (stop_i < self.stopItr):
                itr += 1

                hole_checking = True
                while hole_checking:
                    if self.tetrisify:
                        mask_temp, block_ele_swapped, block_swapped = self.swap_tetris()
                    else:
                        if not self.sectioning:
                            mask_temp = self.mask.copy()
                            mask_temp = self.mask.reshape(self.mask_size**2, )
                            rand_pop   = np.random.choice(np.where(mask_temp > 0)[0]) # Random populated position
                            rand_empty = np.random.choice(np.where(mask_temp < 1)[0]) # Random empty position
                            mask_temp[rand_pop] = 0
                            mask_temp[rand_empty] = 1
                            mask_temp = mask_temp.reshape(self.mask_size, self.mask_size)
                        else:
                            group_pick = np.random.choice(list(np.arange(len(self.group_indices))))

                            mask_temp = self.mask.copy()
                            chosen_group = mask_temp[self.group_indices[group_pick][0][0]:self.group_indices[group_pick][1][0],
                                                     self.group_indices[group_pick][0][1]:self.group_indices[group_pick][1][1]]
                            original_shape = chosen_group.shape
                            chosen_group = chosen_group.reshape(chosen_group.size, )

                            rand_pop   = np.random.choice(np.where(chosen_group > 0)[0]) # Random populated position
                            rand_empty = np.random.choice(np.where(chosen_group < 1)[0]) # Random empty position
                            chosen_group[rand_pop] = 0
                            chosen_group[rand_empty] = 1

                            chosen_group = chosen_group.reshape(original_shape)

                            mask_temp[self.group_indices[group_pick][0][0]:self.group_indices[group_pick][1][0],
                                      self.group_indices[group_pick][0][1]:self.group_indices[group_pick][1][1]] = chosen_group

                    if not self.balanced:
                        hole_checking = False
                    else:
                        if self.hole_size_checking(mask_temp):
                            hole_checking = False

                self.mask = mask_temp.copy()
                if self.tetrisify:
                    self.block_assigns[block_ele_swapped] = block_swapped+1

                new_Q_metric, new_sens_metric, new_flat_metric = self.CalculateMetric(self.mask, init_metrics=initial_metrics)

                if (new_Q_metric+new_sens_metric) < water_level:
                    print('\033[32mImprovement on itr: {}\033[0m'.format(itr))
                    stop_i = 0
                    self.last_imp_mask = self.mask.copy()
                    if self.tetrisify:
                        self.last_imp_block_assigns = copy.deepcopy(self.block_assigns)

                    water_level -= self.decay_rate * (water_level - (new_Q_metric+new_sens_metric))

                    if (new_Q_metric+new_sens_metric+new_flat_metric) < min_metric:
                        min_metric = new_Q_metric+new_sens_metric+new_flat_metric
                        self.min_mask = self.mask.copy()
                        if self.tetrisify:
                            self.min_block_assigns = copy.deepcopy(self.block_assigns)

                else:
                    stop_i += 1
                    print('\033[31mNo improvement on itr: {}\nIt has been {} iterations with no improvement\033[0m'.format(itr, stop_i))
                    self.mask = self.last_imp_mask.copy()
                    if self.tetrisify:
                        self.block_assigns = copy.deepcopy(self.last_imp_block_assigns)
                print('----------------------------------')

                data['Metrics'].append(new_Q_metric+new_sens_metric)
                data['Q Metrics'].append(new_Q_metric)
                data['Sens Metrics'].append(new_sens_metric)
                data['Flat Metrics'].append(new_flat_metric)
                data['Water Levels'].append(water_level)
                data['Iterations'].append(itr)
                data['Stopping Iterations'].append(stop_i)

                if (itr % self.save_ev == 0) and itr != 1 and stop_i != self.stopItr:
                    temp_data = {}
                    for key in data:
                        temp_data[key] = data[key][:-1]
                        data[key] = [data[key][-1]]
                    np.save(self.data_dir+'data_{}-{}.npy'.format(self.save_ev*saves, self.save_ev*(saves+1)-1), temp_data)
                    print('----------------------------------')
                    print('Completed {} iterations! Creating a save point now'.format(self.save_ev*(saves+1)))
                    print('----------------------------------')
                    print('----------------------------------')
                    if not self.tetrisify:
                        self.VisualizeMask(self.min_mask, '{}-{}_best_current'.format(self.save_ev*saves, self.save_ev*(saves+1)-1))
                    else:
                        self.VisualizeTetrisMask(self.min_mask, '{}-{}_best_current'.format(self.save_ev*saves, self.save_ev*(saves+1)-1),
                                                _block_assigns=self.min_block_assigns, label=self.label_blocks)
                    saves += 1
                    del temp_data



            np.savetxt(self.data_dir+'final_mask.txt', self.min_mask, fmt='%i')
            if not self.tetrisify:
                self.VisualizeMask(self.min_mask, 'Final')
            else:
                self.VisualizeTetrisMask(self.min_mask, 'Final', _block_assigns=self.min_block_assigns, label=self.label_blocks)
                self.createOrderSheet(self.min_block_assigns)
            np.save(self.data_dir+'data_{}-{}.npy'.format(self.save_ev*saves, itr), data)

            if self.tetrisify:
                np.save(self.data_dir+'Final_block_assigns.npy', self.min_block_assigns)

            final_data, files_list = {}, []
            for f in os.listdir(self.data_dir):
                if f.endswith('.npy') and not f.endswith('data.npy') and not f.endswith('assigns.npy'):
                    files_list.append([int(f.split('_')[1].split('-')[0]), f])
            files_list.sort()

            for i, fd in enumerate(files_list):
                files_list[i] = np.load(self.data_dir+fd[1], allow_pickle=True).item()

            for key in files_list[0]:
                final_data[key] = np.concatenate(list(d[key] for d in files_list))
            del files_list

            np.save(self.data_dir+'final_data.npy', final_data)

            self.VisualizeWaterLevel()
            self.VisualizeMetricsEvolution()


        except KeyboardInterrupt:
            print('--------------------------------')
            print('Keyboard Interrupt, saving work to continue later')
            print('--------------------------------')

            self.SaveRandomState()
            np.savetxt(self.data_dir+'INCOMPLETE_mask.txt', self.mask, fmt='%i')
            np.savetxt(self.data_dir+'INCOMPLETE_min_mask.txt', self.min_mask, fmt='%i')
            np.savetxt(self.data_dir+'INCOMPLETE_last_imp_mask.txt', self.last_imp_mask, fmt='%i')

            if self.tetrisify:
                np.save(self.data_dir+'INCOMPLETE_block_assigns.npy', self.block_assigns)
                np.save(self.data_dir+'INCOMPLETE_min_block_assigns.npy', self.min_block_assigns)
                np.save(self.data_dir+'INCOMPLETE_last_imp_block_assigns.npy', self.last_imp_block_assigns)

            np.save(self.data_dir+'INCOMPLETE_data.npy', data)

    def JustRun(self):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', '-m', type=str, default='GreatDeluge',
        help=('Name of optimization method to use, either "GreatDeluge" or "JustRun"'))
    parser.add_argument(
        '--stopItr', type=int, default=200,
        help=('Number of iterations to run or run without improvement'))
    parser.add_argument(
        '--save_ev', type=int, default=2500,
        help=('Save a progress file every X iterations'))
    parser.add_argument(
        '--decay_rate', type=float, default=0.025,
        help=('Modifier to the rate at which the water level decays in GD'))
    parser.add_argument(
        '--seed', '-s', type=int, default=int(time.time()),
        help=('Randomization seed'))
    parser.add_argument(
        '--mask_size', type=int, default=46,
        help=('Size of coded aperture mask on one side of the square mask'))
    parser.add_argument(
        '--detector_size', type=int, default=37,
        help=('Number of detector strips on each detector'))
    parser.add_argument(
        '--open_frac', type=float, default=0.5,
        help=('The fraction of elements without masking elements'))
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help=('Prints all print outs if called'))
    parser.add_argument(
        '--magnification', type=float, default=4,
        help=('Mask magnification'))
    parser.add_argument(
        '--hole_limit', type=int, default=80,
        help=('Hole size limit'))
    parser.add_argument(
        '--balanced', '-b', action='store_false', default=True,
        help=('Whether to implement hole limitations'))
    parser.add_argument(
        '--sectioning', action='store_false', default=True,
        help=('Whether to implement mask sectioning'))
    parser.add_argument(
        '--section_offset', type=int, default=0,
        help=('Size of the section offset from magnified detector plane size'))
    parser.add_argument(
        '--transmission', type=float, default=0.05,
        help=('Transmission probability of mask elements'))
    parser.add_argument(
        '--corr_weight', type=float, default=1,
        help=('Weighting factor for cross correlation in metric'))
    parser.add_argument(
        '--sens_weight', type=float, default=1,
        help=('Weighting factor for sensitivity in metric'))
    parser.add_argument(
        '--flat_weight', type=float, default=1,
        help=('Weighting factor for mean diff metric'))
    parser.add_argument(
        '--dual_corr', action='store_true', default=False,
        help=('Dual optimizes for magnification 2 and 4'))
    parser.add_argument(
        '--tetrisify', action='store_false', default=True,
        help=('Whether to make mask out of tetris-like pieces or not'))
    parser.add_argument(
        '--frac_tetris', type=float, default=0.5,
        help=('Fraction of tetrisified mask made out of 4 element blocks'))
    parser.add_argument(
        '--static_blocks', action='store_true', default=False,
        help=('Whether to use the same list of blocks in tetrisifying'))
    parser.add_argument(
        '--label_blocks', action='store_true', default=False,
        help=('Whether to add block type numbering to tetris block plots'))
    parser.add_argument(
        '--bw_tetris', action='store_true', default=False,
        help=('Whether to plot the tetris mask in black/white also'))


    args = parser.parse_args()
    arg_dict = vars(args)

    optimizer = OptimizerClass(**arg_dict)

    optimizer.Optimize()
