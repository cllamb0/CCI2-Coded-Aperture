import numpy as np
import os
import argparse
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from distinctipy import distinctipy
import math
import scipy.signal as signal

class OptimizerClass:
    def __init__(self,
                 method        = 'GreatDeluge',
                 stopItr       = 500,
                 data_fname    = None,
                 data_dir      = None,
                 plots_dir     = None,
                 decay_rate    = 0.025,
                 cont          = False,
                 seed          = int(time.time()),
                 save_ev       = 1000,
                 mask_size     = 46,
                 detector_size = 37,
                 magnification = 3,
                 fill_frac     = 0.5,
                 hole_limit    = 80,
                 balanced      = True,
                 mask          = None,
                 verbose       = False,
                 section_offset= 0,
                 sectioning    = True,
                 group_indices = None,
                 transmission  = 0.05,
                 corr_weight   = 1,
                 sens_weight   = 1):
        """
        Initializer for all class variables and intial
        """

        assert method in ['GreatDeluge', 'JustRun']

        if verbose:
            print('Initializing optimization configuration')

        self.method = method
        self.stopItr = stopItr
        self.data_fname = data_fname
        self.decay_rate = decay_rate
        self.cont = cont
        self.seed = seed
        self.save_ev = save_ev
        self.mask_size = mask_size
        self.detector_size = detector_size
        self.magnification = magnification
        self.sample_size = math.floor(self.detector_size/self.magnification)
        self.sens_sample = np.ones((self.sample_size, self.sample_size))
        self.fill_frac = fill_frac
        self.hole_limit = hole_limit
        self.balanced = balanced
        self.mask = mask
        self.verbose = verbose
        self.section_offset = section_offset
        self.section_size = self.sample_size - self.section_offset
        self.sectioning = sectioning
        self.group_indices = group_indices
        self.transmission = transmission
        self.corr_weight = corr_weight
        self.sens_weight = sens_weight

        try:
            os.mkdir('Optimizations')
        except:
            pass

        if data_dir is None:
            if self.method == 'GreatDeluge':
                data_dir = 'Optimizations/GD_ms_{}-ff_{}-mag_{}-seed_{}-hl_{}/'.format(self.mask_size, str(self.fill_frac).split('.')[1], self.magnification, self.seed, self.hole_limit)
            elif self.method == 'JustRun':
                data_dir = 'Optimizations/JR_ms_{}-ff_{}-mag_{}-seed_{}-hl_{}/'.format(self.mask_size, str(self.fill_frac).split('.')[1], self.magnification, self.seed, self.hole_limit)

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

        else:
            self.CreateMask()
            self.init_mask = self.mask.copy()
            self.VisualizeMask(self.init_mask, 'Initial')

        self.corr_size = signal.correlate2d(self.mask, self.mask[0:self.sample_size, 0:self.sample_size], mode='valid').shape[0]
        print(self.sample_size)
        print(self.corr_size)

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
            if not self.sectioning:
                mask = np.ones(self.mask_size**2)
                num_fill = math.ceil(self.fill_frac * (self.mask_size**2))

                mask[np.random.choice(mask.size, num_fill, replace=False)] = 0

                mask = mask.reshape(self.mask_size, self.mask_size)
            else:
                num_fill = math.ceil(self.fill_frac * (self.section_size**2))

                group_indices = []
                for col_section in range(math.floor(self.mask_size/self.section_size)):
                    temp_column = np.ones(self.section_size**2)
                    temp_column[np.random.choice(temp_column.size, num_fill, replace=False)] = 0
                    temp_column = temp_column.reshape(self.section_size, self.section_size)
                    group_indices.append([[0, col_section*self.section_size],
                                              [self.section_size, (col_section+1)*self.section_size]])
                    for row_section in range(math.floor(self.mask_size/self.section_size)-1):
                        temp_mask = np.ones(self.section_size**2)
                        temp_mask[np.random.choice(temp_mask.size, num_fill, replace=False)] = 0
                        temp_mask = temp_mask.reshape(self.section_size, self.section_size)
                        temp_column = np.concatenate((temp_column, temp_mask), axis=0)
                        group_indices.append([[(row_section+1)*self.section_size, (col_section)*self.section_size],
                                              [(row_section+2)*self.section_size, (col_section+1)*self.section_size]])
                    if col_section == 0:
                        mask = np.copy(temp_column)
                    else:
                        mask = np.concatenate((mask, temp_column), axis=1)

                if self.mask_size % self.section_size != 0:
                    fix_column = np.ones((self.mask_size%self.section_size)*(mask.shape[0]))
                    col_fix_fill = math.floor(self.fill_frac * (fix_column.size))
                    fix_column[np.random.choice(fix_column.size, col_fix_fill, replace=False)] = 0
                    fix_column = fix_column.reshape((mask.shape[0]), (self.mask_size%self.section_size))

                    mask = np.concatenate((mask, fix_column), axis=1)
                    group_indices.append([[0, mask.shape[0]], list(mask.shape)])

                    fix_row = np.ones((self.mask_size%self.section_size)*(mask.shape[1]))
                    row_fix_fill = math.floor(self.fill_frac * (fix_row.size))
                    fix_row[np.random.choice(fix_row.size, row_fix_fill, replace=False)] = 0
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

        if save:
            np.savetxt(self.data_dir+'Initial_mask.txt', mask, fmt='%i')

            self.mask = mask.copy()

    def VisualizeMask(self, plot_mask, sname):
        """
        Plotter for way of visualizing a mask in a clean looking way
        """
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

        plt.title('Mask Size: {} | Fill Fraction: {}'.format(self.mask_size, self.fill_frac) +
                  '\nRandom Seed: {}'.format(self.seed), fontsize='small', y = 0.95)

        plt.savefig(self.plots_dir+'{}_coded_mask.png'.format(sname), bbox_inches='tight')

        plt.close()

        print('Saved {} aperture mask image'.format(sname))

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

    def CalculateMetric(self, mask):
        # Cross correlation metric
        F_matrix = np.array([(signal.correlate2d(mask, mask[m:m+self.sample_size, n:n+self.sample_size], mode='valid')/(self.corr_size**2)).reshape(self.corr_size**2, ) for m in range(0, self.mask_size-self.sample_size+1) for n in range(0, self.mask_size-self.sample_size+1)])
        Q1 = (1/self.sample_size)*np.sum((np.diag(F_matrix)-self.fill_frac)**4)
        np.fill_diagonal(F_matrix, self.fill_frac**2)
        Q2 = (1/(self.sample_size**2-self.sample_size))*np.sum((F_matrix-self.fill_frac**2)**4)

        # "Sensitivity" metric
        sens_mask = np.copy(mask)
        sens_mask[sens_mask == 0] = self.transmission
        sensitivity_matrix = signal.correlate2d(sens_mask, self.sens_sample, mode='valid')

        return self.corr_weight*(Q1+Q2) + self.sens_weight*(np.var(sensitivity_matrix))

    def hole_size_checking(self, mask, plot=False):
        # Returns True if all holes are less than the limit and False otherwise
        # -------- Hole Detection ------------
        row_holes, holes = [], []
        for m, row in enumerate(mask):
            temp_holes = []
            for n, col in enumerate(row):
                if col == 1:
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
        changed = [[], []]
        for i, change_pair in enumerate(change_groups):
            change_from, change_to = change_pair[0], change_pair[1]
            if change_from == change_to:
                continue
            for m, row in enumerate(groups):
                if i != 0:
                    if change_to in changed[0]:
                        change_to = changed[1][changed[0].index(change_to)]
                    elif change_from in changed[0] and change_to not in changed[1]:
                        change_to_old = change_to
                        change_to = changed[1][changed[0].index(change_from)]
                        change_from = change_to_old

                    changed[0].append(change_from)
                    changed[1].append(change_to)
                if change_from in row:
                    change_indices = [j for j, x in enumerate(row) if x == change_from]
                    for c in change_indices:
                        groups[m][c] = change_to
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

    def Optimize(self):
        if self.method == 'GreatDeluge':
            self.GreatDeluge()
        elif self.method == 'JustRun':
            self.JustRun()

    def GreatDeluge(self):
        if self.cont:
            data = np.load(self.data_dir+'INCOMPLETE_data.npy', allow_pickle=True).item()
            os.remove(self.data_dir+'INCOMPLETE_data.npy')

            Q_init_metric = self.CalculateMetric(self.init_mask)
            min_metric = self.CalculateMetric(self.min_mask)

            itr = data['Iterations'][-1]
            water_level = data['Water Levels'][-1]
            stop_i = data['Stopping Iterations'][-1]

            try:
                saves = sorted([[int(f.split('_')[1].split('-')[0]), f] for f in os.listdir(self.data_dir) if f.startswith('data_')])[-1][0]//self.save_ev + 1
            except IndexError:
                saves = 0

            print('\033[36m\033[1mA total of {} iterations has been run so far\033[0m'.format(itr))

        else:
            Q_init_metric = self.CalculateMetric(self.mask)
            min_metric, water_level = 1, 1
            itr, stop_i = 0, 0
            saves = 0

            data = {}
            data['Metrics'] = [1]
            data['Water Levels'] = [water_level]
            data['Iterations'] = [itr]
            data['Stopping Iterations'] = [stop_i]

            self.last_imp_mask = self.mask.copy()
            self.min_mask = self.mask.copy()

        try:
            while (stop_i < self.stopItr):
                itr += 1

                hole_checking = True
                while hole_checking:
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

                new_metric = self.CalculateMetric(self.mask)/Q_init_metric

                if new_metric < water_level:
                    print('\033[32mImprovement on itr: {}\033[0m'.format(itr))
                    stop_i = 0
                    self.last_imp_mask = self.mask.copy()

                    water_level -= self.decay_rate * (water_level - new_metric)

                    if new_metric < min_metric:
                        min_metric = new_metric
                        self.min_mask = self.mask.copy()

                else:
                    stop_i += 1
                    print('\033[31mNo improvement on itr: {}\nIt has been {} iterations with no improvement\033[0m'.format(itr, stop_i))
                    self.mask = self.last_imp_mask.copy()
                print('----------------------------------')

                data['Metrics'].append(new_metric)
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
                    print('----------------------------------')
                    print('Completed {} iterations! Creating a save point now'.format(self.save_ev*(saves+1)))
                    print('----------------------------------')
                    print('----------------------------------')
                    self.VisualizeMask(self.min_mask, '{}-{}_best_current'.format(self.save_ev*saves, self.save_ev*(saves+1)-1))
                    saves += 1
                    del temp_data



            np.savetxt(self.data_dir+'final_mask.txt', self.min_mask, fmt='%i')
            self.VisualizeMask(self.min_mask, 'Final')
            np.save(self.data_dir+'data_{}-{}.npy'.format(self.save_ev*saves, itr), data)

            final_data, files_list = {}, []
            for f in os.listdir(self.data_dir):
                if f.endswith('.npy') and not f.endswith('data.npy'):
                    files_list.append([int(f.split('_')[1].split('-')[0]), f])
            files_list.sort()

            for i, fd in enumerate(files_list):
                files_list[i] = np.load(self.data_dir+fd[1], allow_pickle=True).item()

            for key in files_list[0]:
                final_data[key] = np.concatenate(list(d[key] for d in files_list))
            del files_list

            np.save(self.data_dir+'final_data.npy', final_data)

            self.VisualizeWaterLevel()


        except KeyboardInterrupt:
            print('--------------------------------')
            print('Keyboard Interrupt, saving work to continue later')
            print('--------------------------------')

            self.SaveRandomState()
            np.savetxt(self.data_dir+'INCOMPLETE_mask.txt', self.mask, fmt='%i')
            np.savetxt(self.data_dir+'INCOMPLETE_min_mask.txt', self.min_mask, fmt='%i')
            np.savetxt(self.data_dir+'INCOMPLETE_last_imp_mask.txt', self.last_imp_mask, fmt='%i')

            np.save(self.data_dir+'INCOMPLETE_data.npy', data)

    def JustRun(self):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', '-m', type=str, default='GreatDeluge',
        help=('Name of optimization method to use, either "GreatDeluge" or "JustRun"'))
    parser.add_argument(
        '--stopItr', type=int, default=100,
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
        '--fill_frac', type=float, default=0.5,
        help=('The fraction of elements filled with masking elements'))
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help=('Prints all print outs if called'))
    parser.add_argument(
        '--magnification', type=float, default=3,
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


    args = parser.parse_args()
    arg_dict = vars(args)

    optimizer = OptimizerClass(**arg_dict)

    optimizer.Optimize()
