import numpy as np
import os
import argparse
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
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
                 cont_file     = None,
                 seed          = int(time.time()),
                 save_ev       = 1000,
                 mask_size     = 64,
                 detector_size = 37,
                 magnification = 2,
                 fill_frac     = 0.5,
                 mask          = None,
                 verbose       = False):
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
        self.cont_file = cont_file
        self.seed = seed
        self.save_ev = save_ev
        self.mask_size = mask_size
        self.detector_size = detector_size
        self.magnification = magnification
        self.sample_size = math.floor(self.detector_size/self.magnification)
        self.fill_frac = fill_frac
        self.mask = mask
        self.verbose = verbose

        try:
            os.mkdir('Optimizations')
        except:
            pass

        if data_dir is None:
            if self.method == 'GreatDeluge':
                data_dir = 'Optimizations/GD_ms_{}-ff_{}-mag_{}-seed_{}/'.format(self.mask_size, str(self.fill_frac).split('.')[1], self.magnification, self.seed)
            elif self.method == 'JustRun':
                data_dir = 'Optimizations/JR_ms_{}-ff_{}-mag_{}-seed_{}/'.format(self.mask_size, str(self.fill_frac).split('.')[1], self.magnification, self.seed)

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

        # with os.listdir(self.data_dir) as files:


        np.random.seed(self.seed)
        if self.cont:
            if self.verbose:
                print('Continuing from previous optimization state')
            self.LoadRandomState()
        else:
            self.CreateMask()
            self.corr_size = signal.correlate2d(self.mask, self.mask[0:self.sample_size, 0:self.sample_size], mode='valid').shape[0]
            self.init_mask = self.mask.copy()
            self.VisualizeMask(self.init_mask, 'Initial')

    def CreateMask(self):
        """
        Creation of the first mask for the "optimization" process
        """
        mask = np.ones(self.mask_size**2)
        num_fill = math.ceil(self.fill_frac * (self.mask_size**2))

        mask[np.random.choice(mask.size, num_fill, replace=False)] = 0

        mask = mask.reshape(self.mask_size, self.mask_size)

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
        plt.plot(final_data['Iterations'], final_data['Water Levels'], label='Water Level', color='black')
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

    def CalculateMetric(self):
        ### IMPLEMENT METRIC CALCULATOR
        F_matrix = np.array([(signal.correlate2d(self.mask, self.mask[m:m+self.sample_size, n:n+self.sample_size], mode='valid')/(self.corr_size**2)).reshape(self.corr_size**2, ) for m in range(0, self.mask_size-self.sample_size+1) for n in range(0, self.mask_size-self.sample_size+1)])
        Q1 = (1/self.sample_size)*np.sum((np.diag(F_matrix)-self.fill_frac)**4)
        np.fill_diagonal(F_matrix, self.fill_frac**2)
        Q2 = (1/(self.sample_size**2-self.sample_size))*np.sum((F_matrix-self.fill_frac**2)**4)
        Q = Q1 + Q2

        return Q

    def Optimize(self):
        if self.method == 'GreatDeluge':
            self.GreatDeluge()
        elif self.method == 'JustRun':
            self.JustRun()

    def GreatDeluge(self):
        start_time = time.time()
        if self.cont:
            pass
        else:
            Q_init_metric = self.CalculateMetric()
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

                self.mask = self.mask.reshape(self.mask_size**2, )
                rand_pop   = np.random.choice(np.where(self.mask > 0)[0]) # Random populated position
                rand_empty = np.random.choice(np.where(self.mask < 1)[0]) # Random empty position
                self.mask[rand_pop] = 0
                self.mask[rand_empty] = 1
                self.mask = self.mask.reshape(self.mask_size, self.mask_size)

                new_metric = self.CalculateMetric()/Q_init_metric

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
                    print('Completed {} iterations! Creating a save point now'.format(self.save_ev*(saves+1)))
                    saves += 1
                    del temp_data

            np.savetxt(self.data_dir+'final_mask.txt', self.min_mask, fmt='%i')
            self.VisualizeMask(self.min_mask, 'Final')
            np.save(self.data_dir+'data_{}-{}.npy'.format(self.save_ev*saves, itr), data)

            end_time = time.time()
            tt = end_time - start_time
            print('It took {} minutes and {} seconds to run {} iterations'.format(tt//60, round(tt%60, 2), itr))
            print('Combining all save points into a singular file now')

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
        '--stopItr', type=int, default=500,
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
        '--mask_size', type=int, default=64,
        help=('Size of coded aperture mask on one side of the square mask'))
    parser.add_argument(
        '--fill_frac', type=float, default=0.5,
        help=('The fraction of elements filled with masking elements'))
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help=('Prints all print outs if called'))
    parser.add_argument(
        '--cont', action='store_true', default=False,
        help=('User set method of continuation'))
    parser.add_argument(
        '--magnification', type=float, default=2,
        help=('Mask magnification'))

    args = parser.parse_args()
    arg_dict = vars(args)

    optimizer = OptimizerClass(**arg_dict)

    optimizer.Optimize()
