import numpy as np
import time
from tqdm import tqdm
import math
import os

# ALL UNITS IN mm

# Outputs a attenuation length for a specific plane in z

################################################################################
# Parameter setting
det_pix_size = 2 # Size of each strip
det_thickness = 2.4 # Thickness of the detector # DON'T KNOW THIS VALUE YET
det_size = 32 # Number of strips on each detector face

# a = 23 1/3 mm and b = 70 mm --> m = 4
a = 23 + 1/3 # Distance between source and front of mask plane
b = 70 # Distance between front of mask plane and detector plane

mask_pix_size = 2 # Size of each mask element location
mask_thickness = 3 # Thickness of the mask

voxel_size = 2 # Size of cubic voxel aka step size in the image space

delta_alpha = 0.1 # Step size of alpha affine parameter

z_depth = 100 # Width of imaging space in z direction

mu_rho = 6.422E-01 # cm^2/g (for Fr221's 218keV gamma on tungsten)
rho = 19.28 # g/cm^3 Tungsten density
mu = (mu_rho * rho)/10 # [1/mm]
################################################################################

start_time = time.time()
data_dir = 'Optimizations-Desktop-Tetris/Optimizations-Tetris-Final/'
try:
    os.mkdir(data_dir+'Response-Matrix/')
except:
    pass
save_dir = data_dir + 'Response-Matrix/'

mask_dir = 'GD_ms_46-of_3-mag_4-seed_44-hl_80-cw_1.0-sw_2.0-ft_0.4/'
CA_mask = np.loadtxt(data_dir+mask_dir+'final_mask.txt')
CA_mask = np.where(CA_mask == 0, 0, 1)

det_start = ((CA_mask.shape[0] - det_size) / 2) * mask_pix_size + (det_pix_size / 2)

# detector_pixels = np.array([[det_start+(m*det_pix_size), det_start+(n*det_pix_size), (b-mask_thickness)] for m in range(det_size) for n in range(det_size)])
det_x, det_y = np.mgrid[det_start:det_start+(det_size*det_pix_size):det_pix_size, det_start:det_start+(det_size*det_pix_size):det_pix_size]
detector_pixels = np.hstack((np.array([det_x.flatten(), det_y.flatten()]).T, (np.ones(det_x.flatten().size)*((b-mask_thickness))).reshape(-1,1)))

vox_start = -(100-(CA_mask.shape[0]*mask_pix_size))/2 + (voxel_size / 2) # 100 from 10cm extension @ m = 4 and b = 7cm
vox_x, vox_y, vox_z = np.mgrid[vox_start:100+vox_start:voxel_size, vox_start:100+vox_start:voxel_size, -(b+z_depth+(voxel_size/2))+voxel_size:-(b+(voxel_size/2))+voxel_size:voxel_size]
voxels = np.array([vox_x.flatten(), vox_y.flatten(), vox_z.flatten()]).T

response = np.empty((voxels.shape[0], detector_pixels.shape[0]))

### Looping over each voxel (Vj) and detector pixel (Ui)
for m, Vj in tqdm(enumerate(voxels), total=voxels.shape[0], desc='Iterating voxel --> detector pixels'):
    for n, Ui in enumerate(detector_pixels):
        Sij = Ui - Vj
        s_hat = Sij/np.sqrt(Sij.dot(Sij)) # unit vector in direction of Vj --> Ui

        alpha_t = (a/(a+b))*np.sqrt(Sij.dot(Sij)) # Initial alpha param
        alpha_0 = ((a+mask_thickness)/(a+b))*np.sqrt(Sij.dot(Sij)) # Final alpha param

        alpha_steps = (alpha_0 - alpha_t)/delta_alpha
        excess_step = alpha_steps % 1 # The excess small step for error
        alpha_steps = math.floor(alpha_steps)

        ray_pos_t = Vj + alpha_t*s_hat # Ray vector coord at entrance to the mask

        # List of XY coordinates along the Ray vector within the mask
        L_coords = [list((ray_pos_t+(n+1)*delta_alpha*s_hat)[:-1]) for n in range(alpha_steps)]
        L_excess_coord = list((ray_pos_t+(alpha_steps+excess_step)*delta_alpha*s_hat)[:-1])

        # Calculating the total attenuation length
        L = sum([CA_mask[int(xn//mask_pix_size), int(yn//mask_pix_size)]*delta_alpha for xn, yn in L_coords])
        L += CA_mask[int(L_excess_coord[0]//mask_pix_size), int(L_excess_coord[1]//mask_pix_size)]*(excess_step*delta_alpha)

        atten_frac = np.exp(-mu*L)

        # Angle between Sij and normal vector to detector face
        theta = np.arccos(np.clip(np.dot(Sij/np.linalg.norm(Sij), np.array([0,0,-1])), -1.0, 1.0))

        solidAngle = ((det_pix_size**2)*math.cos(theta))/(4*math.pi*(np.linalg.norm(Sij)**2)+2*(det_pix_size**2)*math.cos(theta))

        response[m,n] = -solidAngle * atten_frac
    # print(atten_frac)
np.save(save_dir+'response_matrix-seed-{}.npy'.format(mask_dir.split('seed')[1].split('-')[0].strip('_')), response)
total_time = time.time()-start_time
print('Iterating took a total of: {} minutes and {} seconds'.format(total_time//60, round(total_time%60, 2)))
