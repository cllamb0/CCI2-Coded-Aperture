import os

os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 2 --sens_weight 1 --seed 1')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 2 --sens_weight 1 --seed 4')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 2 --sens_weight 1 --seed 7')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 2 --sens_weight 1 --seed 10')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 2 --sens_weight 1 --seed 13')

os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 1 --sens_weight 1 --seed 16')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 1 --sens_weight 1 --seed 19')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 1 --sens_weight 1 --seed 22')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 1 --sens_weight 1 --seed 25')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 1 --sens_weight 1 --seed 28')

os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 1 --sens_weight 2 --seed 31')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 1 --sens_weight 2 --seed 34')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 1 --sens_weight 2 --seed 37')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 1 --sens_weight 2 --seed 40')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 1 --sens_weight 2 --seed 43')

print('ALL DONE')
