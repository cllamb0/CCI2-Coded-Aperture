import os

print('Now running seed 2')
os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 2 --sens_weight 1 --seed 2')
print('Now running seed 5')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 2 --sens_weight 1 --seed 5')
print('Now running seed 8')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 2 --sens_weight 1 --seed 8')
print('Now running seed 11')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 2 --sens_weight 1 --seed 11')
print('Now running seed 14')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 2 --sens_weight 1 --seed 14')


print('Now running seed 17')
os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 1 --sens_weight 1 --seed 17')
print('Now running seed 20')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 1 --sens_weight 1 --seed 20')
print('Now running seed 23')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 1 --sens_weight 1 --seed 23')
print('Now running seed 26')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 1 --sens_weight 1 --seed 26')
print('Now running seed 29')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 1 --sens_weight 1 --seed 29')

print('Now running seed 32')
os.system('python optimize.py --open_frac 0.4 --hole_limit 17 --corr_weight 1 --sens_weight 2 --seed 32')
print('Now running seed 35')
os.system('python optimize.py --open_frac 0.45 --hole_limit 30 --corr_weight 1 --sens_weight 2 --seed 35')
print('Now running seed 38')
os.system('python optimize.py --open_frac 0.5 --hole_limit 50 --corr_weight 1 --sens_weight 2 --seed 38')
print('Now running seed 41')
os.system('python optimize.py --open_frac 0.55 --hole_limit 107 --corr_weight 1 --sens_weight 2 --seed 41')
print('Now running seed 44')
os.system('python optimize.py --open_frac 0.6 --hole_limit 300 --corr_weight 1 --sens_weight 2 --seed 44')

print('ALL DONE')