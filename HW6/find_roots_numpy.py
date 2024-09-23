import numpy as np 

coeff = [2, -4]
print(np.roots(coeff))
coeff = [1, -8, +4]
print(np.roots(coeff))
coeff = [4, -5, +1, -1]
print(np.roots(coeff))
coeff = [186, -7.22, 15.5, -13.2]
print(np.roots(coeff))

# [2.]
# [7.46410162 0.53589838]
# [1.21372896+0.j         0.01813552+0.45348417j 0.01813552-0.45348417j]
# [-0.15984864+0.4152082j -0.15984864-0.4152082j  0.35851449+0.j       ]