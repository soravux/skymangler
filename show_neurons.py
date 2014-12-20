import scipy.misc
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt

if not os.path.isfile('out.pkl'):
    data = []
    for x in range(3):
        with open('params_op_'+str(x)+'.txt', 'rb') as f:
            # Note: (W, b, b_prime)
            data.append(pickle.load(f))

    last_layer = data[0][0] + data[0][1]
    for x in [1, 2]:
        last_layer = np.dot(last_layer, data[x][0]) + data[x][1]
        import pdb; pdb.set_trace()

    with open('out.pkl', 'w') as f:
        pickle.dump(last_layer, f)
else:
    with open('out.pkl', 'r') as f:
        last_layer = pickle.load(f)

#fig = plt.figure()
n = last_layer.shape[1]
for x in range(n):
    plt.subplot(4,4,x+1)
    #scipy.misc.imsave('{}.png'.format(x), )
    plt.imshow(last_layer[:,x].reshape(216, 216))
    plt.axis('off')
    plt.title('{}'.format(x+1))

plt.tight_layout()

plt.show()