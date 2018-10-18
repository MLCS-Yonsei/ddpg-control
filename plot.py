import numpy as np
from glob import glob
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

log_dirs = glob('./log/*')
log_files = sorted(glob(os.path.join(log_dirs[len(log_dirs)-1],'*')))

_l = np.loadtxt(log_files[0])
logs = np.zeros((len(log_files),_l.shape[0]))

# fig, ax = plt.subplots()

# x = np.arange(0, _l.shape[0], 1)
# line, = ax.plot(x, logs[0])

for i, _l in enumerate(log_files):
    _index = int(os.path.basename(_l).split('.')[0])
    logs[_index] = np.loadtxt(_l)

# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)

# ax1.set_xlim([0,logs.shape[1]])
# ax1.autoscale_view()

# im, = ax1.plot([], [], color=(0,0,1))

# def func(n):
#     print(logs[n, 0])
#     im.set_xdata(x)
#     im.set_ydata(logs[n, :])

#     return im

# ani = animation.FuncAnimation(fig, func, frames=logs.shape[0], interval=1, blit=False)

# plt.show()

plt.plot(logs[4000])
plt.show()

# plt.plot(np.loadtxt('episode_reward.txt'))
# plt.show()