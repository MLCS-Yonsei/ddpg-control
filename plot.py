import numpy as np
from glob import glob
import os

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
import matplotlib.animation as animation

from system import ControlSystem

cs = ControlSystem(enable_actuator_dynamics=False)
Y_ref = cs.getYRef(display=False)


# log_dirs = glob('./log/*')
# log_files = sorted(glob(os.path.join(log_dirs[len(log_dirs)-1],'*')))
folder = input("folder name:")
folder_type = input("folder type name:")
log_dirs = glob('./181021/'+folder_type+'/' + folder)
log_files = sorted(glob(os.path.join(log_dirs[len(log_dirs)-1],'*')))

_l = np.loadtxt(log_files[0])
logs = np.zeros((len(log_files),_l.shape[0]))

print("Total Length: ", len(log_files))
# fig, ax = plt.subplots()

# x = np.arange(0, _l.shape[0], 1)
# line, = ax.plot(x, logs[0])

for i, _l in enumerate(log_files):
    _index = int(os.path.basename(_l).split('.')[0])
    # print(_index)
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

while True:
    try:
        plt.close()
    except Exception as ex:
        pass
    data = input("Index:")
    plt.plot(Y_ref,label='y_optimal')
    plt.plot(logs[int(data)],label='y_hat')
    plt.xlabel('t') 
    plt.show()

# plt.plot(np.loadtxt('episode_reward.txt'))
# plt.show()