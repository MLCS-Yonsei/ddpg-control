import numpy as np
from glob import glob
import os

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from system import ControlSystem

folder = input("folder name:")
# folder = '2018-10-23_20-42-38'

log_dir = os.path.join('logs',folder)
targets = ['action','filtered_action','function','y_hat','y_ref']
# targets = ['action']

logs = {}
for target in targets:
    _d = os.path.join(log_dir,target)
    if os.path.exists(_d):
        _log_files = sorted(glob(os.path.join(_d,'*.txt')))

        if len(_log_files) > 0:
            _ls = np.zeros((len(_log_files),np.loadtxt(_log_files[0]).shape[0]))
            for i, _l in enumerate(_log_files):
                _index = int(os.path.basename(_l).split('.')[0])
                _ls[_index] = np.loadtxt(_l)

            logs[target] = _ls

while True:
    if 'filtered_action' in logs:
        cs = ControlSystem(enable_actuator_dynamics=True)
    else:
        cs = ControlSystem(enable_actuator_dynamics=False)

    index = int(input('Episode Index (max '+str(len(logs['action'])-1)+'):'))

    plt_cnt = 0
    fig = plt.figure(0, figsize=(12, 9), )
    fig.canvas.set_window_title('Episode '+str(index))
    gs = gridspec.GridSpec(3,2)
    for target in targets:
        if target in logs:
            _l = logs[target][index]
            
            if target == 'action':
                ax1 = plt.subplot(gs[plt_cnt, :])
                ax1.plot(_l,label='Input Action')
                if 'function' in logs:
                    ax1.plot(logs['function'][index],label='Random Function')
                else:
                    if 'filtered_action' in logs:
                        ax1.plot(cs.getZetaRef('unfiltered_input'),label='Random Function')
                    else:
                        ax1.plot(cs.getZetaRef('zeta'),label='Random Function')
                # ax1.xlabel('t')
                h,l=ax1.get_legend_handles_labels()
                ax1.legend(h,l)
            elif target == 'filtered_action' and 'filtered_action' in logs:
                ax1=plt.subplot(gs[plt_cnt, :])
                ax1.plot(_l,label='Filtered Action')
                ax1.plot(cs.getZetaRef('zeta'),label='Filtered Action Ref')
                # plt.xlabel('t')
                h,l=ax1.get_legend_handles_labels()
                ax1.legend(h,l)
            elif target == 'y_hat':
                ax1=plt.subplot(gs[plt_cnt, :])
                ax1.plot(_l,label='y_hat')
                ax1.plot(logs['y_ref'][index],label='Y_ref')
                # plt.xlabel('t')
                h,l=ax1.get_legend_handles_labels()
                ax1.legend(h,l)
            else:
                plt_cnt -= 1
            plt_cnt += 1

    # plt.show()
    plt.draw()
    plt.pause(1) # <-------
    input("<Hit Enter To Close>")
    plt.close(fig)

