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

_at = os.path.join(log_dir,targets[0],'*.txt')

while True:
    index = int(input('Episode Index (max '+str(len(_at)-1)+'):'))

    logs = {}
    for target in targets:
        _d = os.path.join(log_dir,target)
        if os.path.exists(_d):
            try:
                _log_file = os.path.join(_d,str(index).zfill(7)+'.txt')
                _l = np.loadtxt(_log_file)

                logs[target] = _l
            except Exception as ex:
                print(ex)

    if 'filtered_action' in logs:
        cs = ControlSystem(enable_actuator_dynamics=True)
    else:
        cs = ControlSystem(enable_actuator_dynamics=False)

    plt_cnt = 0
    fig = plt.figure(0, figsize=(12, 9), )
    fig.canvas.set_window_title('Episode '+str(index))
    gs = gridspec.GridSpec(3,2)
    for target in targets:
        if target in logs:
            _l = logs[target]
            
            if target == 'action':
                ax1 = plt.subplot(gs[plt_cnt, :])
                ax1.plot(_l,label='Input Action')
                if 'function' in logs:
                    ax1.plot(logs['function'],label='Random Function')
                else:
                    if 'filtered_action' in logs:
                        ax1.plot(cs.getZetaRef('unfiltered_input'),label='Cos function')
                    else:
                        ax1.plot(cs.getZetaRef('zeta'),label='Cos function')
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
                ax1.plot(logs['y_ref'],label='Y_ref')
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

