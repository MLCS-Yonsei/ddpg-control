try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

from control import *
from control.matlab import *

from gym import error, spaces

class ControlSystem:
    '''
    Initial values are declared in each function.
    Override the value when calling the function if needed.

    zeta = 0.707
    w0 = 1.
    ts = 0.02
    duration = 16 # in seconds
    '''
    
    def __init__(self, enable_actuator_dynamics = False):
        self.enable_actuator_dynamics = enable_actuator_dynamics
        self.impulse_magnitude = 10

        self.T = self.getTimeVector(duration=16,ts=0.02)
        self.resetValues()

        # Setting for DDPG
        self.timestep_limit = len(self.T)

        self.observation_space = spaces.Box(low=0, high=1,
                                    shape=(2,))
        self.action_space = spaces.Box(low=-0., high=1.,
                                    shape=(1,))

        

    def resetValues(self):
        self._y = 0.
        self.Y = []
        self.Y_ref = self.getYRef(display=False)
        
        if self.enable_actuator_dynamics == True:
            self.ad = {}
            self.ad['co'] = self.computeActuatorDynamics(T_s=0.5, ts=0.02)
            self.ad['_z'] = 0.
            self.ad['Z'] = []
            self.ad['th1'] = 0
            self.ad['u1'] = 0
            self.ad['th'] = None

        # Impulse Response
        self.theta_2 = 0
        self.theta_1 = 0
        self.u_2 = 0
        self.u_1 = self.impulse_magnitude

        self.theta = None

    def getTimeVector(self, duration=16, ts=0.02):
        return np.arange(0, duration, ts)

    def computeSystem(self, zeta=0.707, w0=1., ts=0.02):
        # if zeta > 1:
        #     zeta = 1
        # elif zeta < 0:
        #     zeta = 0
        g = tf(w0*w0, [1,2*zeta,w0*w0])
        gz = c2d(g,ts)

        coeffs = tfdata(gz)

        co = {
            'a1':coeffs[1][0][0][1],
            'a2':coeffs[1][0][0][2],
            'b1':coeffs[0][0][0][0],
            'b2':coeffs[0][0][0][1],
            'dt':gz.dt
        }

        return co

    def computeActuatorDynamics(self, T_s=0.5, ts=0.02):
        '''
        ts : sampling time
        '''
        g = tf(1, [T_s, 1])
        gz = c2d(g,ts)

        coeffs = tfdata(gz)

        co = {
            'a1':coeffs[1][0][0][1],
            'b1':coeffs[0][0][0][0],
            'dt':gz.dt
        }

        return co

    def computeZetaFromActuatorDynamics(self, action):
        self.ad['Z'].append(self.ad['_z'])

        self.ad['u1'] = action

        self.ad['th'] = - self.ad['co']['a1'] * self.ad['th1'] + self.ad['co']['b1'] * self.ad['u1']
        self.ad['_z'] += self.ad['th'] - self.ad['th1']

        self.ad['th1'] = self.ad['th']

        return self.ad['_z']

    def computeNextStep(self, action, current_step):
        self.Y.append(self._y)

        if self.enable_actuator_dynamics == True:
            zeta = self.computeZetaFromActuatorDynamics(action)
        else:
            zeta = action

        co = self.computeSystem(zeta=zeta)

        self.theta = - co['a1'] * self.theta_1 - co['a2'] * self.theta_2 + co['b1'] * self.u_1 + co['b2'] * self.u_2
        self._y += self.theta - self.theta_1

        self.theta_2 = self.theta_1
        self.theta_1 = self.theta
        self.u_2 = self.u_1
        if current_step == int(len(self.T) / 2):
            # Another Impulse at half time
            # 타임스텝이 홀수개일 경우 두번 연속될 수 있으므로 보완필요
            self.u_1 = self.impulse_magnitude
        else:
            try:
                self.u_1 = self.u
            except:
                # Impulse Response
                self.u_1 = 0

    def zetaFunction(self, time_vector, index):
        _x = index * (np.pi / len(time_vector))
        return np.cos(_x) / 6 + 2/6 # Cos curve drop from 0.5 to 0.167

    def plotZetaRef(self):
        Zeta = []
        for time_index, _t in enumerate(self.T):
            _zeta = self.zetaFunction(time_vector=self.T, index=time_index)
            Zeta.append(_zeta)

        self.plotGraph(self.T, Zeta)

    def plotGraph(self, x, y):
        plt.step(x,y)
        plt.grid()
        plt.xlabel('x') 
        plt.ylabel('y')
        plt.show()

    def getYRef(self, display=False):
        def computeZetaFromActuatorDynamicsRef(ad, action):
            ad['Z'].append(ad['_z'])

            ad['u1'] = action

            ad['th'] = - ad['co']['a1'] * ad['th1'] + ad['co']['b1'] * ad['u1']
            ad['_z'] += ad['th'] - ad['th1']

            ad['th1'] = ad['th']

            return ad['_z']

        Y_ref = []
        Zeta = []
        
        # impulse
        th1 = 0
        th2 = 0
        u1 = self.impulse_magnitude
        u2 = 0

        if self.enable_actuator_dynamics == True:
            ad = {}
            ad['co'] = self.computeActuatorDynamics(T_s=0.5, ts=0.02)
            ad['_z'] = 0.
            ad['Z'] = []
            ad['th1'] = 0
            ad['u1'] = 0
            ad['th'] = None

            unfiltered_input = []

        _y_ref = 0.
        for time_index, _t in enumerate(self.T):
            Y_ref.append(_y_ref)

            # Compute
            if self.enable_actuator_dynamics == True:
                _input = self.zetaFunction(time_vector=self.T, index=time_index)
                unfiltered_input.append(_input)
                zeta = computeZetaFromActuatorDynamicsRef(ad, _input)
            else:
                zeta = self.zetaFunction(time_vector=self.T, index=time_index)

            Zeta.append(zeta)
            co = self.computeSystem(zeta=zeta)

            theta = - co['a1'] * th1 - co['a2'] * th2 + co['b1'] * u1 + co['b2'] * u2
            _y_ref += theta - th1

            th2 = th1
            th1 = theta
            u2 = u1

            if time_index == int(len(self.T) / 2):
                # Another Impulse at half time
                # 타임스텝이 홀수개일 경우 두번 연속될 수 있으므로 보완필요
                u1 = self.impulse_magnitude
            else:
                try:
                    u1 = u
                except:
                    # Impulse Response
                    u1 = 0
            
        if display is True:
            if self.enable_actuator_dynamics == True:
                print("unfiltered_input")
                print("Zeta")
                plt.step(self.T,unfiltered_input)
                plt.grid()
                plt.xlabel('t') 
                plt.ylabel('Unfiltered Input')
                plt.show()

            print("Zeta")
            plt.step(self.T,Zeta)
            plt.grid()
            plt.xlabel('t') 
            plt.ylabel('Filtered Zeta')
            plt.show()

            print("y_ref")
            plt.step(self.T,Y_ref)
            plt.grid()
            plt.xlabel('t') 
            plt.ylabel('y_ref')
            plt.show()

        return Y_ref

    def getReward(self, time_index):
        return -(abs(self.Y_ref[time_index] - self.Y[time_index])**2)

    def step(self, action, time_index):
        '''
        Computes system with given action
        returns y_t and y_t - y_t-1 as states
        reward is calculated with given time_index
        '''
        self.computeNextStep(action=action[0],current_step=time_index)

        obs = [self.theta, self.theta - self.theta_1]
        reward = self.getReward(time_index)

        if self.enable_actuator_dynamics == False:
            return obs, reward, self.Y, self.T
        elif self.enable_actuator_dynamics == True:
            return obs, reward, self.ad['_z'], self.Y, self.T

    def reset(self):
        self.resetValues()
        return [0, 0]


    def render(self):
        # print("This code doesn't render anything.")
        pass

if __name__ == "__main__":
    cs = ControlSystem(enable_actuator_dynamics=False)
    cs.getYRef(display=True)