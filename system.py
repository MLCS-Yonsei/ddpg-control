from control import *
from control.matlab import *
import matplotlib.pyplot as plt 

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
    def __init__(self):
        self.T = self.getTimeVector(duration=16,ts=0.02)
        self.resetValues()

        # Setting for DDPG
        self.timestep_limit = len(self.T)

        self.observation_space = spaces.Box(low=0, high=1,
                                    shape=(2,))
        self.action_space = spaces.Box(low=-0., high=1.,
                                    shape=(1,))

        print(self.action_space.low)
   
    def resetValues(self):
        self._y = 0.
        self.Y = []
        self.Y_ref = self.getYRef(display=False)

        # Impulse Response
        self.theta_2 = 0
        self.theta_1 = 0
        self.u_2 = 0
        self.u_1 = 1

        self.theta = None

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

    def getTimeVector(self, duration=16, ts=0.02):
        return np.arange(0, duration, ts)

    def computeNextStep(self, zeta):
        self.Y.append(self._y)

        co = self.computeSystem(zeta=zeta)

        self.theta = - co['a1'] * self.theta_1 - co['a2'] * self.theta_2 + co['b1'] * self.u_1 + co['b2'] * self.u_2
        self._y += self.theta - self.theta_1

        self.theta_2 = self.theta_1
        self.theta_1 = self.theta
        self.u_2 = self.u_1
        try:
            self.u_1 = self.u
        except:
            # Impulse Response
            self.u_1 = 0

    def zetaFunction(self, time_vector, index):
        _x = index * (np.pi / len(time_vector))
        return np.cos(_x) / 4 + 0.75

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
        Y_ref = []

        # impulse
        th1 = 0
        th2 = 0
        u1 = 1
        u2 = 0

        _y_ref = 0.
        for time_index, _t in enumerate(self.T):
            Y_ref.append(_y_ref)

            # Compute
            zeta = self.zetaFunction(time_vector=self.T, index=time_index)
            co = self.computeSystem(zeta=zeta)

            theta = - co['a1'] * th1 - co['a2'] * th2 + co['b1'] * u1 + co['b2'] * u2
            _y_ref += theta - th1

            th2 = th1
            th1 = theta
            u2 = u1
            try:
                u2 = u
            except:
                # Impulse Response
                u1 = 0
            
        if display is True:
            plt.step(self.T,self.y)
            plt.grid()
            plt.xlabel('t') 
            plt.ylabel('y')
            plt.show()

        return Y_ref

    def getReward(self, time_index, y):
        return -((self.Y_ref[time_index] - y) ** 2)

    def step(self, action, time_index):
        '''
        Computes system with given action
        returns y_t and y_t-1 as states
        reward is calculated with given time_index
        '''
        self.computeNextStep(zeta=action[0])
        obs = [self.theta, self.theta - self.theta_1]
        reward = self.getReward(time_index, self.theta)
        return obs, reward

    def reset(self):
        self.resetValues()
        return [0, 0]


    def render(self):
        # print("This code doesn't render anything.")
        pass

if __name__ == "__main__":
    cs = ControlSystem()
    cs.getYRef(display=True)