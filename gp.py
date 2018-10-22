from math import exp
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

def getTimeVector(duration=16, ts=0.02):
    return np.arange(0, duration, ts)

def createRandomFunction(T, display=False):
    def rbf_kernel(x1, x2, variance = 1, _lambda = 12):

        return exp(-1 * (((x1-x2) ** 2 ) / (_lambda ** 2)) / (variance))

    def gram_matrix(xs):
        return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

    xs = T
    mean = [0 for x in xs]
    gram = gram_matrix(xs)

    plt_vals = []
    ys = abs(np.random.multivariate_normal(mean, gram))
    plt_vals.extend([xs, ys, "k"])

    if display == True:
        plt.plot(*plt_vals)
        plt.show()
    
    return ys

if __name__ == '__main__':
    ys = createRandomFunction(T=getTimeVector(),display=True)
    print(ys)
    print(ys.shape)
