import numpy as np

class Transport:
    def __init__(self, u):
        # do something
        self.u = u

    def update(self, i, du, lr=1):
        self.u = self.u - lr / np.sqrt(i + 1) * du


class AffineTransport(Transport):
    def __init__(self, u):
        Transport.__init__(self, u)

    def h(self, eps):
        return eps * np.exp(self.u[0]) + np.exp(self.u[1])

    def du(self, eps):
        du = np.zeros(2)
        du[0] = eps * np.exp(self.u[0])
        du[1] = 1.0 * np.exp(self.u[1])
        return du
