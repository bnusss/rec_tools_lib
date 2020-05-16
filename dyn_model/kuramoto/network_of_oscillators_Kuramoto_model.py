import numpy as np
from scipy.integrate import odeint


# from ddeint import ddeint

# for bugfix ModuleNotFoundError: No module named 'ddeint'
##################

# Copied from http://codegist.net/snippet/python/ddeintpy_julianeagu_python
# Obs: This DDE integrator uses linear interpolation.
# I still need to determine the convergency order of this method...

# REQUIRES PACKAGES Numpy AND Scipy INSTALLED
import numpy as np
import scipy.integrate
import scipy.interpolate

class ddeVar:
    """ special function-like variables for the integration of DDEs """

    def __init__(self,g,tc=0):
        """ g(t) = expression of Y(t) for t<tc """

        self.g = g
        self.tc= tc
        # We must fill the interpolator with 2 points minimum
        self.itpr = scipy.interpolate.interp1d(
            np.array([tc-1,tc]), # X
            np.array([self.g(tc),self.g(tc)]).T, # Y
            kind='linear', bounds_error=False,
            fill_value = self.g(tc))

    def update(self,t,Y):
        """ Add one new (ti,yi) to the interpolator """

        self.itpr.x = np.hstack([self.itpr.x, [t]])
        Y2 = Y if (Y.size==1) else np.array([Y]).T
        self.itpr.y = np.hstack([self.itpr.y, Y2])
        self.itpr.fill_value = Y
        self.itpr._y = self.itpr._reshape_yi(self.itpr.y)

    def __call__(self,t=0):
        """ Y(t) will return the instance's value at time t """

        return (self.g(t) if (t<=self.tc) else self.itpr(t))

class dde(scipy.integrate.ode):
    """ Overwrites a few functions of scipy.integrate.ode"""

    def __init__(self,f,jac=None):

        def f2(t,y,args):
            return f(self.Y,t,*args)
        scipy.integrate.ode.__init__(self,f2,jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):

        scipy.integrate.ode.integrate(self,t,step,relax)
        self.Y.update(self.t,self.y)
        return self.y

    def set_initial_value(self,Y):

        self.Y = Y #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)

def ddeint(func,g,tt,fargs=None):
    """
    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by ``func`` with history function ``g``
    and potential additional arguments for the model, ``fargs``.
    Returns the values of the solution at the times given by the array ``tt``.

    Example:
    --------

    We will solve the delayed Lotka-Volterra system defined as

    For t < 0:
    x(t) = 1+t
    y(t) = 2-t

    For t > 0:
    dx/dt =  0.5* ( 1- y(t-d) )
    dy/dt = -0.5* ( 1- x(t-d) )

    Note that here the delay ``d`` is a tunable parameter of the model.

    ---

    import numpy as np

    def model(XY,t,d):
        x, y = XY(t)
        xd, yd = XY(t-d)
        return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])

    g = lambda t : np.array([1+t,2-t]) # 'history' at t<0
    tt = np.linspace(0,30,20000) # times for integration
    d = 0.5 # set parameter d
    yy = ddeint(model,g,tt,fargs=(d,)) # solve the DDE !

    """

    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g,tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    return np.array([g(tt[0])]+[dde_.integrate(dde_.t + dt)
                                 for dt in np.diff(tt)])

############################



class KuramotoModel:
    """ Class to hold the Kuramoto Model configuration and perform numerical integration.

        It instantiates a callable object f in such a way that f(theta, t) returns the value of the Kuramoto Model
        vector field, where theta is the N-dimensional state variable of the system and t is the time variable.

        Attributes
        ----------
        N : int
            Quantity of oscillators in the network.
        w : ndarray of shape (N)
            Natural frequency of each oscillator.
        A : list of ndarray of size (N)
            Coupling graph as adjacency list, i.e., A[i] contains the neighbor list of the i-th vertex.
        d : ndarray of shape (N)
            In-degree of the each vertex of the coupling graph.
        c : float
            Coupling parameter.
        tau : float
            Overall time delay among nodes.
        d_theta : ndarray of shape (N)
            Value evaluated of the vector field.
    """

    def __init__(self, w, E, c=1, tau=0):
        """

        Parameters
        ----------
        w : ndarray of shape (N)
            Natural frequency of each oscillator.
        E : ndarray of shape (Ne,2)
            Coupling graph as directed edge list of size Ne, i.e,
            e[i] = [p, q] means that the p -> q is the i-th directed edge of the coupling graph.
        c : float
            Coupling parameter.
        tau : float
            Overall time delay among nodes.
        """
        self.N = w.size
        self.w = w
        self.A = [E[E[:, 0] == i, 1].astype(int) for i in range(0, self.N)]  # Edge list E into adjacency list A.
        self.d = [len(self.A[i]) for i in range(0, self.N)]
        self.c = c
        self.d_theta = np.zeros(self.N)
        self.tau = tau

    def __call__(self, theta, t=0):
        """Evaluate the instance of the Kuramoto Model vector field with the given attributes.

        Parameters
        ----------
        theta : ndarray of shape (N)
            The phase value of the N oscillators in the network. It must be a function if tau>0.
        t: float
            The time variable. This parameter allows compatibility with odeint and ddeint from scipy.integrate.
        tau: float
            Overall time delay among nodes. This parameter is included here to allow compatiblitly with ddeint.

        Returns
        -------
        d_theta : ndarray of shape (N)
            The evaluated value, that is, d_theta = f(theta, t)

        """
        if self.tau == 0:
            theta_t, theta_tau = theta, theta
        else:
            theta_t, theta_tau = theta(t), theta(t - self.tau)

        for i in range(0, self.N):
           
            self.d_theta[i] = self.w[i] + \
                              (self.c / self.d[i]) * sum(np.sin(theta_tau[j] - theta_t[i]) for j in self.A[i])

        return self.d_theta

    def integrate(self, theta0, tf=100, h=0.01):
        """Numerical integation of the Kuramoto Model with odeint (tau=0) or ddeint (tau>0).

        Parameters
        ----------
        theta0 : ndarray of shape (N)
            Initial condition. If tau>0, (lambda t: theta0 - t * self.w) is used as initial condition.
        tf : float
            The end of the integration interval. It starts at t=0.
        h : float
            Numerical integration step.

        Returns
        -------
        t : ndarray
            Time discretization points, t = [0, h, ..., tf-h, tf].
        theta : ndarray of shape (len(t), N)
            Numerical solution for each time in t.
        """

        t = np.arange(0, tf + h / 2, h)
        if self.tau == 0:
            theta = odeint(self, theta0, t, hmax=h)
        else:
            theta0_hist = lambda t: theta0 + t * self.w
            theta = ddeint(self, theta0_hist, t)

        return t, theta
