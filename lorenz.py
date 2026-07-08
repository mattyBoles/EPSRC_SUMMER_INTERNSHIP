import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class LorenzGenerator():

    '''
    Class to generate trajectories of a Lorenz attrcator, based on Lorenz63, the simplified weather model describing fluid motion and heat:
    
    dx/dt = sigma(y - x)
    dy/dt = x(rho - z) - y
    dz/dt = xy - beta*z

    Here, x is the intensity of the fluid's motion.
    y is the temperature difference between the rising and falling fluid currents.
    z is the distortion of the vertical temperatuer profile from a stright line.

    '''
    def __init__(self,
                 sigma: float = 10,
                 rho: float = 28,
                 beta: float = 8/3):
        
        '''
        Initiates Lorenz generator class for given parameters:

        Inputs:
            sigma (float): Prandtl number, the ratio of a fluid's momentum to its heat diffusion.
            rho (float): Rayleigh number, the temperature difference between top and bottom fluid layer.
            beta (float): The ratio of the width to the height of the convection cell.
        '''
        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.dxdt = lambda x, y, z: self.sigma * (y - x)
        self.dydt = lambda x, y, z: (x * (self.rho - z)) - y
        self.dzdt = lambda x, y, z: (x * y) - (self.beta * z)

    def calc_derivatives(self,
            x_: tuple[float, float, float]) -> tuple[float, float, float]:
        ''' 
        Calculates and returns xdot_, the vector time derivative of the vector [x,y,z]

        Inputs:
            x_ (tuple[float, float, float]): The input vector, [x, y, z]
        '''
        
        x_dot = self.dxdt(x_[0], x_[1], x_[2])
        y_dot = self.dydt(x_[0], x_[1], x_[2])
        z_dot = self.dzdt(x_[0], x_[1], x_[2])

        return np.array([x_dot, y_dot, z_dot])
    
    @staticmethod
    def rk4(f: Callable,
            x_: tuple[float, float, float],
            h: float) -> tuple[float, float, float]:
        
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth.
        '''
        k1 = f(x_)

        k2 = f(x_ + (k1 * h/2))

        k3 = f(x_ + (k2 * h/2))

        k4 = f(x_ + (k3*h))

        x_ = x_ + h*(k1 + 2*k2 + 2*k3 + k4)/6


        return x_
    

    def J(self,
          x_: np.ndarray) -> np.ndarray:
        
        x, y, z = x_[0], x_[1], x_[2]
        return np.array([[-1*self.sigma, self.sigma, 0],
                           [self.rho - z, -1, -1*x],
                           [y, x, -1*self.beta]])
    @staticmethod
    def rk4_matrix_and_x(f: Callable,
                         J: Callable,
                         x_: np.ndarray,
                         U: np.ndarray,
                         h: float) -> tuple[np.ndarray, np.ndarray]:
        
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth.
        '''

        k1_x = f(x_)
        k1_U = J(x_) @ U

        k2_x = f(x_ + (k1_x * h/2))
        k2_U = J(x_ + (k1_x * h/2)) @ (U + (k1_U * h/2))

        k3_x = f(x_ + (k2_x * h/2))
        k3_U = J(x_ + (k2_x * h/2)) @ (U + (k2_U * h/2))

        k4_x = f(x_ + (k3_x * h))
        k4_U = J(x_ + (k3_x * h)) @ (U + (k3_U * h))

        x_new = x_ + (h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x))
        U_new = U + (h/6 * (k1_U + 2*k2_U + 2*k3_U + k4_U))


        return x_new, U_new
    

    def generate_trajectory(self,
                            x0: tuple[float, float, float],
                            n_steps: int,
                            h: float = 1/100):
        
        x_out = np.empty([n_steps+1, 3])

        x_out[0] = x0
        x_ = x0

        for step_idx in range(1,n_steps+1):
            x_new = self.rk4(self.calc_derivatives, x_= x_, h = h)
            x_out[step_idx] = x_new
            x_ = x_new
        
        return x_out
    
    @staticmethod
    def plot(two_traj: bool,
             traj1: np.ndarray,
             traj2: Optional[np.ndarray],
             png_name: str):
        
        x, y, z = traj1[:,0], traj1[:,1], traj1[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, lw=1, alpha=0.5)          # line plot of the path
        # ax.scatter(x, y, z, s=2)      # or scatter, if you prefer points

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz')
        if two_traj:
            x, y, z = traj2[:,0], traj2[:,1], traj2[:,2]
            ax.plot(x, y, z, lw=1, c='r', alpha=0.5)          # line plot of the path
        plt.savefig(png_name)
        #plt.show()
        


        
    def find_lyapunov_exponent(self,
                      x0: tuple[float, float, float],
                      h: float, 
                      d0: float, 
                      t: int,
                      n_samples):
        
        #Let settle
        x_ = self.generate_trajectory(x0, n_steps=5000, h=h)
        x0 = x_[-1]

        tally = 0
        x0_pertubated = x0 + np.array([d0, 0, 0])
        for i in range(n_samples):
            x_ = self.generate_trajectory(x0, n_steps=t, h=h)
            x_pertubated = self.generate_trajectory(x0_pertubated, n_steps=t, h=h)

            x0 = x_[-1]
            x0_pertubated = x_pertubated[-1]

            delta = np.linalg.norm(x0 - x0_pertubated)
            lambda1 =np.log(delta / d0)
            tally += lambda1

            delta_vec = x0_pertubated - x0
            delta = np.linalg.norm(delta_vec)

            delta_vec *= d0 / delta
            x0_pertubated = x0 + delta_vec
        tally/=(n_samples*h*t)
        #print(f'Lyapunov Exponent: {tally}')

        return tally
    

    def find_lyapunov_spectrum(self,
                               x: np.ndarray,
                               transient_steps: int,
                               trajectory_steps: int,
                               QR_steps: int,
                               h: float):


        #Initialise U
        Q = np.identity(3)
        lambda_ = np.empty((3,0))
        singular_values = []
        l_vectors = []
        r_vectors = []
        zdot = []

        xout = np.empty((0,3))
        for i in range(transient_steps):
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x_ = x, U = Q,  h=h)

            if i % QR_steps == 0:
                U, S, Vt = np.linalg.svd(Q)
                Q, R = np.linalg.qr(Q)
                

        for i in range(trajectory_steps):
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x_ = x, U = Q,  h=h)
            
            xout = np.vstack([xout, x.reshape(1,3)])
            J = self.J(x)
            U, S, Vt = np.linalg.svd(J)
            singular_values.append(S)
            l_vectors.append(U)
            r_vectors.append(Vt)
            zdot.append(self.calc_derivatives(x)[2])

            if i % QR_steps == 0:
                
                Q, R = np.linalg.qr(Q)
                lambda_ = np.hstack([lambda_, np.array([[np.log(np.abs(R[0,0]))/(h*QR_steps)],
                            [np.log(np.abs(R[1,1]))/(h*QR_steps)],
                            [np.log(np.abs(R[2,2]))/(h*QR_steps)]])])
        
        Lyapunov_spectrum = np.mean(lambda_, axis=1)

        print(f"Lambda1: {Lyapunov_spectrum[0]}\nLambda2: {Lyapunov_spectrum[1]}\nLambda3: {Lyapunov_spectrum[2]}\n")

        return zdot, singular_values, l_vectors, r_vectors, xout
    

        
if __name__ == '__main__':
    
    generator = LorenzGenerator()

    #Dominant Lyapunov Exponent
    # avg = 0
    # for i in range(100):
    #     x0 =np.array([
    #     np.random.uniform(-15, 15),
    #     np.random.uniform(-20, 20),
    #     np.random.uniform(5, 40)
    #     ])
    #     tally = genarator.find_lyapunov(x0=x0, h=0.01, d0=1e-8, t=10, n_samples = 10000)
    #     avg += tally
    # avg /=100
    # print(f'Lyapunov Exponent: {avg}')


    #Lyapunov Spectrum


    zdot, singular_values, l_vectors, r_vectors, xout = generator.find_lyapunov_spectrum(x=np.array([1,1,0]), transient_steps=5000, trajectory_steps=10000, QR_steps=10, h=0.01)
    


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create line segments
    points = xout.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Colour by sv_2
    #sv2 = np.array([s[1] for s in singular_values])
    zdot = np.array(np.abs(zdot))

    norm = plt.Normalize(zdot.min(), zdot.max())

    lc = Line3DCollection(segments, cmap='inferno', norm=norm)
    lc.set_array(zdot)
    ax.add_collection3d(lc)

    ax.set_xlim(xout[:,0].min(), xout[:,0].max())
    ax.set_ylim(xout[:,1].min(), xout[:,1].max())
    ax.set_zlim(xout[:,2].min(), xout[:,2].max())

    plt.colorbar(lc, ax=ax, label='sv_2')
    plt.title('Lorenz attractor coloured by sv_2')
    plt.show()





    




