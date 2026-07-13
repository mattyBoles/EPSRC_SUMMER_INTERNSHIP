import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from dataclasses import dataclass



class LorenzGenerator():

    '''
    Class to generate trajectories of a Lorenz attrcator, based on Lorenz63, the simplified weather model describing fluid motion and heat:
    
    dx/dt = sigma(y - x)
    dy/dt = x(rho - z) - y
    dz/dt = xy - beta*z

    Here, x is the intensity of the fluid's motion.
    y is the temperature difference between the rising and falling fluid currents.
    z is the distortion of the vertical temperature profile from a stright line.

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
            x: np.ndarray) -> np.ndarray:
        ''' 
        Calculates and returns xdot_, the vector time derivative of the vector [x,y,z]

        Inputs:
            x (tuple[float, float, float]): The input vector, [x, y, z]
        
        Returns:
            np.ndarray: The instantanious time derivatives at the input position, in shape [3,]
        '''
        
        x_dot = self.dxdt(x[0], x[1], x[2])
        y_dot = self.dydt(x[0], x[1], x[2])
        z_dot = self.dzdt(x[0], x[1], x[2])

        return np.array([x_dot, y_dot, z_dot])
    
    @staticmethod
    def rk4(f: Callable,
            x: np.ndarray,
            h: float) -> np.ndarray:
        
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth.
        https://www.geeksforgeeks.org/dsa/runge-kutta-4th-order-method-solve-differential-equation/

        Inputs: 
            f (Callable): The derivative function of the system.
            x (np.ndarray): The innput posotin vector, shape [x, y, z]
            h (float): The timestep length.
        
        Returns:
            np.ndarray: The timestepped position vector.
        '''
        k1 = f(x)

        k2 = f(x + (k1 * h/2))

        k3 = f(x + (k2 * h/2))

        k4 = f(x + (k3*h))

        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6

        return x
    

    def J(self,
          x_: np.ndarray) -> np.ndarray:
        '''
        Calculates the instantanious Jacobian of the input vector.
        '''
        
        x, y, z = x_[0], x_[1], x_[2]
        return np.array([[-1*self.sigma, self.sigma, 0],
                           [self.rho - z, -1, -1*x],
                           [y, x, -1*self.beta]])
    @staticmethod
    def rk4_matrix_and_x(f: Callable,
                         J: Callable,
                         x: np.ndarray,
                         U: np.ndarray,
                         h: float) -> tuple[np.ndarray, np.ndarray]:
        
        '''
        Runge-Kutta 4th order time stepping scheme, to be used as ground truth. https://www.geeksforgeeks.org/dsa/runge-kutta-4th-order-method-solve-differential-equation/
        But we also calculate the time stepped M, the tangent propogator. We can then find the instantaneous Singualr Values, which then inform our Lyapunov Spectrum.
        
        Inputs:
            f (Callable): The derivative function of x, in this case the Lorenz63 system, self.calc_derivatives, which retruns dx_/dt, at the input vector.
            J (Callable): The Jacobian function of the system, which returns a matrix with partial derivatives calculated at x.
            x_ (np.ndarray): The current position vector of the system.
            U (np.ndarray): The current tangent propogater of the system.
            h (float): The delta t, timestep.
        
        Returns:
            tuple[np.ndarray, np.ndarray]:
                x_new: The new position vector after timestepping.
                U_new: The new tangent propogater after timestepping.
        '''

        k1_x = f(x)
        k1_U = J(x) @ U

        k2_x = f(x + (k1_x * h/2))
        k2_U = J(x + (k1_x * h/2)) @ (U + (k1_U * h/2))

        k3_x = f(x + (k2_x * h/2))
        k3_U = J(x + (k2_x * h/2)) @ (U + (k2_U * h/2))

        k4_x = f(x + (k3_x * h))
        k4_U = J(x + (k3_x * h)) @ (U + (k3_U * h))

        x_new = x + (h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x))
        U_new = U + (h/6 * (k1_U + 2*k2_U + 2*k3_U + k4_U))


        return x_new, U_new
    

    def generate_trajectory(self,
                            x0: np.ndarray,
                            n_steps: int,
                            h: float = 1/100) -> np.ndarray:
        
        '''
        Generates a trajetcory by tiemstepping in rk4.

        Input:
            x0 [np.ndarray]: The starting position vector, np.array([x, y, z])
            n_steps (int): The number of steps to iterate.
            h (float): The lengfth of timesteps.
        
        Returns:
            np.ndarray: The trajectory, shape [n_steps + 1, 3]
        '''
        
        x_out = np.empty([n_steps+1, 3])
        x_out[0] = x0

        for step_idx in range(1,n_steps+1):
            x_out[step_idx] = self.rk4(self.calc_derivatives, x = x_out[step_idx - 1], h = h)
        
        return x_out
    
    @staticmethod
    def plot(png_name: str,
             traj1: np.ndarray,
             traj2: Optional[np.ndarray] = None) -> None:

        '''
        Plots 1 or 2 trajectories of Lorenz.

        Inputs:
            png_name (str): The out file path that the png will be saves as.
            traj1 (np.ndarray): The first trajectory to plot, in shape (n_steps, 3) 
            traj2 (Optional[np.ndarray]): The second trajectory to plot, same shape as traj1.
        
        '''        
        x, y, z = traj1[:,0], traj1[:,1], traj1[:,2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, lw=1, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz')

        if traj2 is not None:
            x, y, z = traj2[:,0], traj2[:,1], traj2[:,2]
            ax.plot(x, y, z, lw=1, c='r', alpha=0.5)      
        plt.savefig(png_name)
        plt.close('all')
        


        
    def find_lyapunov_exponent(self,
                               x: np.ndarray,
                               n_transient: int = 5000,
                               n_steps: int = 10000,
                               d0: np.ndarray = None, 
                               re_norm_steps: int = 10, 
                               h: float = 0.01) -> float:
        '''
        Estimates the Dominant Lyapunov Exponent of the system:
        1. Integrate over transient period and throw away.
        2. Pertubate x0 by a small margin, d0.
        3. Integrate over  small number of steps, t.
        4. Find an estaimet of lambda1, lambda = ln(delta/d0) [/ (t*h), bu we do this at the end for compute].
        5. Normalise the delta to |d0|, but in the same direction as delta. We do this as we need a small enough pertuabtion to assume linearity, and too many timesteps and it's no lomnger close enough.
        6. Repeat

        Inputs:
            x (np.ndarray): Starting posiiton vector, [x,y,z].
            n_transient (int): Number of timesteps to disregard, to ensure accumulating lambdas are on the attractor.
            n_steps (int): Number of timesteps total to intergrate over.
            d0 (float): Starting pertubation length.
            re_norm_steps: (int): Number of timesteps of each renormalisation cycle.
            h (float): Timestep length.

        Returns:
            float: Estimate of dominant Lyapunov Exponent.

        '''
        if d0 is None:
            d0 = np.array([1e-8, 0.0, 0.0])
        #Let settle
        for _ in range(n_transient):
            x = self.rk4(f = self.calc_derivatives, x = x, h = h)

        lambda_cum = 0
        count = 0

        x_pertubated = x + d0

        d0_abs = np.linalg.norm(d0)

        for i in range(n_steps):
            x = self.rk4(f=self.calc_derivatives, x = x, h = h)
            x_pertubated = self.rk4(f = self.calc_derivatives, x = x_pertubated, h = h)

            if (i+1) % re_norm_steps == 0:
            
                delta_vec = (x_pertubated - x)
                delta_abs = np.linalg.norm(delta_vec)

                if delta_abs == 0:
                    raise ValueError("Perturbation collapsed to zero")

                lambda_cum += np.log(delta_abs / d0_abs)

                delta_vec *= d0_abs / delta_abs
                x_pertubated = x + delta_vec
                count += 1

        return lambda_cum/(count*h*re_norm_steps)
    

    def find_lyapunov_spectrum(self,
                               x: np.ndarray,
                               transient_steps: int = 5000,
                               trajectory_steps: int = 10000,
                               QR_steps: int = 10,
                               h: float = 0.01):


        #Initialise
        Q = np.identity(3)
        n_qr_samples = trajectory_steps // QR_steps
        lambda_ = np.empty((3, n_qr_samples))
        singular_values, l_vectors, r_vectors, zdot = [], [], [], []
        count = 0

        for i in range(transient_steps):
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Q,  h=h)

            if (i+1) % QR_steps == 0:
                Q, R = np.linalg.qr(Q)
                

        for i in range(trajectory_steps):
            x, Q = self.rk4_matrix_and_x(f = self.calc_derivatives, J = self.J, x = x, U = Q,  h=h)
            
            J = self.J(x)
            J = np.eye(3) + h*J
            U, S, Vt = np.linalg.svd(J)

            singular_values.append(S)
            l_vectors.append(U)
            r_vectors.append(Vt)
            zdot.append(self.calc_derivatives(x)[2])

            if (i+1) % QR_steps == 0:
                
                Q, R = np.linalg.qr(Q)
                lambda_[:, count] = np.log(np.abs(np.diag(R))) / (h * QR_steps)
                count += 1
        
        lambda_ = lambda_[:, :count]
        lyapunov_spectrum = np.mean(lambda_, axis=1)



        return {
            'lyapunov_spectrum':lyapunov_spectrum,
            'zdot':zdot,
            'singular_values':singular_values,
            'l_vectors':l_vectors,
            'r_vectors':r_vectors,
        }
    

        
if __name__ == '__main__':
    
    generator = LorenzGenerator()

    # #Dominant Lyapunov Exponent
    d_lyapunov_exp = generator.find_lyapunov_exponent(x=np.array([1,1,0]))
    print(f'Dominant Lyapunov Exponent: {d_lyapunov_exp}')

    #Lyapunov Spectrum
    lyapunov_spectrum = generator.find_lyapunov_spectrum(x=np.array([1,1,0]))['lyapunov_spectrum']
    print(f"Lambda1: {lyapunov_spectrum[0]}\nLambda2: {lyapunov_spectrum[1]}\nLambda3: {lyapunov_spectrum[2]}\n")


    




