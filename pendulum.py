"""A module for solving the equations of motion for an n-pendulum system."""
import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp
from tqdm import tqdm


class nBodyPendulum:
    def __init__(self, num_of_masses: int, mass: float, length: float, t_eval: np.ndarray, thetas_initial: np.ndarray, omegas_initial: np.ndarray, g: float = 9.81) -> None:
        """
        Initialize the n-pendulum system.

        Parameters
        ----------
        num_of_masses : int
            The number of masses in the n-pendulum system.
        mass : float
            The mass of each mass in the n-pendulum system.
        length : float
            The length of each mass in the n-pendulum system.
        t_eval : np.ndarray
            The time points at which to solve for the n-pendulum system.
        thetas_initial : np.ndarray
            The initial angles of each mass in the n-pendulum system.
        omegas_initial : np.ndarray
            The initial angular velocities of each mass in the n-pendulum system.
        g : float, optional
            The gravitational acceleration, by default 9.81
        """
        self.num_of_masses = num_of_masses
        self.t_eval = t_eval
        self.thetas_initial = thetas_initial
        self.omegas_initial = omegas_initial
        self.mass = mass
        self.length = length
        self.g = g

    def solve(self, progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the equations of motion for the n-pendulum system.

        Parameters
        ----------
        progress : bool, optional
            Whether to show a progress bar, by default False
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The x and y coordinates of the n-pendulum system.
        """
        t_span = self.t_eval[0], self.t_eval[-1]
        y0 = np.hstack([self.thetas_initial, self.omegas_initial])

        mass_matrix = self.mass * self.length ** 2 * \
            np.array([[self.num_of_masses - max(i, j)
                     for j in range(self.num_of_masses)] for i in range(self.num_of_masses)])

        force_matrix = self.mass * self.length * self.g * \
            np.diag(range(self.num_of_masses, 0, -1))

        def equations_of_motion(t, y, n, mass_matrix, force_matrix, progress_bar=None):
            """
            The equations of motion for the n-pendulum system.

            Parameters
            ----------
            t : float
                The current time.
            y : np.ndarray
                The current state of the n-pendulum system.
            n : int
                The number of masses in the n-pendulum system.
            mass_matrix : np.ndarray
                The mass matrix for the n-pendulum system.
            force_matrix : np.ndarray
                The force matrix for the n-pendulum system.
            progress_bar : tqdm, optional
                The progress bar to update, by default None

            Returns
            -------
            np.ndarray
                The derivative of the current state of the n-pendulum system.
            """
            if progress_bar is not None:
                progress = (t - t_span[0]) / (t_span[1] - t_span[0]) * 100
                progress_bar.update(int(progress) - progress_bar.n)
            theta_vec = y[:n]
            omega_vec = y[n:]
            omega_dot_vec = np.linalg.solve(- mass_matrix,
                                            force_matrix).dot(theta_vec)
            return np.hstack([omega_vec, omega_dot_vec])

        if progress:
            with tqdm(total=100, desc="Solving ODE", unit="%", ncols=80, leave=True) as ode_progress_bar:
                sol = solve_ivp(equations_of_motion, t_span, y0,
                                t_eval=self.t_eval, args=(self.num_of_masses, mass_matrix, force_matrix, ode_progress_bar), rtol=1e-5, atol=1e-5)
        else:
            sol = solve_ivp(equations_of_motion, t_span, y0,
                            t_eval=self.t_eval, args=(self.num_of_masses, mass_matrix, force_matrix, None), rtol=1e-5, atol=1e-5)

        self.thetas = sol.y[:self.num_of_masses]
        self.omegas = sol.y[self.num_of_masses:]
        self.X = self.length * np.sin(self.thetas).cumsum(axis=0)
        self.Y = - self.length * np.cos(self.thetas).cumsum(axis=0)

        self.states_rot = np.vstack([self.thetas, self.omegas])
        self.states_cart = np.vstack([self.X, self.Y])
        return self.X, self.Y
