import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from tqdm import tqdm


class n_pendulum:
    def __init__(self, num_of_masses: int, mass: float, length: float, t_eval: np.ndarray, thetas_initial: np.ndarray, omegas_initial: np.ndarray, g: float = 9.81) -> None:
        """
        Initialize the n-pendulum system.
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
        """
        t_span = self.t_eval[0], self.t_eval[-1]
        y0 = np.hstack([self.thetas_initial, self.omegas_initial])

        mass_matrix = self.mass * self.length ** 2 * \
            np.array([[self.num_of_masses - max(i, j)
                     for j in range(self.num_of_masses)] for i in range(self.num_of_masses)])

        force_matrix = self.mass * self.length * self.g * \
            np.diag(range(self.num_of_masses, 0, -1))

        def equations_of_motion(t, y, n, mass_matrix, force_matrix, progress_bar=None):
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

    def animate(self) -> FuncAnimation:
        """
        Create an animation of the n-pendulum system.
        """
        real_time = self.t_eval[-1] / len(self.t_eval) * 1000

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.length * self.num_of_masses,
                    self.length * self.num_of_masses)
        ax.set_ylim(-self.length * self.num_of_masses,
                    self.length * self.num_of_masses)
        ax.set_aspect("equal", adjustable="box")
        ax.grid()

        lines = [plt.Line2D([], [], color="blue", lw=1)
                 for _ in range(self.num_of_masses)]
        points = [plt.Line2D([], [], color="red", marker="o", markersize=8/self.num_of_masses)
                  for _ in range(self.num_of_masses)]

        def init():
            for line in lines:
                ax.add_line(line)
            for point in points:
                ax.add_artist(point)
            return lines + points

        def update(frame):
            x_data = np.hstack(([0], self.X[:, frame]))
            y_data = np.hstack(([0], self.Y[:, frame]))

            for i, line in enumerate(lines):
                line.set_data(x_data[i:i+2], y_data[i:i+2])

            for i, point in enumerate(points):
                point.set_data(x_data[i+1], y_data[i+1])

            return lines + points

        ani = FuncAnimation(fig, update, frames=len(self.t_eval),
                            init_func=init, blit=True)

        return ani

    def plot(self) -> go.Figure:
        fig = go.Figure()
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', name=f"Mass {i+1}")
            )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor="#1E1E1E"
        )
        return fig


if __name__ == '__main__':
    NUM_OF_MASSES = 10
    MASS = 1
    LENGTH = 1
    theta0 = np.ones(NUM_OF_MASSES) * np.pi + 1e-6
    omega0 = np.zeros(NUM_OF_MASSES)

    t_final = 10  # seconds
    FPS = 60

    pendulum = n_pendulum(NUM_OF_MASSES, MASS, LENGTH,
                          np.linspace(0, NUM_OF_MASSES, t_final * FPS), theta0, omega0)
    pendulum.solve(progress=True)
    ani = pendulum.animate()
    ani.save('pendulum.gif', fps=FPS)
