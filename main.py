import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from tqdm import tqdm


class n_pendulum:
    def __init__(self, num_of_masses, mass, length, t_eval, thetas_initial, omegas_initial, g=9.81) -> None:
        self.num_off_masses = num_of_masses
        self.t_eval = t_eval
        self.thetas_initial = thetas_initial
        self.omegas_initial = omegas_initial
        self.mass = mass
        self.length = length
        self.g = g
        pass
    
    def solve(self):
        t_span = self.t_eval[0], self.t_eval[-1]
        y0 = np.hstack([self.thetas_initial, self.omegas_initial])

        M = self.mass * self.length ** 2 * \
            np.array([[self.num_off_masses - max(i, j)
                     for j in range(self.num_off_masses)] for i in range(self.num_off_masses)])

        K = self.mass * self.length * self.g * \
            np.diag(range(self.num_off_masses, 0, -1))

        def eom(t, y, n, M, K, progress_bar):
            progress = (t - t_span[0]) / (t_span[1] - t_span[0]) * 100
            progress_bar.update(int(progress) - progress_bar.n)
            theta_vec = y[:n]
            omega_vec = y[n:]
            omega_dot_vec = np.linalg.solve(- M, K).dot(theta_vec)
            return np.hstack([omega_vec, omega_dot_vec])
        
        with tqdm(total=100, desc="Solving ODE", unit="%", ncols=80, leave=True) as ode_progress_bar:
            sol = solve_ivp(eom, t_span, y0,
                            t_eval=self.t_eval, args=(self.num_off_masses, M, K, ode_progress_bar), rtol=1e-5, atol=1e-5)

        self.thetas = sol.y[:self.num_off_masses]
        self.omegas = sol.y[self.num_off_masses:]
        self.X = self.length * np.sin(self.thetas).cumsum(axis=0)
        self.Y = - self.length * np.cos(self.thetas).cumsum(axis=0)

        self.states_rot = np.vstack([self.thetas, self.omegas])
        self.states_cart = np.vstack([self.X, self.Y])
        return self.X, self.Y
    
    def animate(self, speed=1.0):
        real_time = self.t_eval[-1] / len(self.t_eval) * 1000

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.length * self.num_off_masses,
                    self.length * self.num_off_masses)
        ax.set_ylim(-self.length * self.num_off_masses,
                    self.length * self.num_off_masses)
        ax.set_aspect("equal", adjustable="box")
        ax.grid()

        lines = [plt.Line2D([], [], color="blue", lw=1)
                 for _ in range(self.num_off_masses)]
        points = [plt.Line2D([], [], color="red", marker="o", markersize=8/self.num_off_masses)
                  for _ in range(self.num_off_masses)]

        def init():
            for line in lines:
                ax.add_line(line)
            for point in points:
                ax.add_artist(point)
            return lines + points

        def update(frame):
            x_data = np.hstack(([0], self.x[:, frame]))
            y_data = np.hstack(([0], self.x[:, frame]))

            for i, line in enumerate(lines):
                line.set_data(x_data[i:i+2], y_data[i:i+2])

            for i, point in enumerate(points):
                point.set_data(x_data[i+1], y_data[i+1])

            return lines + points

        ani = FuncAnimation(fig, update, frames=len(self.t_eval),
                            init_func=init, blit=True, interval=real_time * speed)

        plt.show()
    
    def plot(self):
        fig = go.Figure()
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', name=f"Mass {i+1}")
            )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor="#1E1E1E"
        )
        fig.show()
