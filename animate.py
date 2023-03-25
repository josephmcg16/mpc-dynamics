"""This module contains functions to animate and plot the n-pendulum system."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from pendulum import nBodyPendulum


def animate(pendulum: nBodyPendulum) -> FuncAnimation:
    """
    Create an animation of the n-pendulum system

    Parameters
    ----------
    pendulum : nBodyPendulum
        The n-pendulum system to animate.
    
    Returns
    -------
    FuncAnimation
        The animation of the n-pendulum system.
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-pendulum.length * pendulum.num_of_masses,
                pendulum.length * pendulum.num_of_masses)
    ax.set_ylim(-pendulum.length * pendulum.num_of_masses,
                pendulum.length * pendulum.num_of_masses)
    ax.set_aspect("equal", adjustable="box")
    ax.grid()

    lines = [plt.Line2D([], [], color="blue", lw=1)
             for _ in range(pendulum.num_of_masses)]
    points = [plt.Line2D([], [], color="red", marker="o", markersize=8/pendulum.num_of_masses)
              for _ in range(pendulum.num_of_masses)]

    def init():
        """Initialize the animation
        
        Returns
        -------
        list
            The list of lines and points to animate.
        """
        for line in lines:
            ax.add_line(line)
        for point in points:
            ax.add_artist(point)
        return lines + points

    def update(frame):
        """
        Update the animation
        
        Parameters
        ----------
        frame : int
            The current frame of the animation.

        Returns
        -------
        list
            The list of lines and points to animate.
        """
        x_data = np.hstack(([0], pendulum.X[:, frame]))
        y_data = np.hstack(([0], pendulum.Y[:, frame]))

        for i, line in enumerate(lines):
            line.set_data(x_data[i:i+2], y_data[i:i+2])

        for i, point in enumerate(points):
            point.set_data(x_data[i+1], y_data[i+1])

        return lines + points

    ani = FuncAnimation(fig, update, frames=len(pendulum.t_eval),
                        init_func=init, blit=True)

    return ani


def plot(pendulum: nBodyPendulum) -> go.Figure:
    """
    Create a plot of the n-pendulum system

    Parameters
    ----------
    pendulum : nBodyPendulum
        The n-pendulum system to plot.

    Returns
    -------
    go.Figure
        The plot of the n-pendulum system.
    """
    fig = go.Figure()
    for i, (x, y) in enumerate(zip(pendulum.X, pendulum.Y)):
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name=f"Mass {i+1}")
        )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor="#1E1E1E"
    )
    return fig
