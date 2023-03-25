"""Main file for the n-body pendulum simulation."""
import numpy as np
from animate import animate, plot
from pendulum import nBodyPendulum

if __name__ == '__main__':
    NUM_OF_MASSES = 100
    MASS = 1  # kg
    LENGTH = 1  # m
    theta0 = np.ones(NUM_OF_MASSES) * np.pi + 1e-6  # radians
    omega0 = np.zeros(NUM_OF_MASSES)  # radians/second
    t_final = 10  # seconds
    FPS = 30

    pendulum = nBodyPendulum(NUM_OF_MASSES, MASS, LENGTH,
                             np.linspace(0, t_final, t_final * FPS), theta0, omega0)
    pendulum.solve(progress=True)
    ani = animate(pendulum)
    ani.save('pendulum.gif', fps=FPS)
