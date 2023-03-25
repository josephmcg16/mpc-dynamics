import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

g = 9.81  # acceleration due to gravity, m/s^2
LENGTH = 1.0  # length of the pendulum arms, m
MASS = 1.0  # mass of the pendulum bobs, kg


def triple_pendulum_eom(t, y, mass, length):
    theta_vec = y[:3]
    omega_vec = y[3:]

    M = mass * length ** 2 * np.array([[3, 2, 1], [2, 2, 1], [1, 1, 1]])
    K = mass * length * g * np.diag([3,2,1])

    omega_dot_vec = np.linalg.solve(- M, K).dot(theta_vec)

    return np.hstack([omega_vec, omega_dot_vec])


t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

theta_vec_initial = np.array([120, 30, 60]) * np.pi / 180
omega_vec_initial = np.array([10, 0, 0])
y0 = np.hstack([theta_vec_initial, omega_vec_initial])

sol = solve_ivp(triple_pendulum_eom, t_span, y0, t_eval=t_eval, args=(MASS, LENGTH))

theta1, theta2, theta3, omega1, omega2, omega3 = sol.y
x1 = LENGTH * np.sin(theta1)
y1 = -LENGTH * np.cos(theta1)
x2 = x1 + LENGTH * np.sin(theta2)
y2 = y1 - LENGTH * np.cos(theta2)
x3 = x2 + LENGTH * np.sin(theta3)
y3 = y2 - LENGTH* np.cos(theta3)

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")
ax.grid()

lines = [plt.Line2D([], [], color="blue", lw=1) for _ in range(4)]
points = [plt.Line2D([], [], color="red", marker="o", markersize=8)
          for _ in range(3)]


def init():
    for line in lines:
        ax.add_line(line)
    for point in points:
        ax.add_artist(point)
    return lines + points


def update(frame):
    x = [0, x1[frame], x2[frame], x3[frame]]
    y = [0, y1[frame], y2[frame], y3[frame]]

    for i, line in enumerate(lines):
        line.set_data(x[i:i+2], y[i:i+2])

    for i, point in enumerate(points):
        point.set_data(x[i+1], y[i+1])

    return lines + points


ani = FuncAnimation(fig, update, frames=len(t_eval),
                    init_func=init, blit=True, interval=20)

# Uncomment the following line to save the animation as a GIF file
# ani.save("triple_pendulum.gif", writer="imagemagick", fps=50)

plt.show()
