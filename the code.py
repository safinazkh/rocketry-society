
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Thrust Curve
thrust = pd.read_csv('Thrust_curve.csv', skiprows=3)  # read thrust curve data
m_propellant = 52 / 1000  # mass of propellant in kg 


t_interp = interp1d(thrust['Time (s)'], thrust['Thrust (N)'], kind='linear', fill_value=(0, 0), bounds_error=False)

# Assume linear fuel burn throughout the burn time
fuel_mass_rate = m_propellant / thrust['Time (s)'].max()

# Rocket Parameters
m_dry = 0.76023  # mass of rocket without propellant but including motor casing, electronics, etc. in kg
area = np.pi * (0.029 / 2) ** 2  
chute_area = np.pi * 0.3 ** 2  # parachute area in m^2
Cd = 0.75  # drag coefficient

# Simulation Parameters
dt = 0.01  # in seconds
simulation_time = 200  # total simulation time
t = np.arange(0, simulation_time, dt)  

# Initial Conditions - create arrays and give initial values
acceleration = [0]
velocity = [0]
position = [0]
mass = [m_dry + m_propellant]
drag = [0]
chute = [False]

# Simulation
for i in range(len(t) - 1):
    F_thrust = t_interp(t[i])  # get thrust at time t[i]
    F_drag = 0.5 * 1.225 * Cd * area * velocity[i] ** 2  # drag force
    F_chute = 0.5 * 1.225 * Cd * chute_area * velocity[i] ** 2  # additional drag from chute
    F_gravity = 9.80665 * mass[i]  # force due to gravity
    if mass[i] > m_dry:  # if there is still propellant left
        F_net = F_thrust - (F_gravity + F_drag)  # net force
        chute.append(False)  # chute not deployed (just for tracking not used in simulation)
    else:  # if all propellant has been used
        if np.gradient(position)[-1] <= 0:  # if past apogee
            try:
                F_net = F_chute + F_drag - F_gravity  # include chute drag if descending
            except IndexError:
                F_net = -F_drag - F_gravity  # assume chute deployed if IndexError
            chute.append(True)  # chute deployed (just for tracking not used in simulation)
        else:
            F_net = -F_drag - F_gravity  # ascending and decelerating
            chute.append(False)  # chute not deployed (just for tracking not used in simulation)

    a = F_net / mass[i]  # calculate acceleration
    v = velocity[i] + (a * dt)  # calculate new velocity
    x = np.max([position[i] + (v * dt), 0])  # calculate new position, ensuring it doesn't go below 0
    m = np.max([mass[i] - (fuel_mass_rate * dt), m_dry])  # calculate new mass, ensuring it doesn't go below m_dry

    # append new values to arrays
    drag.append(F_drag)
    acceleration.append(a)
    velocity.append(v)
    position.append(x)
    mass.append(m)

# Find apogee, descent rate, and total flight time
try:
    apogee_index = np.where(np.gradient(position) == 0)[0][1]
    print('Time to Apogee:', t[int(np.argmax(position[:int(apogee_index)]))], 's')
    print('Time to ground: ', t[np.where(np.gradient(position) == 0)[0][1]], 's')
    print('Descent Rate: ', np.min(velocity), 'm/s')
except IndexError:
    print("Simulation did not reach apogee.")
    apogee_index = np.nan


print('Max Position:', np.max(position), 'm')

# Desired descent rate
descent_rate = 5  # m/s


chute_area_needed = np.max(np.array(drag)) / (0.5 * 1.225 * Cd * (descent_rate ** 2))
print("Chute area needed:", chute_area_needed, "m^2")

# Given weight per unit area of the parachute material (in grams per square meter)
weight_per_unit_area = 50  # grams per square meter assumption according to literature

# Calculate the total weight of the parachute
parachute_weight = weight_per_unit_area * chute_area_needed
print("Weight of the parachute:", parachute_weight, "grams")


fig, ax = plt.subplots(5, 1, dpi=150, figsize=(10, 10), sharex=True)

ax[0].plot(t, acceleration)
ax[0].axhline(np.max(acceleration), xmax=(22 / 30), color='r', linestyle='--')
ax[0].text(t[-1], np.max(acceleration), f'Max Acceleration: {np.max(acceleration):.2f} m/s$^2$', fontsize=10,
           color='r', verticalalignment='top', ha='right')
ax[0].set_ylabel('Acceleration [m/s$^2$]')

ax[1].plot(t, velocity)
ax[1].axhline(np.max(velocity), xmax=(23 / 30), color='r', linestyle='--')
ax[1].text(t[-1], np.max(velocity), f'Max Velocity: {np.max(velocity):.2f} m/s', fontsize=10, color='r',
           verticalalignment='top', ha='right')
ax[1].set_ylabel('Velocity [m/s]')

alt_feet = 1
ftm = 3.28084
ax[2].plot(t, np.array(position) * (alt_feet * ftm))
ax[2].axhline(np.max(position) * (alt_feet * ftm), xmax=(23 / 30), color='r', linestyle='--')
ax[2].axhline((2500 / ftm) * (alt_feet * ftm), xmax=23 / 30, color='g', linestyle='--')
ax[2].text(t[-1], (2500 / ftm) * (alt_feet * ftm), '2500ft Target Altitude', fontsize=10, color='g',
           verticalalignment='top', ha='right')
if alt_feet == 1:
    ax[2].set_ylabel('Position [ft]')
    ax[2].text(t[-1], np.max(position) * (alt_feet * ftm),
               f'Max Position: {(np.max(position) * (alt_feet * ftm)):.2f} ft', fontsize=10, color='r',
               verticalalignment='top', ha='right')
else:
    ax[2].text(t[-1], np.max(position) * (alt_feet * ftm),
               f'Max Position: {(np.max(position) * (alt_feet * ftm)):.2f} m', fontsize=10, color='r',
               verticalalignment='top', ha='right')
    ax[2].set_ylabel('Position [m]')

ax[3].plot(t, mass)
ax[3].set_ylabel('Mass [kg]')

ax[4].plot(t, drag)
ax[4].axhline(0.5 * 1.225 * Cd * chute_area_needed * np.max(np.array(velocity)**2), color='b', linestyle='--', label='Chute Needed')
ax[4].set_ylabel('Drag [N]')
if not np.isnan(apogee_index):
    ax[4].axvline(t[int(apogee_index)], c='purple', linestyle='--', label='Chute Deployed', alpha=0.5)
ax[4].legend(frameon=False)

ax[4].set_xlabel('Time [s]')

fig.tight_layout()
fig.savefig('Simulation.png')

plt.show()
