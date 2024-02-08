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
m_dry = 0.78  # mass of rocket without propellant but including motor casing, electronics, etc. in kg
area = np.pi * (0.029 / 2) ** 2  
# chute_area = np.pi * 0.3 ** 2  # parachute area in m^2
chute_area = 0.481  # parachute area in m^2
Cd = 0.75  # drag coefficient

# motor chute delay tested at 13, 10, 8, 6, and 4 seconds
chute_delay = 13 # time to deploy chute in seconds

# Simulation Parameters
dt = 0.01  # in seconds
simulation_time = 125  # total simulation time
t = np.arange(0, simulation_time, dt)  

# Initial Conditions - create arrays and give initial values
acceleration = [0]
velocity = [0]
position = [0]
mass = [m_dry + m_propellant]
drag = [0]

chute_deployed = False

# Simulation
for i in range(len(t) - 1):
    F_thrust = t_interp(t[i])  # get thrust at time t[i]
    F_drag = 0.5 * 1.225 * Cd * area * velocity[i] ** 2  # drag force
    F_chute = 0.5 * 1.225 * Cd * chute_area * velocity[i] ** 2  # additional drag from chute
    F_gravity = 9.80665 * mass[i]  # force due to gravity
    if mass[i] > m_dry:  # if there is still propellant left
        F_net = F_thrust - (F_gravity + F_drag)  # net force
    else:  # if all propellant has been used
        if velocity[-1] < 0:  # if past apogee (descending)
            if t[i]>chute_delay:
                F_net = F_chute + F_drag - F_gravity  # include chute drag if descending
            else:
                F_net = F_drag - F_gravity
        else:
            if t[i]>chute_delay:
                if not chute_deployed: # check to only print once
                    print('Chute Deployed before Apogee!')
                    chute_deployed = True
                F_net = -F_chute-F_drag - F_gravity  # ascending with chute deployed
            else:
                F_net = -F_drag - F_gravity  # ascending and decelerating

    a = F_net / mass[i]  # calculate acceleration
    v = velocity[i] + (a * dt)  # calculate new velocity
    x = np.max([position[i] + (v * dt), 0])  # calculate new position, ensuring it doesn't go below 0
    m = np.max([mass[i] - (fuel_mass_rate * dt), m_dry])  # calculate new mass, ensuring it doesn't go below m_dry

    # append new values to arrays
    drag.append(F_drag)
    acceleration.append(a)
    velocity.append(v if x > 0 else 0)
    position.append(x)
    mass.append(m)

# Find apogee, descent rate, and total flight time
try:
    apogee_index = np.where(np.array(velocity) < 0)[0][1]
    print('Time to Apogee:', t[apogee_index], 's')
except IndexError:
    print("Simulation did not reach apogee.")
    apogee_index = np.nan

try:
    ground_index = t[np.where(np.gradient(position) == 0)[0][1]]
    print('Time to ground: ', t[ground_index], 's')
except IndexError:
    print("Simulation did not reach ground.")

print('Descent Rate: ', round(np.min(np.array(velocity)[t>(chute_delay+5)]),2), 'm/s')


print('Max Position:', round(np.max(position)*3.28084,2), 'ft')

# Desired descent rate
descent_rate = 5  # m/s

chute_area_needed = np.max(np.array(drag)) / (0.5 * 1.225 * Cd * (descent_rate ** 2))
print("Chute area needed for {} m/s descent rate:".format(descent_rate), round(chute_area_needed,2), "m^2")

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

ax[2].axhspan((2300 / ftm) * (alt_feet * ftm), (2500 / ftm) * (alt_feet * ftm), facecolor='g' if np.max(position)>2300/ftm else 'r', alpha=0.3, edgecolor='none')

ax[2].set_ylabel('Position [ft]')
ax[2].text(t[-1], np.max(position) * (alt_feet * ftm),
            f'Max Position: {(np.max(position) * (alt_feet * ftm)):.2f} ft', fontsize=10, color='r',
            verticalalignment='top', ha='right')

ax[3].plot(t, mass)
ax[3].set_ylabel('Mass [kg]')

ax[4].plot(t, drag)
# ax[4].axhline(0.5 * 1.225 * Cd * chute_area_needed * np.max(np.array(velocity)**2), color='b', linestyle='--', label='Chute Needed')
ax[4].set_ylabel('Drag [N]')
ax[4].set_xlabel('Time [s]')

for a in range(5):
    ax[a].axvline(chute_delay, c='purple', linestyle='--', label='Chute Deployed', alpha=0.3, zorder=-100)

ax[4].legend(frameon=False)

fig.tight_layout()
fig.savefig('Simulation.png')

# plt.show()
