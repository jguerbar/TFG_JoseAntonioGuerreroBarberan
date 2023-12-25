import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import os
import re

QUIVER_STEP = 20
##Load numpy data in shape (T, H, W, C)
data = np.load("./data/EnclosedFlow_RandomInit_3.npy")
data = data[1:,:,:,:]
x = np.linspace(0, 1, data.shape[2])
y = np.linspace(0, 1, data.shape[1])
X, Y = np.meshgrid(x, y)

# Select data
u = data[:, :, :, 0]
v = data[:, :, :, 1]
p = data[:, :, :, 2]
mod = np.sqrt(np.square(data[:, :, :, 0])+np.square(data[:, :, :, 1]))
dudy = (u-np.roll(u,1,2))/(1/255)
dvdx = (v-np.roll(v,1,1))/(1/255)
omega = dvdx-dudy

mod = omega #TODO remove

# Set up the figure and animation
fig_1, ax_1 = plt.subplots(1,2,figsize=(14,7), dpi=100, squeeze=True)
fig_2, ax_2 = plt.subplots(2,2,figsize=(14,7), dpi=100, squeeze=True)

if False:
    t = 0
    # Plotting the pressure field as a contour
    contour_p = ax_1[0].contourf(X, Y, p[t], alpha=0.5, cmap=cm.viridis)
    # Plotting velocity field every 10 elements
    quiver = ax_1[0].quiver(X[::QUIVER_STEP, ::QUIVER_STEP], Y[::QUIVER_STEP, ::QUIVER_STEP], u[t,::QUIVER_STEP, ::QUIVER_STEP], v[t,::QUIVER_STEP, ::QUIVER_STEP])

    # Plotting the velocity modulo as a contour
    contour_uv = ax_1[1].contourf(X, Y, mod[t], alpha=0.5, cmap=cm.viridis)
    # Plot of velocity field
    stream = ax_1[1].streamplot(X, Y, u[t], v[t])
    plt.show()
    print(np.mean(mod[t]))
    exit()
# Function to update the plot
def update_1(t):
    ax_1[0].clear()
    ax_1[1].clear()

    # Plotting the pressure field as a contour
    contour_p = ax_1[0].contourf(X, Y, p[t], alpha=0.5, cmap=cm.viridis)
    # Plotting velocity field every 10 elements
    quiver = ax_1[0].quiver(X[::QUIVER_STEP, ::QUIVER_STEP], Y[::QUIVER_STEP, ::QUIVER_STEP], u[t,::QUIVER_STEP, ::QUIVER_STEP], v[t,::QUIVER_STEP, ::QUIVER_STEP])

    # Plotting the velocity modulo as a contour
    contour_uv = ax_1[1].contourf(X, Y, mod[t], alpha=0.5, cmap=cm.viridis)
    # Plot of velocity field
    stream = ax_1[1].streamplot(X, Y, u[t], v[t])


    return contour_p, stream, quiver, contour_uv
def update_2(t):
    ax_2[0][0].clear()
    ax_2[0][1].clear()
    ax_2[1][0].clear()
    ax_2[1][1].clear()

    img_u = ax_2[0][0].imshow(u[t], cmap='gray', vmin=u[t].min(), vmax=u[t].max(), origin='lower')
    img_v = ax_2[0][1].imshow(v[t], cmap='gray', vmin=v[t].min(), vmax=v[t].max(), origin='lower')

    img_p = ax_2[1][0].imshow(p[t], cmap='gray', vmin=p[t].min(), vmax=p[t].max(), origin='lower')
    img_uv = ax_2[1][1].imshow(mod[t], cmap='gray', vmin=mod[t].min(), vmax=mod[t].max(), origin='lower')

    return img_u, img_v, img_p, img_uv

R = data.shape[0]
ani_1 = FuncAnimation(fig_1, update_1, frames=range(R), blit=False)
ani_2 = FuncAnimation(fig_2, update_2, frames=range(R), blit=False)

ani_1.save('animation_1.mp4', fps=3)  # Adjust fps as needed
ani_2.save('animation_2.mp4', fps=3)  # Adjust fps as needed

exit()













t = 100
fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p[t], alpha=0.5, cmap=cm.viridis)  
plt.colorbar()
#Plot of velocity field
plt.streamplot(X, Y, u[t], v[t])
# plotting velocity field every two elements
plt.quiver(X[::2, ::2], Y[::2, ::2], u[t,::2, ::2], v[t,::2, ::2]) 
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
exit()

# Initialize four image plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
imgs = []
for i in range(3):
    imgs.append(axs[i // 2, i % 2].imshow(data[0, :, :, i], cmap='gray', vmin=data[0, :, :, i].min(), vmax=data[0, :, :, i].max(), origin='lower'))

# For the square root of the square sum of the first two channels
modulo = np.sqrt(data[0, :, :, 0]**2 + data[0, :, :, 1]**2)
imgs.append(axs[1, 1].imshow(modulo, cmap='gray'))

def update(frame):
    for i in range(3):
        frame_data = data[frame, :, :, i]
        imgs[i].set_data(frame_data)
        imgs[i].set_clim(frame_data.min(), frame_data.max())

    modulo = np.sqrt(data[frame, :, :, 0]**2 + data[frame, :, :, 1]**2)
    imgs[3].set_data(modulo)
    imgs[3].set_clim(modulo.min(), modulo.max())
    return imgs

# Create the animation
ani = FuncAnimation(fig, update, frames=data.shape[0], blit=True)

# Display the animation:
#plt.show()

# To save the animation as a video file, uncomment the following line:
ani.save('animation.mp4', fps=3)  # Adjust fps as needed