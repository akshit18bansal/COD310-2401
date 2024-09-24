import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

#Grid and parameters
nx, ny = 256,256
L = 2*math.pi
hx, hy = L / nx, L / ny  
h = hx
dt = 0.1

#Initialize velocity fields and smoke
u = np.zeros((nx, ny))  # u velocity component
v = np.zeros((nx, ny))  # v velocity component

smoke = np.ones((nx, ny))


def set_init_velocity(do_taylor_green):
    if do_taylor_green:
        
        for j in range(ny):
            for i in range(nx):
                x = hx * i + 0.5*hx
                y = hy * j
                u[i, j] = math.sin(x) * math.cos(y)
                # print(u[i, j])
                

       
        for j in range(ny):
            for i in range(nx):
                x = hx * i
                y = hy * j + 0.5 * hy
                v[i, j] = -math.cos(x) * math.sin(y)
                # v1[i, j] = v[i,j]

set_init_velocity(do_taylor_green=True)

def sampleField(x,y, field):
   
    # Find grid cell indices (i, j)
    i = math.floor(x / h)
    j = math.floor(y / h)
    
    # Compute fractional offsets within the cell
    cx = (x / h) - i
    cy = (y / h) - j

    i = mirror_pad_index(math.floor(x / h),u.shape[0] - 1)
    j = mirror_pad_index(math.floor(y / h),v.shape[0] - 1)
    i1 = mirror_pad_index(i+1,u.shape[0] - 1)
    j1 = mirror_pad_index(j+1,v.shape[0] - 1)
    
    # Sample the field at the four corners of the grid cell
    v00 = field[i,j]         # Bottom-left
    v01 = field[i1,j]    # Bottom-right
    v10 = field[i,j1]    # Top-left
    v11 = field[i1,j1]  # Top-right
    
    # Perform bilinear interpolation
    return bilerp(v00, v01, v10, v11, cx, cy)

def clamp_pos(x, y):
    x = min(max(0.0 * h, x), float(nx * h) - 0.0 * h)
    y = min(max(0.0 * h, y), float(ny * h) - 0.0 * h)
    return x, y



#RK4 trace method
def traceRK4(x, y, dt):
    c1, c2, c3, c4 = (1.0 / 6.0 * dt, 1.0 / 3.0 * dt, 1.0 / 3.0 * dt, 1.0 / 6.0 * dt)
    x1, y1 = x, y
    
    u11 = bilerp_u(x1, y1)
    x2 = x1 + 0.5 * dt * u11
    v11 = bilerp_v(x1, y1)
    y2 = y1 + 0.5 * dt * v11
    u2 = bilerp_u(x2, y2)
    x3 = x2 + 0.5 * dt * u2
    v2 = bilerp_v(x2, y2)
    y3 = y2 + 0.5 * dt * v2
    u3 = bilerp_u(x3, y3)
    x4 = x3 + dt * u3
    v3 = bilerp_v(x3, y3)
    y4 = y3 + dt * v3
    u4 = bilerp_u(x4, y4)
    v4 = bilerp_v(x4, y4)
    
    x, y = x + c1 * u11 + c2 * u2 + c3 * u3 + c4 * u4, y + c1 * v11 + c2 * v2 + c3 * v3 + c4 * v4
    return clamp_pos(x,y)

#Bilinear interpolation and velocity sampling
def bilerp_u(x,y):
    
    return sampleField(x-0.5*h,y,u)

def bilerp_v(x,y):
   
    return sampleField(x,y-0.5*h,v)

def lerp(v0, v1, c):
    return (1 - c) * v0 + c * v1

def bilerp(v00, v01, v10, v11, cx, cy):
    return lerp(lerp(v00, v01, cx), lerp(v10, v11, cx), cy)

def sample_velocity(x, y):

    u_val = bilerp_u(x,y)
    v_val = bilerp_v(x,y)
  
    return u_val, v_val

def solveODE(x,y,dt):
       
        x1,y1 = traceRK4(x,y,dt)
        ddt = dt/2
        x2,y2 = traceRK4(x,y,ddt)
        x2,y2 = traceRK4(x2,y2,ddt)
        
        iter_count = 0
        while dist(x1,y1,x2,y2) > 0.0001 * hx and iter_count < 6:
            x1,y1 = x2,y2
            ddt /= 2.0
            x2,y2 = x,y
            for _ in range(2 ** (iter_count + 1)):  #substeps doubling
                x2,y2 = traceRK4(x2,y2,ddt)
            iter_count += 1
            
        return x2,y2

def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*2+(y1-y2)*2)

def mirror_pad_index(i, max_i):

    if i < 0:
        return -i
    elif i > max_i:
        return int((i/max_i +1)*max_i - i)
    
    else:
        return i
#Semi-Lagrangian advection
def semi_lag_advect(x, y, field):
    
    x1,y1 = solveODE(x,y,-dt)
    
    value = sampleField(x1,y1,field)
    return value


def calculate_curl(u,v):
    curl = np.zeros((nx, ny))
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            curl[i,j] = -(u[i,j]-u[i,j-1])/hy + (v[i,j]-v[i-1,j])/hx
   
    return curl

def DisplayOutputVorticity(curl):
    curl_temp=curl
    for i in range(nx-1):
        for j in range(ny-1):
            curl[i,j]=0.25*(curl_temp[i,j]+curl_temp[i+1,j]+curl_temp[i,j+1]+curl_temp[i+1,j+1])
    for i in range(nx):
        for j in range(ny):
            curl[i,j]=abs(curl[i,j])
    curl_norm = (curl - np.min(curl))/(np.max(curl) - np.min(curl) + 1e-5)
    curl_8bit = (curl_norm * 255).astype(np.uint8)
    
    # Apply a colormap to the curl values
    curl_colored = cv.applyColorMap(curl_8bit, cv.COLORMAP_HSV) 
    
    # Display the colored vorticity field
    cv.imshow("Vorticities", curl_colored)
    cv.waitKey(1)

import matplotlib.pyplot as plt


def display_density(density, t):
    plt.imshow(density, cmap='inferno', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Density')
    plt.title(f"Density Field at Time Step: {t}")
    plt.pause(0.1)  # Pause to update the plot
    plt.clf()  # Clear the figure for the next time step

def set_init_rayleigh_taylor():
    # Center position for the Rayleigh-Taylor initialization
    
    radius = 0.04

    # Loop over all grid points
    for tIdx in range(nx * ny):
        i = tIdx % nx
        j = tIdx // ny

        # Calculate the position of the cell in the grid
        x = h * (i + 0.5)
        y = h * (j + 0.5)

        # Check if the distance from the center is within the specified radius
        if dist(0.1,0.1,x,y) < radius:
            # Set density to 1 inside the radius
            smoke[i, j] = 1.0
          
set_init_rayleigh_taylor()      
#Simulation loop
timesteps = 10
for t in range(timesteps):
    smoke_new = np.ones_like(smoke)
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    curl = calculate_curl(u,v)
    DisplayOutputVorticity(curl)
    display_density(smoke, t)
    print(t)
    for i in range(nx):
        for j in range(ny):
            # Semi-Lagrangian advection to update velocities at each grid point
            u_new[i, j] = semi_lag_advect(i,j,u)
            v_new[i, j] = semi_lag_advect(i,j,v)
            
            # Semi-Lagrangian advection to update the smoke field
            smoke_new[i, j] = semi_lag_advect(i, j, smoke)

    u = u_new.copy()
    v = v_new.copy()
    smoke = smoke_new.copy()
    display_density(smoke, t)

plt.show()
