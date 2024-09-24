import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

# Grid and parameters
nx, ny = 256, 256
L = 20
hx, hy = L / nx, L / ny  
dt = 0.1

def calculate_curl(u,v):
    curl = np.zeros((nx + 1, ny+1))
    for i in range(1,nx):
        for j in range(1,ny):
            curl[i,j] = -(u[i,j]-u[i,j-1])/hy + (v[i,j]-v[i-1,j])/hx
    return curl

def DisplayOutputVorticity(curl):
    curl_temp=curl
    for i in range(nx+1):
        for j in range(ny+1):
            curl[i,j]=0.25*(curl_temp[i,j]+curl_temp[i+1,j]+curl_temp[i,j+1]+curl_temp[i+1,j+1])
    for i in range(nx+1):
        for j in range(ny+1):
            curl[i,j]=abs(curl[i,j])
    curl_norm = (curl - np.min(curl))/(np.max(curl) - np.min(curl))
    cv.imshow("Vorticities", curl_norm)
    cv.waitKey(1)

# Initialize velocity fields and smoke
u = np.ones((nx, ny))  # u velocity component
v = np.ones((nx, ny))  # v velocity component
u1 = np.ones((nx, ny))  # u velocity component
v1 = np.ones((nx, ny))  # v velocity component
smoke = np.zeros((nx, ny))

# Starting particle positions
x_pos, y_pos = 128,128
smoke[x_pos, y_pos] = 1

def set_init_velocity(do_taylor_green):
    if do_taylor_green:
        # Loop for u velocity component (no parallelization)
        for j in range(ny):
            for i in range(nx):
                x = hx * i + hx * 0.5
                y = hy * j + hy * 0.0
                u[i, j] = math.sin(x) * math.cos(y)
                u1[i, j] = u[i, j]
                

        # Loop for v velocity component (no parallelization)
        for j in range(ny):
            for i in range(nx):
                x = hx * i + hx * 0.5
                y = hy * j + hy * 0.0
                v[i, j] = -math.cos(x) * math.sin(y)
                v1[i, j] = v[i,j]

# Initialize the velocity field with the Taylor-Green vortex
set_init_velocity(do_taylor_green=True)


# RK4 trace method (from your code)
def traceRK4(x, y, dt):
    c1, c2, c3, c4 = (1.0 / 6.0 * dt, 1.0 / 3.0 * dt, 1.0 / 3.0 * dt, 1.0 / 6.0 * dt)
    x1, y1 = x, y
    
    u1 = bilerp_u(x1, y1)
    x2 = x1 + 0.5 * dt * u1
    v1 = bilerp_v(x1, y1)
    y2 = y1 + 0.5 * dt * v1
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
    
    x, y = x + c1 * u1 + c2 * u2 + c3 * u3 + c4 * u4, y + c1 * v1 + c2 * v2 + c3 * v3 + c4 * v4
    return mirror_pad_index(x, u.shape[0] - 1), mirror_pad_index(y, v.shape[1] - 1)

# Bilinear interpolation and velocity sampling (same as before)
def bilerp_u(x,y):
    x0, y0 = int(x//hx), int(y//hy)
    x1, y1 = x0+1, y0+1
    # x1 = np.clip(x1, 0, u.shape[0] - 1)
    # y1 = np.clip(y1, 0, u.shape[1] - 1)
    x0 = mirror_pad_index(x0, u.shape[0] - 1)
    y0 = mirror_pad_index(y0, u.shape[1] - 1)
    x1 = mirror_pad_index(x1, u.shape[0] - 1)
    y1 = mirror_pad_index(y1, u.shape[1] - 1)
    a = (x - x0 * hx) / hx  
    b = (y - y0 * hy) / hy  

    # a = x-x0*hx
    # b = y-y0*hy-hy/2
 
    return bilerp(u[x0,y0],u[x0,y1],u[x1,y0],u[x1,y1],a,b)
    # return (1-a)(1-b)*u[x0,y0] + a*b*u[x1,y1] + (1-a)*b*u[x0,y1] + a(1-b)*u[x1,y0]

def bilerp_v(x,y):

    x0, y0 = int(x//hx),int(y//hy)
    x1, y1 = x0+1, y0+1

    # x1 = np.clip(x1, 0, u.shape[0] - 1)
    # y1 = np.clip(y1, 0, u.shape[1] - 1)
    x0 = mirror_pad_index(x0, u.shape[0] - 1)
    y0 = mirror_pad_index(y0, u.shape[1] - 1)
    x1 = mirror_pad_index(x1, u.shape[0] - 1)
    y1 = mirror_pad_index(y1, u.shape[1] - 1)
    a = (x - x0 * hx) / hx  
    b = (y - y0 * hy) / hy  

    # a = abs(x-x0*hx-hx/2)
    # b = y-y0*hy
  
    return bilerp(v[x0,y0],v[x0,y1],v[x1,y0],v[x1,y1],a,b)
def lerp(v0, v1, c):
    return (1 - c) * v0 + c * v1

def bilerp(v00, v01, v10, v11, cx, cy):
    return lerp(lerp(v00, v01, cx), lerp(v10, v11, cx), cy)
def sample_velocity(x, y):
    # Bilinear interpolation of u and v
    # i, j = int(x // hx), int(y // hy)
    # fx, fy = (x / hx) - i, (y / hy) - j  # Fractional part

    u_val = bilerp_u(x,y)
    v_val = bilerp_v(x,y)
    # print(u_val, v_val)
    return u_val, v_val
def solveODE(x,y):
       
        x1,y1 = traceRK4(x,y,dt)
        ddt = dt/2
        x2,y2 = traceRK4(x,y,ddt)
        x2,y2 = traceRK4(x2,y2,ddt)
        
        iter_count = 0
        while dist(x1,y1,x2,y2) > 0.0001 * hx and iter_count < 6:
            x1,y1 = x2,y2
            ddt /= 2.0
            x2,y2 = x,y
            for _ in range(2 ** (iter_count + 1)):  # substeps doubling
                x2,y2 = traceRK4(x2,y2,ddt)
            iter_count += 1
            
        return x2,y2

def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*2+(y1-y2)*2)
# [...] Your interpolation functions: bilerp, bilerp_u, bilerp_v, mirror_pad_index...
def mirror_pad_index(i, max_i):

    if i < 0:
        return -i
    elif i > max_i:
        return int((i/max_i +1)*max_i - i)
    
    else:
        return i
    
def semi_lag_advect(x, y, u, v, dt):
    x1, y1 = solveODE(x, y)
    x1 = mirror_pad_index(x1, u.shape[0] - 1) 
    y1 = mirror_pad_index(y1, u.shape[1] - 1)
    u1_val, v1_val = sample_velocity(x1, y1)
    x_new = x + dt * u1_val  
    y_new = y + dt * v1_val
    return mirror_pad_index(x_new, u.shape[0] - 1), mirror_pad_index(y_new, u.shape[1] - 1)

# Initialize velocity with Taylor-Green vortex
set_init_velocity(do_taylor_green=True)

# Simulation loop
timesteps = 200
for t in range(timesteps):
    smoke.fill(0)
    smoke[int(x_pos), int(y_pos)] = 10
    x_pos, y_pos = semi_lag_advect(x_pos, y_pos, u, v, dt)

    # Visualization
    plt.imshow(smoke, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    plt.title(f"Time step: {t}")
    plt.pause(0.1)
    plt.clf()

plt.show()