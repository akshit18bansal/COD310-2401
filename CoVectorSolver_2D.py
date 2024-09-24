import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

#Grid and parameters
nx, ny = 256, 256
L = 2*math.pi
hx, hy = L / nx, L / ny  
dt = 0.1

#Initialize velocity fields and smoke
u = np.ones((nx, ny))  # u velocity component
v = np.ones((nx, ny))  # v velocity component
u1 = np.ones((nx, ny))  # u velocity component
v1 = np.ones((nx, ny))  # v velocity component
smoke = np.zeros((nx, ny))


#Starting particle positions
x_pos, y_pos = 128,128
smoke[x_pos, y_pos] = 1



def set_init_velocity(do_taylor_green):
    if do_taylor_green:
        
        for j in range(ny):
            for i in range(nx):
                x = hx * i
                y = hy * j
                u[i, j] = math.sin(x) * math.cos(y)
                u1[i, j] = u[i, j]
                

       
        for j in range(ny):
            for i in range(nx):
                x = hx * i
                y = hy * j 
                v[i, j] = -math.cos(x) * math.sin(y)
                v1[i, j] = v[i,j]

set_init_velocity(do_taylor_green=True)


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
    return mirror_pad_index(x, u.shape[0] - 1), mirror_pad_index(y, v.shape[1] - 1)

#Bilinear interpolation and velocity sampling
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
def semi_lag_advect(x, y, u, v, dt):
    
    x1,y1 = solveODE(x,y)
   
    x1 = mirror_pad_index(x1, u.shape[0] - 1) 
    y1 = mirror_pad_index(y1, u.shape[1] - 1)
    u1_val, v1_val = sample_velocity(x1, y1)
    # smoke[int(x_pos), int(y_pos)] = smoke_dup[int(x_new), int(y_new)]
    u1[int(x),int(y)] = u1_val
    v1[int(x),int(y)] = v1_val
    x_new = x + dt * u1_val  
    y_new = y + dt * v1_val
    x_new = mirror_pad_index(x_new, u.shape[0] - 1)  
    y_new = mirror_pad_index(y_new, u.shape[1] - 1)
    u[int(x),int(y)] = u1[int(x),int(y)] 
    v[int(x),int(y)] = v1[int(x),int(y)]
    # print(x_new,y_new)
    return x_new, y_new

def set_init_rayleigh_taylor(ni, nj, h, layer_height, rho, temperature, rho_init, rho_orig, T_init, T_orig):
    # Center position for the Rayleigh-Taylor initialization
    center = np.array([0.1, layer_height])
    radius = 0.04

    # Loop over all grid points
    for tIdx in range(ni * nj):
        i = tIdx % ni
        j = tIdx // ni

        # Calculate the position of the cell in the grid
        pos = h * (np.array([i, j]) + 0.5)

        # Check if the distance from the center is within the specified radius
        if dist(center, pos) < radius:
            # Set density to 1 inside the radius
            rho[i, j] = 1.0
            rho_init[i, j] = 1.0
            rho_orig[i, j] = 1.0
        else:
            # Set temperature to 1 outside the radius
            temperature[i, j] = 1.0
            T_init[i, j] = 1.0
            T_orig[i, j] = 1.0

def semi_lag_advect_density(x, y, u, v, dt, field):
   
    # Trace back the particle's position
    x_prev, y_prev = solveODE(x, y)
    
    # Ensure the traced-back positions stay within the grid bounds
    x_prev = mirror_pad_index(x_prev, field.shape[0] - 1)
    y_prev = mirror_pad_index(y_prev, field.shape[1] - 1)

    # Bilinearly interpolate the field value (density/smoke) at the traced-back position
    field_val = bilerp(field[int(x_prev), int(y_prev)],
                       field[int(x_prev), int(y_prev+1)],
                       field[int(x_prev+1), int(y_prev)],
                       field[int(x_prev+1), int(y_prev+1)],
                       (x_prev % 1), (y_prev % 1))

    return field_val


p = np.ones((nx, ny)) 

def gauss_seidel_project(dx):
    for i in range(1,nx,2):
        for j in range(1,ny,2):
            divergence =  (u[i+1,j]-u[i,j])/dx + (v[i,j+1]-v[i,j])/dx
            del2_p = (p[i+1,j]+p[i,j+1]+p[i,j+1]+p[i,j-1]-4*p[i,j])/(dx*dx)
            p[i,j] = 0.25*(p[i+1,j]+p[i,j+1]+p[i,j+1]+p[i,j-1]+divergence)
    for i in range(2,nx,2):
        for j in range(2,ny,2):
            divergence =  (u[i+1,j]-u[i,j])/dx + (v[i,j+1]-v[i,j])/dx
            del2_p = (p[i+1,j]+p[i,j+1]+p[i,j+1]+p[i,j-1]-4*p[i,j])/(dx*dx)
            p[i,j] = 0.25*(p[i+1,j]+p[i,j+1]+p[i,j+1]+p[i,j-1]+divergence)

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
    curl_norm = (curl - np.min(curl))/(np.max(curl) - np.min(curl))
    cv.imshow("Vorticities", curl_norm)
    cv.waitKey(1)


#Simulation loop
timesteps = 100
for t in range(timesteps):
    smoke_new = np.zeros_like(smoke)
    smoke.fill(0)
    x_pos, y_pos = semi_lag_advect(x_pos, y_pos, u, v, dt)
    smoke[int(x_pos),int(y_pos)]=10
    # smoke_new[int(x_pos), int(y_pos)] = semi_lag_advect_density(x_pos,y_pos,u,v,dt,smoke)
    curl = calculate_curl(u,v)
    DisplayOutputVorticity(curl)
    #Visualization

    plt.imshow(smoke, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    plt.title(f"Time step: {t}")
    plt.pause(0.1)
    plt.clf()

plt.show()

#density field
#jax pytorch
