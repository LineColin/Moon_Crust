#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:58:43 2022

@author: linecolin
"""


import numpy as np
import scipy.integrate as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams

def default_settings():
    """ settings for the figure style """
    rcParams['xtick.top'] = False
    rcParams['ytick.right'] = False
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.minor.width'] = 1
    rcParams['ytick.minor.width'] = 1
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    rcParams['xtick.minor.size'] = 2.5
    rcParams['ytick.minor.size'] = 2.5
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.edgecolor'] = 'k'
    rcParams['legend.framealpha'] = .9
    rcParams['legend.shadow'] = True
    rcParams['legend.handlelength'] = 3.0
    rcParams['axes.linewidth'] = 1.5
    rcParams['axes.titlepad'] = 9.0
    rcParams['axes.titlesize'] = 15
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 18
    #rcParams['axes.labelweight'] = 'bold'
    rcParams['font.sans-serif'] = 'Arial'
    rcParams['figure.titlesize'] = 25
    rcParams['font.size'] = 15
    
    return


# =================================

def RK4(f, y, t, dt, *args):
    
    k1 = f(y, t, *args)
    k2 = f(y+k1*dt/2, t+dt/2, *args)
    k3 = f(y+k2*dt/2, t+dt/2, *args)
    k4 = f(y+k3*dt/2, t+dt/2, *args)
    
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# =================================

def Newton(f, x, T, h=1.0E-6, epsilon=1.0E-8):
    NbIterationMax = 100
    n = 0
    while (np.abs(f(x, T)) > epsilon) and (n < NbIterationMax):
        x = x - 2 * h * f(x, T) / ( f(x + h, T) - f(x - h , T) )
        n += 1
    return x if n < NbIterationMax else None

# =================================

def func_Ts(Ts, T):
    
    cste = k * (alpha * rho * g/(K * mu * Rac))**(1/3)
    
    return sigma*e*(Ts**4 - T_eq**4) - cste*(T - Ts)**(4/3)

# =================================

def dRdt(y, t, Ts, h, V):
    
    """
    """
    H = h*V*np.exp(-Lambda*t/3.15E7)
    a = rho * Lat + rho*Cp*p
    return (sigma*e*(Ts**4 - T_eq**4)*R_MOON**2 - H)/(a*y**2)

# =================================

def Rayleigh(T, Ts, d):
    
    return alpha*rho*g*(T - Ts)/(K*mu)*d**3

# =================================

def h_anal(r, D, C0):
    
    n = len(r)
    c_r = np.zeros(n)
    
    for i in range(n):
        
        c_r[i] = C0*((R_MOON**3 - R_CORE**3)/(R_MOON**3 - r[i]**3))**(1-D)
    
    return c_r

# =================================
    

def Evolution(D, C_0, C_E, h0):
    
    ### VARIABLE PARAMETERS 
    
    C = C_0/C_E
    k = 4
    rho = 3E3
    Cp = 1000
    Lat = 4E5
    K = k/(rho*Cp)
    Lambda = np.abs(np.log(21/25)/300E6)
    R_crit = pow(R_MOON**3 - (R_MOON**3 - R_CORE**3)*C, 1/3)
    
    ### INITIAL CONDITIONS ###
    
    E = R_MOON -  R_CORE
    T = m*C_0 + p
    V_LMO = 4*np.pi*(R_MOON**3 - R_CORE**3)/3
    h_LMO = h0*rho
    h_SC = 0
    
    Ts = Newton(func_Ts, Ts0, T) 

    dt = 1E6
    t = 0
    time = [0]
    R = R_CORE
    
    Ra = Rayleigh(T, Ts, E)
    
    ### data storage 
    radius = [R_CORE]
    Temp = [T]
    Temp_Surf = [Ts]
    H_LMO = [h_LMO * V_LMO]
    H_SC = [D*h_LMO]
    H_TOT = [h_LMO*V_LMO]
    
    h_r_t = [D*h_LMO]
    F_Lat = [0]
    F_Rad = [sigma*e*(Ts**4 - T_eq**4)*R_MOON**2]
    f_Conv = [ k * (T - Ts) * (Ra / Rac)**(1/3) /E]
    f_Rad = [sigma*e*(Ts**4 - T_eq**4)]
    
    I = [0]
    F_int = [0]
    F_TOT = [F_Rad[0]]
    
    Rayleigh_t = [Ra]

    i = 1
    
    while R <= R_crit : 
        
        V_LMO = 4*np.pi*(R_MOON **3 - R**3)/3
        
        R = RK4(dRdt, R, t, dt, Ts, h_LMO, V_LMO)
        
        t += dt
        time.append(t)
        radius.append(R)
        
        E = R_MOON - R
        
        dr = radius[i] - radius[i-1]
        dV_SC =4*np.pi*dr*R**2
        
        r = np.linspace(R, R_MOON, i) 
        
        phi = dV_SC/V_LMO
        
        h_LMO = h_LMO/(1 - phi + D*phi)
        h_SC += D*h_LMO*dV_SC
        
        V_LMO = 4*np.pi*(R_MOON **3 - R**3)/3
        H_LMO.append(h_LMO*V_LMO)
        
        H_SC. append(h_SC)
        H_TOT.append(h_LMO*V_LMO+ h_SC)
        h_r_t.append(D*h_LMO)
        
        T = m*C_0*(R_MOON**3 - R_CORE**3)/(R_MOON**3 - R**3) + p
        Temp.append(T)
        
        
        Ts = Newton(func_Ts, Ts, T) 
        Temp_Surf.append(Ts)
        
        Ra = Rayleigh(T, Ts, E)
        
        F_Lat.append(rho*Lat*dV_SC)
        f_Conv.append(k * (T - Ts) * (Ra / Rac)**(1/3) /E)     
        F_Rad.append(sigma*e*(Ts**4 - T_eq**4)*R_MOON**2)
        f_Rad.append(sigma*e*(Ts**4 - T_eq**4))
        Rayleigh_t.append(Ra)
        
        I.append(sc.trapz(4*np.pi*rho*Cp*T*dr*r**2))
        F_int.append(I[i] - I[i-1])
        F_TOT.append(H_TOT[i] + F_Lat[i] - F_int[i] )
        
        
        i+=1 
        
    F_Lat[0] = F_Lat[1]
    
    h_f = h_LMO
    print('h f', h_f)
    
    data = {'time': time, 'radius' : radius, 'Temp' : Temp, 'T surface':Temp_Surf,
            'H LMO' : H_LMO, 'H SC' : H_SC, 'H TOT':H_TOT, 'Rayleigh':Rayleigh_t, 
            'h(r)':h_r_t, 'F Lat':F_Lat, 'F Rad':F_Rad, 'f Conv':f_Conv, 
            'f Rad': f_Rad, 'F TOT':F_TOT, 'F int':F_int, 'h':h_f}
        
    return data

### GLOBAL PARAMETERS ###

R_MOON = 1737E3
R_CORE = 350E3
V_MOON = (R_MOON**3 - R_CORE**3)*np.pi*4/3

m = (2103 - 1585)/(0.1 - 0.72)
p = 2103 - m*0.1

sigma = 5.67E-8
e = 1

R_SUN = 700E3
A_MOON = 0.12
D_SUN = 1495E5
T_SUN = 5780

T_eq = T_SUN * np.sqrt(R_SUN/(2*D_SUN)) * (1 - A_MOON)**0.25 

Rac = 1500
g = 1.62
alpha = 1E-5
mu = 1



k = 4
rho = 3E3
Cp = 1000
Lat = 4E5
K = k/(rho*Cp)
Lambda = np.abs(np.log(21/25)/300E6)

Ts0 = 1300


# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# default_settings()

# data = Evolution()

# print(data['T surface'][1])

# radius = [i/1000 for i in data['radius']]
# time = [i/3.15E7 for i in data['time']]

# fig, ax = plt.subplots(ncols=1, nrows = 3, figsize=(10,10), sharex=True)

# ax[0].plot(time, radius, 'blue')
# ax[0].set_ylabel('radius [km]')
# ax[0].grid(color='gray', linestyle='--', linewidth=0.75)

# ax[1].plot(time, data['T LMO'], 'blue')
# ax[1].set_ylabel(R'$T_{LMO}$ [K]')
# ax[1].grid(color='gray', linestyle='--', linewidth=0.75)

# ax[2].plot(time, data['T surface'], 'blue')
# ax[2].set_ylabel(R'$T_{surface}$ [K]')
# ax[2].set_xlabel('time [years]')
# ax[2].grid(color='gray', linestyle='--', linewidth=0.75)

# plt.subplots_adjust(wspace=0.35, hspace=0)

# #plt.savefig()
# plt.show()

# fig, ax = plt.subplots(ncols=1, nrows = 2, figsize=(10,7))
# fig.tight_layout()

# ax[0].plot(data['T LMO'], radius, 'blue')
# ax[0].set_ylabel('radius [km]')
# ax[0].set_xlabel('T [K]')
# ax[0].grid(color='gray', linestyle='--', linewidth=0.75)

# ax[1].plot(data['h(r)'], radius, 'blue', label='model')
# ax[1].plot(h_anal(data['radius'], D, D*h0*rho), radius, 'red', label= R'h = f(r)')
# ax[1].set_ylabel('radius [km]')
# ax[1].set_xlabel('h(r) $[W\cdot m^{-3}]$')
# ax[1].grid(color='gray', linestyle='--', linewidth=0.75)
# ax[1].legend()

# #plt.savefig()
# plt.show()

# # fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize=(10, 7))
# # fig.tight_layout()

# # ax[0,0].plot(time, data['f Rad'], 'blue', label = R'$q_{rad}$')
# # ax[0,0].plot(time, data['f Conv'], 'red', label = R'$q_{Conv}$')
# # ax[0,0].set_xlabel(R'time [year]')
# # ax[0,0].set_ylabel(R'$[W\cdot m^{-2}]$')
# # ax[0,0].grid(color='gray', linestyle='--', linewidth=0.75)
# # ax[0,0].legend()

# # ax[0,1].plot(time, data['H TOT'], 'blue', label='Total')
# # ax[0,1].plot(time, data['H LMO'], 'red', label='LMO')
# # ax[0,1].plot(time, data['H SC'], 'orange', label = 'Cumulates')
# # ax[0,1].set_xlabel('time [year]')
# # ax[0,1].set_ylabel('[W]')
# # ax[0,1].grid(color='gray', linestyle='--', linewidth=0.75)
# # ax[0,1].legend()

# # ax[1,0].set_yscale('log')
# # ax[1,0].plot(time, data['F Rad'], 'blue', label = R'$\Phi_{rad}$')
# # ax[1,0].plot(time, data['H TOT'], 'red', label='$\Phi_{H}$')
# # ax[1,0].plot(time, data['F Lat'], 'orange', label='$\Phi_{Lat}$')
# # ax[1,0].plot(time[1:], data['F TOT'][1:], 'green', label='$\Phi_{Lat}$')
# # ax[1,0].plot(time, data['F int'], 'darkviolet', label='$\Phi_{int}$')
# # ax[1,0].set_ylabel('time [year]')
# # ax[1,0].set_xlabel('[W]')
# # ax[1,0].grid(color='gray', linestyle='--', linewidth=0.75)

# # ax[1,1].plot(time, data['F Rad'])
# # ax[1,1].set_ylabel('time [year]')
# # ax[1,1].set_xlabel('[W]')
# # ax[1,1].grid(color='gray', linestyle='--', linewidth=0.75)











