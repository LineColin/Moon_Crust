#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:07:14 2022

@author: linecolin
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import sparse
import scipy.sparse.linalg as LA
import pandas as pd 


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
    rcParams['legend.framealpha'] = 1
    rcParams['legend.shadow'] = True
    rcParams['legend.handlelength'] = 3.0
    rcParams['axes.linewidth'] = 1.5
    rcParams['axes.titlepad'] = 9.0
    rcParams['axes.titlesize'] = 15
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 18
    rcParams['axes.labelweight'] = 'bold'
    rcParams['font.sans-serif'] = 'Arial'
    rcParams['figure.titlesize'] = 25
    
    return

# =================================
def Analytique(param, r):
    
    T_S = param['T_S']
    T_E = param['T_E']
    
    R_top = param['R_top']
    R_bot = param['R_bot']
    
    h = param['h']
    k = param['k']
    
    C1 = (-(T_S - T_E) - h*(R_top**2 - R_bot**2)/(6*k))/(1/R_top - 1/R_bot)
    C2 = T_S + C1/R_top + h*R_top**2/(6*k)

    T0 = C2 - C1/r - h*r**2/(6*k)
    
    return T0


# =================================

def time_analytical(param, R_bot, dT):
    
    R_bot = R_bot*1000
    
    C_E = param['C_E']
    k = param['k']
    rho = param['rho']
    Lat = param['Lat']
    
    c = - rho*Lat/(k*C_E*dT*R_top)
    
    time = c*(R_bot**2 *R_top/2 - R_bot**3 /3 - R_top**3 /6)
    
    return time/3.15E13

# =================================


def Diffusion(y, nx, dt, dy, K, T_top, T_bot):

    R = np.zeros(nx)
    
    L = R_top - R_bot
    
    # cells boundary

    y_boundary = np.linspace(y[0], y[-1], n+1) 
    
    r_boundary = (y_boundary - 1)*(R_top - R_bot) + R_bot

    # cells center
        
    y_center = (y_boundary[:-1] + y_boundary[1:])/2

    r_center = (y_center - 1)*(R_top - R_bot) + R_bot
    
    s = dt*K / (dy*dy*r_center*r_center*L*L)
    
    R[0] = 2*T_top*r_boundary[0]*r_boundary[0]*s[0]
    R[-1] = 2*T_bot*r_boundary[-1]*r_boundary[-1]*s[-1]

    a = s*r_boundary[1:]*r_boundary[1:]
    b = - s*r_boundary[1:]*r_boundary[1:] - s*r_boundary[:-1]*r_boundary[:-1]
    c = s*r_boundary[:-1]*r_boundary[:-1]


    M = (np.diag(c[1:], -1) + np.diag(b, 0) + np.diag(a[:-1], +1))

    M[0,0] = - 2*s[0]*r_boundary[0]*r_boundary[0] - s[0]*r_boundary[1]*r_boundary[1]
    M[0,1] = s[0]*r_boundary[1]*r_boundary[1]
  
    M[-1, -2] = s[-1]*r_boundary[-2]*r_boundary[-2]
    M[-1, -1] = - s[-1]*r_boundary[-2]*r_boundary[-2] - 2*s[-1]*r_boundary[-1]*r_boundary[-1]

    #M = sparse.csc_matrix(M)
    
    return M, R

# =================================

def Advection(u, dy, dt):
    
    u_abs = np.abs(u)
    f = dt/(dy*4)
    
    a_a = -f*(u[1:] + u_abs[1:])
    b_a = f*3*(u_abs - u)
    c_a = f*(5*u[:-1] - 3*u_abs[:-1])
    d_a = f*(u_abs[:-2] - u[:-2])
    
    A = (np.diag(a_a, -1) + np.diag(b_a, 0) + np.diag(c_a, +1) + np.diag(d_a, +2))
        
    A[0,0] = f*3*(u_abs[0] - u[0]) - f*3*(u_abs[0] - u[0])
    A[0,1] = f*(5*u[0] - 3*u_abs[0])
    A[0,2] = f*(u_abs[0] - u[0])
        
    A[-1,-2] = -f*(u[-1] + u_abs[-1])
    A[-1,-1] = f*3*(u_abs[-1] - u[-1]) - f*(5*u[-1] - 3*u_abs[-1])
    
    #M = sparse.csc_matrix(M)
    
    return A

# =================================

def Evolution(n, param):
    
    
    print('start')
    
    # parameters
    k = param['k']
    rho = param['rho']
    Cp = param['Cp']
    Lat = param['Lat']
    K = k/(rho*Cp)
    
    h0 = param['h0']
    D_AN = param['D_AN']
    Lambda = param['lambda']
    
    C_E = param['C_E']
    C_0 = param['C_0']
    C = C_0/C_E
    
    # initial parameters
    
    L = param['L0']
    dR_bot = param['dR_bot0']
    
    # initial conditions
    
    R_bot = R_top - L
    
    R_SC_crit = pow(R_MOON**3 - (R_MOON**3 - R_CORE**3)*C, 1/3)
    R_SC = pow((1-C_E)*(R_top**3 - R_bot**3)/C_E + R_SC_crit**3, 1/3)
    
    V_LMO_0 = 4*np.pi*(R_top**3 - R_SC_crit**3)/3
    V_LMO = 4*np.pi*(R_bot**3 - R_SC**3)/3
    V_CRUST = 4*np.pi*(R_top**3 - R_bot**3)/3
    V_SC = (1 - C_E)*V_CRUST/C_E
    
    phi_CRUST = V_CRUST/V_LMO_0
    phi_SC = V_SC/V_LMO_0
    
    h_LMO_0 = h0*V_MOON*rho/V_LMO_0
    h_LMO = h_LMO_0/(D_AN*phi_CRUST + 1 - phi_CRUST - phi_SC)
    h_CRUST = D_AN*h_LMO
    
    
    dT = T_E - T_surf 
    T0 = np.linspace(T_surf, T_E, n)
    T0 = (T0 - T_surf)/dT
    
    T_top = T0[0]
    T_bot = T0[-1]
    
    r = np.linspace(R_top, R_bot, n)
    y = (r - R_bot)/(R_top - R_bot) + 1
    l = y[-1] - y[0]
    
    L = R_top - R_bot

    cfl = 0.4
    dr = L/n
    dy = l/n
    dt = cfl*dr**2 /K
    
    #h_r_0 = np.linspace(0, h_CRUST, n)
    
    # calculation matrix
    h_r = np.ones(n)*h_CRUST
    R = np.zeros(n)
    T = T0.copy()
    I = np.identity(n)
    
    # data storage
    t=0
    
    time = [t]
    
    radius_CRUST = [R_bot/1000]
    radius_SC = [R_SC_crit/1000]
    
    VOLUME_TOT = [V_LMO_0]
    VOLUME_LMO = [V_LMO]
    VOLUME_CRUST = [V_CRUST]
    VOLUME_SC = [V_SC]
    
    dR_dt = [dR_bot]
    
    H_LMO = [h_LMO*V_LMO]
    h_C = h_CRUST*V_CRUST
    H_CRUST = [h_C]
    H_TOT = [h_LMO*V_LMO + h_CRUST*V_CRUST]
    h_LMO_t = [h_LMO]
    
    F_Lat = []
    F_cond = []
    F_h_LMO = []
    F_h_CRUST = []
    F_int = [0]
    F_TOT = [0]
    
    
    i = 1
    
    while R_bot-R_SC > 200 and t/3.15E13 < 500:
        
    
        #TIME
    
        #TIME
    
        a = cfl*dr**2 /K
        b = cfl*dr/np.abs(dR_bot)
        
        dt = np.min([a, b])
        
        t += dt
        time.append(t/3.15E13)
    
        # CRUST AND CUMMULATES EVOLUTION
        
        dR_bot = k*(T[-2] - T[-1])*dT*C_E/(dr*rho*Lat) + h_LMO*np.exp(-Lambda*t/3.15E7)*C_E*V_LMO/(rho*Lat*4*np.pi*R_bot**2)
        dR_SC = -(1-C_E)*dR_bot*R_bot**2 /(C_E*R_SC**2)
        
        dR_dt.append(dR_bot)
        
        dV_CRUST = - 4*np.pi*dR_bot*dt*R_bot**2
        dV_SC = (1 - C_E)*dV_CRUST/C_E
        
        
        
        # LMO EVOLUTION
        V_LMO_m = 4*np.pi*(R_bot**3 -  R_SC**3)/3
        
        R_SC += dt*dR_SC
        R_bot += dR_bot*dt
        
        radius_CRUST.append(R_bot/1000)
        radius_SC.append(R_SC/1000)
        
        V_LMO = 4*np.pi*(R_bot**3 - R_SC**3)/3
        VOLUME_LMO.append(V_LMO)
        VOLUME_CRUST.append(4*np.pi*(R_top**3 - R_bot**3)/3)
        VOLUME_SC.append(4*np.pi*(R_SC**3 - R_SC_crit**3)/3)
        VOLUME_TOT.append(VOLUME_CRUST[i] + VOLUME_SC[i] + V_LMO)
        
        phi_CRUST = dV_CRUST/V_LMO_m
        phi_SC = dV_SC/V_LMO_m
        
        h_LMO = h_LMO/(D_AN*phi_CRUST + 1 - phi_CRUST - phi_SC)
        h_LMO_t.append(h_LMO)
        h_CRUST = D_AN*h_LMO
        h_r[-1] = h_CRUST
        
        h_C += dV_CRUST*h_CRUST*np.exp(-Lambda*t/3.15E7)
        H_CRUST.append(h_C)
        H_LMO.append(h_LMO*V_LMO*np.exp(-Lambda*t/3.15E7))
        H_TOT.append(H_CRUST[i] + H_LMO[i])
        
        
        # TEMPERATURE (diffusion + advection)
        
        L = R_top - R_bot
    
        dr = L/n
    
        u_T = dR_bot*(y - 2)/((R_top - R_bot))
        
        u_h = dR_bot*(y-2)/(R_top - R_bot)
        
        
        L, R = Diffusion(y, n, dt, dy, K, T_top, T_bot)
        
        
        A_h = Advection(u_h, dy, dt)
        A = Advection(u_T, dy, dt)
        
        N = I + A_h
        N = sparse.csc_matrix(N)
        h_r = LA.spsolve(N, h_r)
        
        
        V = T + R + (dt/(rho*Cp*dT))*h_r*np.exp(-Lambda*t/3.15E7)
        M = I - L + A
        M = sparse.csc_matrix(M)
        T = LA.spsolve(M, V)
        
        F_Lat.append(-rho*Lat*dR_bot*R_bot**2/C_E)
        F_cond.append(-k*dT*(T[-2]-T[-1])/dr)
        F_h_LMO.append(H_LMO[i]/(4*np.pi*R_bot**2))
        F_h_CRUST.append(h_C/(4*np.pi*R_bot**2))
        
        #F_TOT.append(F_cond[i] + F_Lat[i] + F_h_LMO[i] + F_h_CRUST[i])# + F_int[i])
        
        i+=1
        
        
        
        ### CUMULATS ###
        
    data = {'time':time, 'Temperature':T, 'h(r, t=tf)': h_r, 
            'r CRUST':radius_CRUST, 'r CUMULATS': radius_SC, 
            'VOLUME TOT':VOLUME_TOT, 'VOLUME CRUST': VOLUME_CRUST, 'VOLUME SC':VOLUME_SC,
            'VOLUME LMO':VOLUME_LMO, 'H TOT': H_TOT, 'H CRUST':H_CRUST, 
            'H LMO':H_LMO, 'dR/dt':dR_dt, 'h LMO':h_LMO_t, 'F Lat':F_Lat,
            'F Cond':F_cond, 'F h LMO': F_h_LMO, 'F h CRUST':F_h_CRUST,
            'F int':F_int, 'F TOT': F_TOT}    
        
    print('end')
    
    return data








# default_settings()

### GLOBAL PARAMETERS ###

R_MOON = 1737E3
R_CORE = 350E3

V_MOON = 4*np.pi*(R_MOON**3 - R_CORE**3)/3

R_top = R_MOON
R_bot = R_top - 1000

T_surf = 250
T_E = 1585

dT = T_E - T_surf

D = 1.496E8
R_SUN = 696340
T_SUN = 5778
A_MOON = 0.12

print(T_SUN*(1-A_MOON)**0.25 *np.sqrt(R_SUN/(2*D)))



# ### VARIABLE PARAMETERS ###

n = 50

# R = np.linspace(R_top, R_bot, n+1)
# Y = (R - R_bot)/(R_top - R_bot) + 1
# Y = (Y[:-1] + Y[1:])/2

# param_test1 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
#           'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':0, 'D_AN':1E-2}
# param_test2 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
#           'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':0}
# param_test3 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
#           'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':1E-2}
# param_test4 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
#           'L0':1000, 'dR_bot0':-1E-8, 'h0':25E-12, 'lambda':np.abs(np.log(21/25)/300E6), 'D_AN':1E-3}
# param_test5 = {'k':2, 'rho':3E3, 'Cp':1000, 'Lat': 4E5, 'C_0': 0.1, 'C_E':0.7, 
#           'L0':1000, 'dR_bot0':-1E-8, 'h0':0, 'lambda':0, 'D_AN':0}

# data_test1 = Evolution(n, param_test1)
# data_test2 = Evolution(n, param_test2)
# data_test3 = Evolution(n, param_test3)
# data_test4 = Evolution(n, param_test4)
# data_test5 = Evolution(n, param_test5)

# param1 = {'T_S':250, 'T_E':1600, 'R_top':1737E3, 'R_bot':1687E3, 'k':2, 'h':data_test1['h(r, t=tf)']}
# param2 = {'T_S':250, 'T_E':1600, 'R_top':1737E3, 'R_bot':1687E3, 'k':2, 'h':data_test2['h(r, t=tf)']}
# param3 = {'T_S':250, 'T_E':1600, 'R_top':1737E3, 'R_bot':1687E3, 'k':2, 'h':data_test3['h(r, t=tf)']}
# param4 = {'T_S':250, 'T_E':1600, 'R_top':1737E3, 'R_bot':1687E3, 'k':2, 'h':data_test4['h(r, t=tf)']}
# param5 = {'T_S':250, 'T_E':1600, 'R_top':1737E3, 'R_bot':1687E3, 'k':2, 'h':data_test5['h(r, t=tf)']}


# r = np.linspace(1737E3, 1687E3, n)

# T_th1 = Analytique(param1, r)
# T_th2 = Analytique(param2, r)
# T_th3 = Analytique(param3, r)
# T_th4 = Analytique(param4, r)
# T_th5 = Analytique(param5, r)



# plt.figure()
# plt.plot(data_test1['time'], data_test1['r CRUST'], 'blue')
# plt.plot(data_test1['time'], data_test1['r CUMULATS'], 'slateblue')

# plt.plot(data_test2['time'], data_test2['r CRUST'], 'red')
# plt.plot(data_test2['time'], data_test2['r CUMULATS'], 'brown')

# plt.plot(data_test3['time'], data_test3['r CRUST'], 'orange')
# plt.plot(data_test3['time'], data_test3['r CUMULATS'], 'goldenrod')

# plt.plot(data_test4['time'], data_test4['r CRUST'], 'darkorchid')
# plt.plot(data_test4['time'], data_test4['r CUMULATS'], 'violet')

# plt.plot(data_test5['time'], data_test5['r CRUST'], 'green')
# plt.plot(data_test5['time'], data_test5['r CUMULATS'], 'darkgreen')

# plt.ylabel('radius [km]')
# plt.xlabel('time [My]')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('radius.png')
# plt.show()

# plt.figure()
# plt.plot(data_test1['time'], data_test1['H TOT'], 'blue')
# plt.plot(data_test2['time'], data_test2['H TOT'], 'red')
# plt.plot(data_test3['time'], data_test3['H TOT'], 'orange')
# plt.plot(data_test4['time'], data_test4['H TOT'], 'darkorchid')
# plt.plot(data_test5['time'], data_test5['H TOT'], 'green')
# plt.ylabel('H TOT [W]')
# plt.xlabel('time [My]')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('H_TOT.png')
# plt.show()

# plt.figure()
# plt.plot(data_test1['time'], data_test1['h LMO'], 'blue')
# plt.plot(data_test2['time'], data_test2['h LMO'], 'red')
# plt.plot(data_test3['time'], data_test3['h LMO'], 'orange')
# plt.plot(data_test4['time'], data_test4['h LMO'], 'darkorchid')
# plt.plot(data_test5['time'], data_test5['h LMO'], 'green')
# plt.ylabel('h LMO $[W m^{-3}]$')
# plt.xlabel('time [My]')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('h_LMO.png')
# plt.show()

# plt.figure()
# plt.plot(data_test1['time'], data_test1['VOLUME TOT'], 'blue')
# plt.plot(data_test2['time'], data_test2['VOLUME TOT'], 'red')
# plt.plot(data_test3['time'], data_test3['VOLUME TOT'], 'orange')
# plt.plot(data_test4['time'], data_test4['VOLUME TOT'], 'darkorchid')
# plt.plot(data_test5['time'], data_test5['VOLUME TOT'], 'green')
# plt.ylabel('VOLUME TOT $[m^{-3}]$')
# plt.xlabel('time [My]')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('V_TOT.png')
# plt.show()

# plt.figure(figsize = (14, 7))
# plt.plot(data_test1['Temperature'], Y, 'blue')
# plt.plot(data_test2['Temperature'], Y, 'red')
# plt.plot(data_test3['Temperature'], Y, 'orange')
# plt.plot(data_test4['Temperature'], Y, 'darkorchid')
# plt.plot(data_test5['Temperature'], Y, 'green')
# plt.ylabel('radius')
# plt.xlabel('Temperature, $t=t_{I}$')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('T.png')
# plt.show()

# plt.figure()
# plt.plot(data_test1['h(r, t=tf)'], Y, 'blue')
# #plt.plot(data_test2['h(r, t=tf)'], Y, 'red')
# plt.plot(data_test3['h(r, t=tf)'], Y, 'orange')
# plt.plot(data_test4['h(r, t=tf)'], Y, 'darkorchid')
# plt.ylabel('radius')
# plt.xlabel('h(r) CRUST, $t=t_{I}$')
# plt.grid(color='gray', linestyle='--', linewidth=0.75)
# plt.savefig('h_r.png')
# plt.show()

# T1 = data_test1['Temperature']
# T2 = data_test2['Temperature']
# T3 = data_test3['Temperature']
# T4 = data_test4['Temperature']
# T5 = data_test5['Temperature']

# T_1 = [i*(T_E - T_surf) + T_surf for i in T1]
# T_2 = [i*(T_E - T_surf) + T_surf for i in T2]
# T_3 = [i*(T_E - T_surf) + T_surf for i in T3]
# T_4 = [i*(T_E - T_surf) + T_surf for i in T4]
# T_5 = [i*(T_E - T_surf) + T_surf for i in T5]

 

# plt.figure(figsize=(14,7))
# plt.plot(T_th1, r, 'blue')
# plt.plot(T_th2, r, 'red')
# plt.plot(T_th3, r, 'orange')
# plt.plot(T_th4, r, 'darkorchid')
# plt.plot(T_th5, r, 'green')

# # plt.plot(T_1, r, 'slateblue')
# # plt.plot(T_2, r, 'brown')
# # plt.plot(T_3, r, 'goldenrod')
# # plt.plot(T_4, r, 'violet')
# # plt.plot(T_5, r, 'darkgreen')
# plt.show()
