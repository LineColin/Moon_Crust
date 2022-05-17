#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:57:18 2022

@author: linecolin
"""

import numpy as np
from matplotlib import rcParams
from scipy import sparse
import scipy.sparse.linalg as LA


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
    rcParams['font.size'] = 20
    
    return


# =================================


def Diffusion(y, n, dt, dy, K, T_top, T_bot):

    R = np.zeros(n)
    
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
    
    return M, R, r_center

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
    H_TOT_verif = [H_TOT[0]/(rho*V_MOON)]
    h_LMO_t = [h_LMO]
    
    F_Lat = []
    F_cond_bot = []
    F_cond_top = []
    F_h_LMO = []
    F_h_CRUST = []
    F_int = []
    
    
    i = 1
    
    while R_bot-R_SC > 200 and t/3.15E13 < 500:
        
    
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
        H_TOT_verif.append(H_TOT[i]/(rho*V_MOON))
        
        
        # TEMPERATURE (diffusion + advection)
        
        L = R_top - R_bot
    
        dr = L/n
    
        u_T = dR_bot*(y - 2)/((R_top - R_bot))
        
        u_h = dR_bot*(y-2)/(R_top - R_bot)
        
        
        L, R, r_center = Diffusion(y, n, dt, dy, K, T_top, T_bot)
        
        
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
        F_cond_bot.append(-k*dT*4*np.pi*R_bot**2 *(T[-2]-T[-1])/dr)
        F_cond_top.append(k*dT*4*np.pi*R_top**2 *(T[1]-T[0])/dr)
        F_h_LMO.append(H_LMO[i])
        F_h_CRUST.append(h_C)
        #F_int.append()
        
        i+=1
        
        
        
        ### CUMULATS ###
        
    data = {'time':time, 'Temperature':T, 'h(r, t=tf)': h_r, 
            'r CRUST':radius_CRUST, 'r CUMULATS': radius_SC, 
            'VOLUME TOT':VOLUME_TOT, 'VOLUME CRUST': VOLUME_CRUST, 'VOLUME SC':VOLUME_SC,
            'VOLUME LMO':VOLUME_LMO, 'H TOT': H_TOT, 'H CRUST':H_CRUST, 
            'H LMO':H_LMO, 'h TOT':H_TOT_verif, 'dR/dt':dR_dt, 'h LMO':h_LMO_t, 'F Lat':F_Lat,
            'F Cond bot':F_cond_bot, 'F Cond top':F_cond_top, 
            'F h LMO': F_h_LMO, 'F h CRUST':F_h_CRUST,
            'F int':F_int}    
        
    print('end')
    
    return data

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

