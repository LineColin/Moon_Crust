#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:13:46 2022

@author: linecolin
"""

import numpy as np
from matplotlib import rcParams
from scipy import sparse
import scipy.sparse.linalg as LA
import scipy.integrate as sc
import matplotlib.pyplot as plt
import CumulatFormation as CU


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


def Diffusion(y, n, dt, dy, K, T_top, T_bot, R_top, R_bot):

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

def Newton( f, x, T, dy, param, h=1.0E-6, epsilon=1.0E-8):
    NbIterationMax = 100
    n = 0
    while ( np.abs( f(x, T, dy, param) ) > epsilon ) and ( n < NbIterationMax ):
        x = x - 2 * h * f( x , T, dy, param) / ( f( x + h, T, dy, param) - f( x - h , T, dy, param) )
        n += 1
    return x if n < NbIterationMax else None


# =================================

def func_Ts(Ts, T, dy, param):
    
    k = param['k']
    a = 2*T + Ts
    
    return sigma*e*(Ts**4 - T_eq**4) - 2*k*(T*(T_E - T_eq) + T_eq -Ts)/dy

# =================================

def h_anal(r, D, h0):
    
    n = len(r)
    c_r = np.zeros(n)
    
    for i in range(n):
        
        c_r[i] = D*h0*((R_MOON**3 - R_CORE**3)/(R_MOON**3 - r[i]**3))**(1-D)
    
    return c_r

# =================================

def Evolution(n, param):
    
    
    print('start')
    
    ### VARIABLE PARAMETERS ###
    
    k_CRUST = param['k']
    k_SC = 4
    rho = param['rho']
    Cp = param['Cp']
    Lat = param['Lat']
    K_CRUST = k_CRUST/(rho*Cp)
    K_SC = k_SC/(rho*Cp)
    
    D = param['D_AN']
    Lambda = param['lambda']
    
    C_E = param['C_E']
    C_0 = param['C_0']
    
    ### INITIAL PARAMETERS ###
    
    L = param['L0']
    dR_CRUST = param['dR_CRUST0']
    
    ### INITIAL CONDITIONS ###
    
    # import the data from first stage 
    
    data = CU.Evolution(D, C_0, C_E)
    
    n_SC = n
    
    n_it = int(len(data['Temp'])/n_SC + 1)
    
    # radius 
    
    R_CRUST = R_MOON - L
    R_SC = pow((1-C_E)*(R_MOON**3 - R_CRUST**3)/C_E + data['radius'][-1]**3, 1/3)
    
    #volume
    
    V_LMO_0 = 4*np.pi*(R_MOON**3 - data['radius'][-1]**3)/3
    V_LMO = 4*np.pi*(R_CRUST**3 - R_SC**3)/3
    dV_CRUST = 4*np.pi*(R_MOON**3 - R_CRUST**3)/3
    dV_SC = (1 - C_E)*dV_CRUST/C_E
    
    phi_CRUST = dV_CRUST/V_LMO_0
    phi_SC = dV_SC/V_LMO_0
    
    # temperature
    
    T_E = int(data['Temp'][-1])
    print(T_E)
    T_S = data['T surface'][-1]
    T_CORE = data['Temp'][0]
    print(T_S)
    
    dT_CRUST = T_E - T_eq
    dT_SC = T_CORE - T_E
    
    T0_CRUST = np.linspace(T_S, T_E, n)
    T0_SC = data['Temp'][::n_it]
    
    
    # radiogenic heat
    h_LMO_0 = data['H LMO'][-1]
    print(h_LMO_0)
    h_LMO = h_LMO_0/(D*(phi_CRUST+phi_SC) + 1 - phi_CRUST - phi_SC)
    h_CRUST = D*h_LMO
    h_SC = D*h_LMO
    print('h_SC', h_SC)
    print('h(r)', data['h(r)'][-1])
    
    ### SCALING ###
    
    # temperature
    
    T0_CRUST = (T0_CRUST - T_eq) / dT_CRUST
    plt.figure()
    plt.plot(T0_CRUST)
    
    Ttop_CRUST = T0_CRUST[0]
    Tbot_CRUST = T0_CRUST[-1]
    
    
    T0_SC = [(i - T_E)/dT_SC for i in T0_SC] #(T0_SC - T_E) / dT_SC
    
    Ttop_SC = T0_SC[0]
    Tbot_SC = T0_SC[-1]
    
    # space 
    
    r_CRUST = np.linspace(R_MOON, R_CRUST, n)
    y_CRUST = (r_CRUST - R_CRUST)/(R_MOON - R_CRUST) + 1
    l_CRUST = y_CRUST[-1] - y_CRUST[0]
    
    r_SC = np.linspace(R_CORE, R_SC, n_SC)
    y_SC = (r_SC - R_CORE)/(R_SC - R_CORE) + 1
    l_SC = y_SC[-1] - y_SC[0]
    
    L_CRUST = R_MOON - R_CRUST
    L_SC = R_SC - R_CORE
    
    dr_CRUST = L_CRUST / n
    dr_SC = L_SC / n_SC
    
    dy_CRUST = l_CRUST / n
    dy_SC = l_SC / n_SC
    
    plt.figure()
    plt.plot(T0_CRUST, r_CRUST)
    print(T0_CRUST[0])
    # time
    
    t= 0
    
    cfl = 0.4
    dt = cfl*dr_CRUST**2 /K_CRUST
    
    ### CALCULATION MATRIX ###
    
    fig, ax = plt.subplots(ncols=1, nrows = 3, figsize=(10,10), sharex=False)
    
    I = np.identity(n)
    
    hr_CRUST = np.ones(n)*h_CRUST
    hr_SC = data['h(r)'][::n_it]
    hr_SC[-1] = h_SC
    
    #ax[1].plot(r_SC, hr_SC, 'black')
    
    T_CRUST = T0_CRUST.copy()
    T_SC = T0_SC.copy()
    
    ### DATA STORAGE ###
    
    time = [t]
    
    radius_CRUST = [R_CRUST/1000]
    radius_SC = [R_SC/1000]
    
    VOLUME_LMO = [V_LMO]
    VOLUME_CRUST = [dV_CRUST]
    VOLUME_SC = [dV_SC + 4*np.pi*(data['radius'][-1]**3 - R_CORE**3)/3]
    VOLUME_TOT = [4*np.pi*(R_MOON**3 - R_CORE**3)/3]

    a_CRUST = h_CRUST*dV_CRUST
    a_SC = h_SC*dV_SC
    
    H_LMO = [h_LMO*V_LMO]
    H_CRUST = [h_CRUST*dV_CRUST]
    H_SC = [h_SC*dV_SC]
    H_TOT = [h_LMO_0*V_LMO_0]
    
    Ts = [T_S]
    
    i = 1
    
    
    
    while R_CRUST - R_SC > 500 and t/3.15E13 < 500 :
        
        
        # surface temperature
        
        dT_CRUST = T_E - T_S
        
        if T_S > T_eq :
            
            T_S = Newton(func_Ts, T_S, T_CRUST[1], dr_CRUST, param)
        
        Ts.append(T_S)
        
        # volume of LMO at precedent time step
        
        V_LMO_m = 4*np.pi*(R_CRUST**3 -  R_SC**3)/3
        
        # time step 
        
        a = cfl*dr_CRUST**2 /K_CRUST
        b = cfl*dr_CRUST/np.abs(dR_CRUST)
        
        dt = np.min([a, b])
        
        t += dt
        time.append(t/3.15E13)
        
        # crust and cumulates evolution 
        
        dR_CRUST = k_CRUST*(T_CRUST[-2] - T_CRUST[-1])*dT_CRUST*C_E/(dr_CRUST*rho*Lat) + h_LMO*np.exp(-Lambda*t/3.15E7)*C_E*V_LMO/(rho*Lat*4*np.pi*R_CRUST**2)
        dR_SC = -(1-C_E)*dR_CRUST*R_CRUST**2 /(C_E*R_SC**2)
        
        R_SC += dt*dR_SC
        R_CRUST += dR_CRUST*dt
        
        L_CRUST = R_MOON - R_CRUST
        L_SC = R_SC - R_CORE
        
        r_CRUST = np.linspace(R_MOON, R_CRUST, n)
        r_SC = np.linspace(R_CORE, R_SC, n_SC)
        
        dr_CRUST = L_CRUST / n
        dr_SC = L_SC / n_SC
        
        radius_CRUST.append(R_CRUST/1000)
        radius_SC.append(R_SC/1000)
        
        u_CRUST =  dR_CRUST*(y_CRUST - 2)/(R_MOON - R_CRUST)
        u_SC = - dR_SC*(y_SC - 1)/(R_SC - R_CORE)
        
        # volume
        
        V_LMO = 4*np.pi*(R_CRUST**3 - R_SC**3)/3
        
        dV_CRUST = - 4*np.pi*dR_CRUST*dt*R_CRUST**2
        dV_SC = (1 - C_E)*dV_CRUST/C_E 
        
        VOLUME_LMO.append(V_LMO)
        VOLUME_CRUST.append(4*np.pi*(R_MOON**3 - R_CRUST**3)/3)
        VOLUME_SC.append(4*np.pi*(R_SC**3 - R_CORE**3)/3)
        VOLUME_TOT.append(VOLUME_CRUST[i] + VOLUME_SC[i] + V_LMO)
        
        phi_CRUST = dV_CRUST/V_LMO_m
        phi_SC = dV_SC/V_LMO_m
        
        # radiogenic heat
        
        h_LMO = h_LMO/(D*(phi_CRUST + phi_SC) + 1 - phi_CRUST - phi_SC)
        h_CRUST = D*h_LMO
        h_SC = D*h_LMO
        
        D_CRUST, res_CRUST, rc_CRUST = Diffusion(y_CRUST, n, dt, dy_CRUST, K_CRUST, Ttop_CRUST, Tbot_CRUST, R_MOON, R_CRUST)
        D_SC, res_SC, rc_SC = Diffusion(y_SC, n_SC, dt, dy_SC, K_SC, Ttop_SC, Tbot_SC, R_SC, R_CORE)
        
        hr_CRUST[-1] = h_CRUST
        hr_SC[-1] = h_SC
        
        A_CRUST = Advection(u_CRUST, dy_CRUST, dt)
        A_SC = Advection(u_SC, dy_SC, dt)
        
        N_CRUST = I + A_CRUST
        N_SC = np.identity(n_SC) + A_SC
        
        N_CRUST = sparse.csc_matrix(N_CRUST)
        N_SC = sparse.csc_matrix(N_SC)
        
        hr_CRUST = LA.spsolve(N_CRUST, hr_CRUST)
        hr_SC = LA.spsolve(N_SC, hr_SC)
        #hr_SC = h_anal(rc_SC, D, param['h0'])
        
        a_CRUST += h_CRUST*dV_CRUST*np.exp(-Lambda*t/3.15E7)
        a_SC += h_SC*dV_SC*np.exp(-Lambda*t/3.15E7)
        
        H_CRUST.append(a_CRUST)
        H_SC.append(a_SC)
        H_LMO.append(h_LMO*V_LMO*np.exp(-Lambda*t/3.15E7))
        H_TOT.append(H_CRUST[i] + H_SC[i] + H_LMO[i])
        
        # temperature
        
        #D_CRUST, res_CRUST, rc_CRUST = Diffusion(y_CRUST, n, dt, dy_CRUST, K_CRUST, Ttop_CRUST, Tbot_CRUST, R_MOON, R_CRUST)
        #D_SC, res_SC, rc_SC = Diffusion(y_SC, n, dt, dy_SC, K_SC, Ttop_SC, Tbot_SC, R_SC, R_CORE)
        
        V_CRUST = T_CRUST + res_CRUST + (dt/(rho*Cp*dT_CRUST))*hr_CRUST*np.exp(-Lambda*t/3.15E7)
        M_CRUST = I - D_CRUST + A_CRUST
        M_CRUST = sparse.csc_matrix(M_CRUST)
        T_CRUST = LA.spsolve(M_CRUST, V_CRUST)
        
        V_SC = T_SC + res_SC + (dt/(rho*Cp*dT_SC))*hr_SC*np.exp(-Lambda*t/3.15E7)
        M_SC = np.identity(n_SC) - D_SC + A_SC
        M_SC = sparse.csc_matrix(M_SC)
        T_SC = LA.spsolve(M_SC, V_SC)
        
        Ttop_CRUST = (T_S - T_eq)/(T_E - T_eq)
        
        
        
        if i%5000 ==  0 : 
            ax[0].plot(T_CRUST, rc_CRUST)
            ax[1].plot(hr_SC, rc_SC)
            ax[2].plot(hr_CRUST, rc_CRUST)
            print(Ttop_CRUST)
            
        i += 1
        
    data2 = {'time':time, 'radius CRUST':radius_CRUST, 'radius SC':radius_SC,
            'T CRUST':T_CRUST, 'T SC':T_SC, 'h(r) CRUST':hr_CRUST, 
            'h(r) SC':hr_SC, 'H LMO':H_LMO, 'H CRUST':H_CRUST, 'H SC':H_SC, 'H TOT':H_TOT,
            'V LMO':VOLUME_LMO, 'V CRUST':VOLUME_CRUST, 'V SC':VOLUME_SC, 'V TOT':VOLUME_TOT, 
            'T surface':Ts, 'y CRUST':r_CRUST, 'y SC':r_SC}
    
    print('end')
    
    return data, data2
        
        
### GLOBAL PARAMETERS ###

R_MOON = 1737E3
R_CORE = 350E3

V_MOON = 4*np.pi*(R_MOON**3 - R_CORE**3)/3

T_E = 1585


sigma = 5.67E-8
e = 1

R_SUN = 700E3
A_MOON = 0.12
D_SUN = 1495E5
T_SUN = 5780

T_eq = T_SUN * np.sqrt(R_SUN/(2*D_SUN)) * (1 - A_MOON)**0.25  
