from typing import Protocol
import numpy as np
from scipy.integrate import solve_ivp

class Heart_Model(Protocol):
    def compute_volume(self, activation: float, pressure: float) -> float: ...
        
    def compute_volume_derivation(self, activation: float, pressure: float) -> float: ...
        
    def get_pressure(self): ...
    
    def get_volume(self): ...
    
    def save(self): ...

class Circulation_Model(Protocol):
    def compute_outflow(self, pressure: float) -> float: ...
        
    def compute_outflow_derivation(self, pressure: float): ...
        


def circulation_solver(Heart_Model: Heart_Model, Circulation_Model: Circulation_Model, activation: np.array, time: np.array):
    """
    The function couple a cardiac pressure model (Model), i.e., a finite element model of heart, driven with activation with a circulation model (Circ)

    """
    
    if not len(time)==len(activation):
        raise ValueError(
            ("Please provide the time series for each activation value (the length of time and activation should be the same!)"),
            )
    
    pres=[]
    vols=[]
    outflows=[]
    
    for i, t in enumerate(time):
        # Getting state variable pressure and volume
        p_old=Heart_Model.get_pressure()
        v_old=Heart_Model.get_volume()
        # Current activation level
        a_current=activation[i]
        # initial guess for the current pressure pressure
        if i==0 or i==1:
            p_current=p_old*1.01
        else:
            p_current=pres[-1]+(pres[-1]-pres[-2])
            
        R=[]
        tol=1e-5*v_old
        while len(R)==0 or (np.abs(R[-1])>tol and circ_iter<20):
            v_current=Heart_Model.compute_volume(a_current, p_current)
            outflow=Circulation_Model.compute_outflow(p_current)
            dt=t[i+1]-t[i]
            v_current_circ=v_old-outflow*dt
            R.append((v_current-v_current_circ)/v_current)
            # Updataing p_current based on relative error using newton method
            if np.abs(R[-1])>tol:
                heart_model_derivation=Heart_Model.compute_volume_derivation(a_current, p_current)     
                circ_model_derivation=Circulation_Model.compute_outflow_derivation(p_current)
                J=heart_model_derivation+circ_model_derivation
                p_current=p_current-R[-1]/J
                circ_iter+=1
        
        p_current=Heart_Model.get_pressure()
        v_current=Heart_Model.get_volume()
        vols.append(v_current)
        pres.append(p_current)
        outflows.append(outflow)
