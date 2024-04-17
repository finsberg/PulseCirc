from typing import Protocol
import numpy as np

class HeartModel(Protocol):
    def compute_volume(self, activation: float, pressure: float) -> float: ...
        
    def compute_volume_derivation(self, activation: float, pressure: float) -> float: ...
        
    def get_pressure(self) -> float: ...
    
    def get_volume(self) -> float: ...
    
    def save(self): ...

class CirculationModel(Protocol):
    def compute_outflow(self, pressure: float) -> float: ...
        
    def compute_outflow_derivation(self, pressure: float) -> float: ...
        


def circulation_solver(heart_model: HeartModel, circulation_model: CirculationModel, activation: np.array, time: np.array):
    """
    Solves the coupled cardiac and circulation model dynamics over a specified time period.

    Parameters:
    heart_model (HeartModel): Instance of HeartModel for cardiac dynamics.
    circulation_model (CirculationModel): Instance of CirculationModel for circulation dynamics.
    activation (np.array): Array of activation levels corresponding to time points.
    time (np.array): Array of time points for the simulation.

    Raises:
    ValueError: If the lengths of time and activation arrays do not match.
    """
    
    if not len(time)==len(activation):
        raise ValueError(
            ("Please provide the time series for each activation value (the length of time and activation should be the same!)"),
            )
    
    presures=[]
    volumes=[]
    outflows=[]
    tol=1e-5*v_old
    
    for i, t in enumerate(time):
        # Getting state variable pressure and volume
        p_old=heart_model.get_pressure()
        v_old=heart_model.get_volume()
        # Current activation level
        a_current=activation[i]
        # initial guess for the current pressure pressure
        if i==0 or i==1:
            p_current=p_old*1.01
        else:
            p_current=presures[-1]+(presures[-1]-presures[-2])
            
        R=[]
        circ_iter=0
        while len(R)==0 or (np.abs(R[-1])>tol and circ_iter<20):
            v_current=heart_model.compute_volume(a_current, p_current)
            outflow=circulation_model.compute_outflow(p_current)
            dt=t[i+1]-t[i]
            v_current_circ=v_old-outflow*dt
            R.append((v_current-v_current_circ)/v_current)
            # Updataing p_current based on relative error using newton method
            if np.abs(R[-1])>tol:
                heart_model_derivation=heart_model.compute_volume_derivation(a_current, p_current)     
                circ_model_derivation=circulation_model.compute_outflow_derivation(p_current)
                J=heart_model_derivation+circ_model_derivation
                p_current=p_current-R[-1]/J
                circ_iter+=1
        
        p_current=heart_model.get_pressure()
        v_current=heart_model.get_volume()
        volumes.append(v_current)
        presures.append(p_current)
        outflows.append(outflow)
