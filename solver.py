from typing import Protocol
import numpy as np
from pathlib import Path
import csv

class HeartModel(Protocol):
    def compute_volume(self, activation: float, pressure: float) -> float: ...
        
    def compute_volume_derivation(self, activation: float, pressure: float) -> float: ...
        
    def get_pressure(self) -> float: ...
    
    def get_volume(self) -> float: ...
    
    def save(self): ...

class CirculationModel(Protocol):
    def compute_outflow(self, pressure_current: float, pressure_old: float, dt: float) -> float: ...
        
    def compute_outflow_derivation(self, pressure_current: float, pressure_old: float, dt: float) -> float: ...
        

    

def circulation_solver(heart_model: HeartModel, circulation_model: CirculationModel, activation: np.array, time: np.array, outdir: Path = Path("results"), start_time: int = 0):
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
    
    results_name='results.xdmf'
    outname = Path(outdir) / results_name
    if outname.is_file():
        outname.unlink()
        outname.with_suffix(".h5").unlink()
    
    presures=[]
    volumes=[]
    outflows=[]
    aortic_pressures=[]
    with open(Path(outdir) / 'results_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time [ms]', 'Activation [kPa]', 'Volume [ml]', 'LV Pressure [kPa]', 'Aortic Pressure [kPa]', 'Aortic Pressure Derivation [kPa/ms]','Outflow[ml/ms]'])
        for i, t in enumerate(time):
            # Getting state variable pressure and volume
            p_old=heart_model.get_pressure()
            v_old=heart_model.get_volume()
            # Current activation level
            a_current=activation[i]
            # initial guess for the current pressure pressure
            if i==0 or i==1:
                p_current=p_old
                dt=time[i+1]-time[i]
            else:
                p_current=presures[-1]+(presures[-1]-presures[-2])
                dt=time[i]-time[i-1]
            
            tol=1e-3
            R=[]
            circ_iter=0
            while len(R)==0 or (np.abs(R[-1])>tol and circ_iter<20):
                v_current=heart_model.compute_volume(a_current, p_current)
                outflow=circulation_model.compute_outflow(p_current, p_old, dt)
                v_current_circ=v_old-outflow*dt
                R.append(v_current-v_current_circ)
                # Updataing p_current based on relative error using newton method
                if np.abs(R[-1])>tol:
                    heart_model_derivation=heart_model.compute_volume_derivation(a_current, p_current)     
                    circ_model_derivation=circulation_model.compute_outflow_derivation(p_current, p_old, dt)
                    J=heart_model_derivation+circ_model_derivation
                    p_current=p_current-R[-1]/J
                    circ_iter+=1
            
            p_current=heart_model.get_pressure()
            v_current=heart_model.get_volume()
            if circulation_model.valve_open:
                circulation_model.update_aortic_pressure()
            volumes.append(v_current)
            presures.append(p_current)
            outflows.append(outflow)
            aortic_pressures.append(circulation_model.aortic_pressure)
            heart_model.save(t+start_time,outname)
            writer.writerow([t, a_current, v_current, p_current,circulation_model.aortic_pressure, circulation_model.aortic_pressure_derivation, outflow])
            if p_current<0.01:
                break
        # for time, activation, vol, pres_val, ao_pres_val, flow in zip(time, activation, volumes, presures, aortic_pressures, outflows):
            # writer.writerow([time, activation, vol, pres_val,ao_pres_val, flow])

    return presures, volumes, outflows, aortic_pressures
