
# cryo_model.py
# Simulation of cell viability during cryopreservation
# Author: Mudassir Waheed, 2025

import numpy as np
from scipy.integrate import odeint

# Default model parameters
params = {
    "initial_water_volume": 1.0,
    "initial_solute_concentration": 1.0,
    "cooling_rate": -1.0,  # deg C/min
    "freezing_threshold": -10.0,
    "toxicity_threshold": 2.5,
    "critical_ice_fraction": 0.6,
    "simulation_time": 60,  # minutes
    "steps": 500
}

def temp_profile(t, cooling_rate):
    return 0 + cooling_rate * t  # Linear cooling from 0°C

def cell_ode(y, t, params):
    water_volume, ice_fraction, solute_conc = y
    T = temp_profile(t, params["cooling_rate"])

    # Ice formation model
    if T < params["freezing_threshold"]:
        d_ice = 0.02 * abs(T - params["freezing_threshold"]) * water_volume
    else:
        d_ice = 0

    d_water = -d_ice
    new_water_volume = max(water_volume + d_water, 0.01)

    # Osmotic effect
    new_conc = solute_conc * (water_volume / new_water_volume)

    return [d_water, d_ice, new_conc - solute_conc]

def simulate_cell_viability(custom_params=None):
    cfg = params.copy()
    if custom_params:
        cfg.update(custom_params)

    t = np.linspace(0, cfg["simulation_time"], cfg["steps"])
    y0 = [cfg["initial_water_volume"], 0.0, cfg["initial_solute_concentration"]]

    sol = odeint(cell_ode, y0, t, args=(cfg,))
    water, ice, solute = sol.T

    # Calculate viability
    viability = np.ones_like(t)
    viability[(solute > cfg["toxicity_threshold"]) | (ice > cfg["critical_ice_fraction"])] = 0

    return {
        "time": t,
        "temperature": temp_profile(t, cfg["cooling_rate"]),
        "water_volume": water,
        "ice_fraction": ice,
        "solute_concentration": solute,
        "viability": viability
    }
