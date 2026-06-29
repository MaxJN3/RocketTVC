from src.controlling.modelling.models import ModelConfig
import numpy as np

def extend_model(base_model: ModelConfig, Gammad, Cd, Cdz, R1d, deadband_threshold):
    Phi = base_model.Phi
    Gamma = base_model.Gamma
    G = base_model.G
    Cy = base_model.Cy
    Cz = base_model.Cz
    R1 = base_model.R1
    
    nbr_x = Phi.shape[1]
    nbr_d = Gammad.shape[1]
    
    Phi_extended = np.block([
                            [Phi, Gammad],                            # x_e = [x, d]
                            [np.zeros((nbr_d, nbr_x)), np.eye(nbr_d)] # Phi_e = [[Phi, Gammad], [0, I]]
                            ])
    Gamma_extended = np.vstack([
        Gamma,
        np.zeros((nbr_d, Gamma.shape[1]))
    ])
    G_extended = np.block([
        [G, np.zeros((nbr_x, nbr_d))],
        [np.zeros((nbr_d, nbr_x)), np.eye(nbr_d)]
    ])  
    
    Cy_extended = np.hstack([Cy, Cd])
    Cz_extended = np.hstack([Cz, Cdz])
    
    R1_extended = np.block([
        [R1, np.zeros((R1.shape[0], R1d.shape[1]))],
        [np.zeros((R1d.shape[0], R1.shape[1])), R1d]
    ])
    
    extended_model = ModelConfig(
        Phi=Phi_extended, Gamma=Gamma_extended, G=G_extended, Cy=Cy_extended, Cz=Cz_extended,
         R1=R1_extended, R2=base_model.R2, deadband_threshold=deadband_threshold
    )
    
    return extended_model