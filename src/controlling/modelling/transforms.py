"""Structure-agnostic transformations of state-space models."""
import numpy as np

from src.controlling.modelling.modelconfig import ModelConfig


def add_disturbance_state(base_model: ModelConfig, Gammad=None, Cd=None, Cdz=None, var_dist=1e-1):
    """
    Augments a state-space model with a constant disturbance state.
    Defaults to assuming the disturbance is an external acceleration (like wind).
    """
    Phi = base_model.Phi
    Gamma = base_model.Gamma
    G = base_model.G
    Cy = base_model.Cy
    Cz = base_model.Cz
    R1 = base_model.R1

    nbr_x = Phi.shape[1]
    nbr_w = G.shape[1]

    if Gammad is None:
        Gammad = G
    nbr_d = Gammad.shape[1]

    if Cd is None:
        Cd = np.zeros((Cy.shape[0], nbr_d))
    if Cdz is None:
        Cdz = np.zeros((Cz.shape[0], nbr_d))

    R1d = var_dist * np.eye(nbr_d)

    Phi_extended = np.block([
        [Phi, Gammad],
        [np.zeros((nbr_d, nbr_x)), np.eye(nbr_d)]
    ])

    Gamma_extended = np.vstack([
        Gamma,
        np.zeros((nbr_d, Gamma.shape[1]))
    ])

    G_extended = np.block([
        [G, np.zeros((nbr_x, nbr_d))],
        [np.zeros((nbr_d, nbr_w)), np.eye(nbr_d)]
    ])

    Cy_extended = np.hstack([Cy, Cd])
    Cz_extended = np.hstack([Cz, Cdz])

    R1_extended = np.block([
        [R1, np.zeros((R1.shape[0], R1d.shape[1]))],
        [np.zeros((R1d.shape[0], R1.shape[1])), R1d]
    ])

    return ModelConfig(
        Phi=Phi_extended,
        Gamma=Gamma_extended,
        Cy=Cy_extended,
        Cz=Cz_extended,
        G=G_extended,
        R1=R1_extended,
        R2=base_model.R2,
        deadband_threshold=base_model.deadband_threshold,
        unmeasured_states=base_model.unmeasured_states,
        K_GAIN=base_model.K_GAIN,
        unit=base_model.unit
    )


def extend_matrices_with_disturbance(Phi, Gamma, Gammad):
    """
    Runtime companion to add_disturbance_state: applies the same block
    structure to time-varying Phi, Gamma. The disturbance is modeled as
    constant over each step (random-walk state).
    """
    nbr_x = Phi.shape[0]
    nbr_d = Gammad.shape[1]

    Phi_extended = np.block([
        [Phi, Gammad],
        [np.zeros((nbr_d, nbr_x)), np.eye(nbr_d)]
    ])

    Gamma_extended = np.vstack([
        Gamma,
        np.zeros((nbr_d, Gamma.shape[1]))
    ])

    return Phi_extended, Gamma_extended
