import numpy as np

# Monkeypatch np.asscalar for compatibility with colormath on newer NumPy versions
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def hex_to_lab(hex_code):
    rgb = hex_to_rgb(hex_code)
    r, g, b = [x / 255.0 for x in rgb]
    srgb = sRGBColor(r, g, b)
    lab = convert_color(srgb, LabColor)
    return [lab.lab_l, lab.lab_a, lab.lab_b]

def rgb_to_lab(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    srgb = sRGBColor(r, g, b)
    lab = convert_color(srgb, LabColor)
    return np.array([lab.lab_l, lab.lab_a, lab.lab_b])

def delta_e_cie2000_vec(lab1, lab_array):
    """Compute real CIE2000 Delta E between lab1 and each row in lab_array using a vectorized NumPy implementation."""
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2 = lab_array[:, 0]
    a2 = lab_array[:, 1]
    b2 = lab_array[:, 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.arctan2(b1, a1_prime) % (2.0 * np.pi)
    h2_prime = np.arctan2(b2, a2_prime) % (2.0 * np.pi)

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = h2_prime - h1_prime
    dh_prime = np.where(np.abs(dh_prime) > np.pi, 
                        np.where(h2_prime <= h1_prime, dh_prime + 2.0*np.pi, dh_prime - 2.0*np.pi), 
                        dh_prime)

    dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(dh_prime / 2.0)

    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h_bar_prime = (h1_prime + h2_prime) / 2.0
    h_bar_prime = np.where(np.abs(h1_prime - h2_prime) > np.pi,
                           np.where(h1_prime + h2_prime < 2.0 * np.pi, h_bar_prime + np.pi, h_bar_prime - np.pi),
                           h_bar_prime)

    T = 1.0 - 0.17 * np.cos(h_bar_prime - np.radians(30.0)) + 0.24 * np.cos(2.0 * h_bar_prime) + 0.32 * np.cos(3.0 * h_bar_prime + np.radians(6.0)) - 0.20 * np.cos(4.0 * h_bar_prime - np.radians(63.0))

    S_L = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(20.0 + (L_bar_prime - 50.0)**2)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T

    dtheta = np.radians(30.0) * np.exp(-((np.degrees(h_bar_prime) - 275.0) / 25.0)**2)
    R_C = 2.0 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25.0**7))
    R_T = -np.sin(2.0 * dtheta) * R_C

    dE = np.sqrt((dL_prime / S_L)**2 + (dC_prime / S_C)**2 + (dH_prime / S_H)**2 + R_T * (dC_prime / S_C) * (dH_prime / S_H))
    return dE

def delta_e_cie2000(lab1, lab2):
    """Compute Delta E CIE2000 between two individual LAB colors using the vectorized engine."""
    return float(delta_e_cie2000_vec(np.array(lab1), np.array([lab2]))[0])
