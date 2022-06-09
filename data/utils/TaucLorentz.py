"""
A unit for calculation of optical spectra according to Tauc-Lorentz model
(Jellison and Modine, APL 69, 371-373 (1996)). Sample material parameters from
J.Appl.Phys. 106, 103509 (2009). Energy in eV.

Yuri Vorobyov
juriy.vorobjov@gmail.com
"""
import numpy as np

def TL_nk(wl, param):
    '''
    Single Tauc-Lorentz oscialltor model
    Constraints on parameter values: 
        G < np.sqrt(2)*E_0; E_g < E_0
        e_inf (all simulation is set to 1, usually it's larger than 1)

    Parameters
    ----------
    wl : numpy ndarray
        Wavelengths.
    param : list
        list of parameter values.

    Returns
    -------
    tuple of numpy ndarrays
        A tuple of four numpy ndarrays that are respectively the real part and 
        imaginary part of the wavelength-resolved dielectric function, and the 
        real part and imaginary part of the wavelength-resolved refractive index.

    '''
    
    h = 4.135667662e-15
    c = 299792458
    e = h * c / (wl*1e-9)
    
    
    A = param[0]
    E_0 = param[1]
    G = param[2]
    E_g = param[3]
    e_inf = param[4]
    
    # auxiliary functions and variables
    def a_ln(E):
        t1 = (E_g**2 - E_0**2) * E**2
        t2 = E_g**2 * G**2
        t3 = -E_0**2 * (E_0**2 + 3 * E_g**2)
        return t1 + t2 + t3
    
    def a_atan(E):
        t1 = (E**2 - E_0**2) * (E_0**2 + E_g**2)
        t2 = E_g**2 * G**2
        return t1 + t2
    
    alpha = (4 * E_0**2 - G**2)**0.5
    
    gamma = (E_0**2 - 0.5 * G**2)**0.5
    
    def ksi(E):
        t1 = np.power(np.power(E, 2) - gamma**2, 2)
        t2 = 0.25 * alpha**2 * G**2
        return np.power(t1 + t2, 0.25)
    
    def e_re(E):
        """Dielectric function (real part)"""
        t1 = e_inf
        t2 = A * G * a_ln(E) / (2 * np.pi * ksi(E)**4 * alpha * E_0) * np.log((E_0**2 + E_g**2 + alpha * E_g) / (E_0**2 + E_g**2 - alpha * E_g))
        t3 = -A * a_atan(E) / (np.pi * ksi(E)**4 * E_0) * (np.pi - np.arctan(1 / G * (2 * E_g + alpha)) + np.arctan(1 / G * (alpha - 2 * E_g)))
        t4 = 4 * A * E_0 * E_g * (E**2 - gamma**2) / (np.pi * ksi(E)**4 * alpha) * (np.arctan(1 / G * (alpha + 2 * E_g)) + np.arctan(1 / G * (alpha - 2 * E_g)))
        t5 = -A * E_0 * G * (E**2 + E_g**2) / (np.pi * ksi(E)**4 * E) * np.log(np.fabs(E - E_g) / (E + E_g))
        t6 = 2 * A * E_0 * G * E_g / (np.pi * ksi(E)**4) * np.log(np.fabs(E - E_g) * (E + E_g) / ((E_0**2 - E_g**2)**2 + E_g**2 * G**2)**0.5)
        return t1 + t2 + t3 + t4 + t5 + t6
    
    def e_im(E):
        """Dielectric function (imaginary part)"""
        result = 1 / E * A * E_0 * G * (E - E_g)**2 / ((E**2 - E_0**2)**2 + G**2 * E**2)
        out = np.where(E > E_g, result, 0)
        return out
        
    def n(E):
        """Refractive index (real part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) + e_re(E)))
        return np.real_if_close(out)
    
    def k(E):
        """Refractive index (imaginary part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) - e_re(E)))
        return np.real_if_close(out)
    
    return e_re(e), e_im(e), n(e), k(e)