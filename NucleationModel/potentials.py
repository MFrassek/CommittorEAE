from openpathsampling.engines.toy.pes import PES, PES_Add, OuterWalls, Gaussian
import numpy as np


class DoublewellPotential(PES_Add):
    def __init__(self):
        super(DoublewellPotential, self).__init__(
            OuterWalls([1.0, 1.0], [0.0, 0.0]),
            PES_Add(
                Gaussian(-0.7, [7.5, 7.5], [-0.5, 0.0]),
                Gaussian(-0.7, [7.5, 7.5], [0.5, 0.0])))


class ZPotential(PES):
    """Z-Potential energy surface as described by Buijsman and Bolhuis (2020):
    https: /  / aip.scitation.org / doi / pdf / 10.1063 / 1.5130760

    :math:'V(x,y) = \frac{x^{4} + y^{4}}{20480}
        - 3 e^{-0.01(x + 5)^{2} - 0.2(y + 5)^{2}}
        - 3 e^{-0.01(x - 5)^{2} - 0.2(y - 5)^{2}} 
        + \frac{5 e^{-0.2(x + 3(y - 3))^{2}}}{1 + e^{-x - 3}}
        + \frac{5 e^{-0.2(x + 3(y + 3))^{2}}}{1 + e^{x - 3}}
        +3 e^{-0.01\left(x^{2} + y^{2}\right)}'
    Parameters
    ----------
    dim: integer
         number of dimensions
    """

    def __init__(self):
        super(ZPotential, self).__init__()
        self._local_dVdx = np.zeros(2)

    def __repr__(self):
        return "Z-Potential"

    def V(self, sys):
        """Potential energy

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        float
            the potential energy
        """
        pos = sys.positions
        myV = (pos[0]**4 + pos[1]**4) / 20480 \
            - 3 * np.exp(-0.01 * (pos[0] + 5)**2 - 0.2 * (pos[1] + 5)**2) \
            - 3 * np.exp(-0.01 * (pos[0] - 5)**2 - 0.2 * (pos[1] - 5)**2) \
            + (5 * np.exp(-0.2 * (pos[0] + 3 * (pos[1] - 3))**2)) \
            / (1 + np.exp(-pos[0] - 3)) \
            + (5 * np.exp(-0.2 * (pos[0] + 3 * (pos[1] + 3))**2)) \
            / (1 + np.exp(pos[0] - 3)) \
            + 3 * np.exp(-0.01 * (pos[0]**2 + pos[1]**2))
        return myV

    def dVdx(self, sys):
        """Derivative of potential energy (-force)

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        np.array
            the derivatives of the potential at this point
        """
        pos = sys.positions
        self._local_dVdx[0] = pos[0]**3 / 5120 - 0.06 * pos[0] \
            * np.exp(-0.01 * (pos[0]**2 + pos[1]**2)) + 0.06 * (pos[0] - 5) \
            * np.exp(-0.01 * (pos[0] - 5)**2 - 0.2 * (pos[1] - 5)**2) \
            + 0.06 * (pos[0] + 5) \
            * np.exp(-0.01 * (pos[0] + 5)**2 - 0.2 * (pos[1] + 5)**2) \
            - (40.1711 * np.exp(pos[0] - 0.2 * (pos[0] + 3 * pos[1] - 9)**2)
                * (pos[0] + 3 * pos[1] - 9)) / (np.exp(pos[0] + 3) + 1) \
            - (40.1711 * np.exp(-0.2 * (pos[0] + 3 * pos[1] + 9)**2)
                * (pos[0] + 3 * pos[1] + 9)) / (np.exp(pos[0]) + np.exp(3)) \
            - (5 * np.exp(-0.2 * (pos[0] + 3 * pos[1] + 9)**2 + pos[0] + 3)) \
            / (np.exp(pos[0]) + np.exp(3))**2 \
            + (5 * np.exp(-0.2 * (pos[0] + 3 * pos[1] - 9)**2 + pos[0] + 3)) \
            / (np.exp(pos[0] + 3) + 1)**2

        self._local_dVdx[1] = pos[1]**3 / 5120 \
            - 0.06 * pos[1] * np.exp(-0.01 * (pos[0]**2 + pos[1]**2)) \
            + 1.2 * (pos[1] - 5) \
            * np.exp(-0.01 * (pos[0] - 5)**2 - 0.2 * (pos[1] - 5)**2) \
            + 1.2 * (pos[1] + 5) \
            * np.exp(-0.01 * (pos[0] + 5)**2 - 0.2 * (pos[1] + 5)**2) \
            - (120.513 * np.exp(pos[0] - 0.2 * (pos[0] + 3 * pos[1] - 9)**2)
                * (pos[0] + 3 * pos[1] - 9)) / (np.exp(pos[0] + 3) + 1) \
            - (120.513 * np.exp(-0.2 * (pos[0] + 3 * pos[1] + 9)**2)
                * (pos[0] + 3 * pos[1] + 9)) / (np.exp(pos[0]) + np.exp(3))
        return self._local_dVdx
