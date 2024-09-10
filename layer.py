from rcwa.shorthand import *
from rcwa import Material, Crystal, MatrixCalculator
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from numpy.typing import ArrayLike
from matplotlib.figure import Figure, Axes

# TODO: Convolution matrix generation must be refactored. It's a hot mess and hard to understand.

USEGRAD = 1

from time import sleep

class Layer(MatrixCalculator):
    """
    Class for defining a single layer of a layer stack used in a simulation

    :param er: Permittivity of the layer. Overridden by crystal permittivity if specified.
    :param ur: Permeability of the layer. Overridden by crystal permeability if specified.
    :param thickness: Thickness of the layer
    :param n: Refractive index of the layer. Overridden by cristal er/ur if specified.
    :param material: Material object containing the material's permittivity and permeability as a function of wavelength/angle.
    :param crystal: Crystal object if the layer is periodic in x and/or y. Overrides er, ur, n, and material
    """
    def __init__(self, er: complex = torch.tensor(1.0, dtype = torch.cdouble), ur: complex = torch.tensor(1.0, dtype = torch.cdouble), thickness: complex = 0.0, n: Union[complex, None] = None,
                 material: Union[None, Material] = None, crystal: Union[None, Crystal] = None):
        # TODO: this should not be set so flagrantly. Does this eliminate the er information?
        if material is None:
            print(f"This is n: {n}")
            if n:
                n = torch.tensor(n, dtype = torch.cdouble)
            self.material = Material(er=er, ur=ur, n=n) # Is this the issue?
            print("Material is none")
        else:
            print("Material is not none")
            self.material = material
        print(f"This is whether the er is a tensor: {torch.is_tensor(self.er)}")
        self.thickness = thickness
        self.crystal = crystal
        self.incident = False  # Whether this is a transmission layer
        self.transmission = False  # Whether this is an incident layer

        if crystal is None:
            self.homogenous = True
        else:
            self.homogenous = False


    # Note: these are all just transparent wrappers for underlying material
    @property
    def er(self) -> Union[ArrayLike, complex]:
        return self.material.er

    @er.setter
    def er(self, er: complex):
        self.material.er = er

    @property
    def ur(self):
        return self.material.ur

    @ur.setter
    def ur(self, ur: complex):
        self.material.ur = ur

    @property
    def n(self) -> Union[ArrayLike, complex]:
        return self.material.n

    @n.setter
    def n(self, n: complex):
        self.material.n = n

    @property
    def source(self):
        return self.material.source

    @source.setter
    def source(self, source):
        self.material.source = source

    def set_convolution_matrices(self, n_harmonics: Union[ArrayLike, int]):
        if self.crystal is not None:
            print("This is the first option") # NOTE: self.crystal.permittivityCellData
            self.er = self._convolution_matrix(self.crystal.permittivityCellData, n_harmonics)
            self.ur = self._convolution_matrix(self.crystal.permeabilityCellData, n_harmonics)
        else:
            print("NONCRYSTAL LAYER")
            print(f"This is the self.er value: {torch.is_tensor(self.er)}")
            self.er = self.er * complexIdentity(prod(n_harmonics)) # First occurrence of breaking. Replace this function with a gradient-compatible function, or at least change it so that the gradient propagates.
            print(f"This is the self.er value: {torch.is_tensor(self.er)}")
            print(f"This is the complexIdentity value: {torch.is_tensor(complexIdentity(prod(n_harmonics)))}")
            self.ur = self.ur * complexIdentity(prod(n_harmonics))

    def _convolution_matrix(self, cellData: ArrayLike, n_harmonics: Union[ArrayLike, int]) -> ArrayLike:
        print(f"This is the cell data variable: {cellData}")
        dimension = self.crystal.dimensions;

        if isinstance(n_harmonics, int):
            n_harmonics = (n_harmonics,)

        if dimension == 1:
            # n_harmonics = (n_harmonics + (1, 1))
            n_harmonics = (1, n_harmonics[0], 1) # fix for extra
        elif dimension == 2:
            n_harmonics = (n_harmonics + (1,))
        (P, Q, R) = n_harmonics
        convolutionMatrixSize = P*Q*R;
        convolutionMatrixShape = (convolutionMatrixSize, convolutionMatrixSize);
        convolutionMatrix = complexZeros(convolutionMatrixShape)
        print(f"This is the cell data object: {cellData}")
        cellData = reshapeLowDimensionalData(cellData);
        (Nx, Ny, Nz) = cellData.shape;

        zeroHarmonicsLocation = np.array([math.floor(Nx/2), math.floor(Ny/2), math.floor(Nz/2)])
        if USEGRAD:
            zeroHarmonicsLocation = torch.from_numpy(zeroHarmonicsLocation)

        cellFourierRepresentation = fftn(cellData);
        for rrow in range(R):
            for qrow in range(Q):
                for prow in range(P):
                    row = rrow*Q*P + qrow*P + prow;
                    for rcol in range(rrow, R):
                        for qcol in range(qrow, Q):
                            for pcol in range(prow, P):
                                col = rcol*Q*P + qcol*P + pcol;
                                # Get the desired harmonics relative to the 0th-order harmonic.
                                desiredHarmonics = -np.array([prow - pcol, qrow - qcol, rrow - rcol])
                                if USEGRAD:
                                    desiredHarmonics = torch.from_numpy(desiredHarmonics)
                                # Get those harmonic locations from the zero harmonic location.
                                desiredHarmonicsLocation = zeroHarmonicsLocation + desiredHarmonics
                                # update the convolutionmatrix to match this description of the filters
                                convolutionMatrix[row:row+1][col:col+1] = cellFourierRepresentation[desiredHarmonicsLocation[0]:desiredHarmonicsLocation[0]+1,desiredHarmonicsLocation[1]:desiredHarmonicsLocation[1]+1, desiredHarmonicsLocation[2]:desiredHarmonicsLocation[2]+1];
        if convolutionMatrix.shape == (1, 1):
            if USEGRAD:
                convolutionMatrix = convolutionMatrix.squeeze(0).squeeze(0)
            else:
                convolutionMatrix = convolutionMatrix[0][0]
        return convolutionMatrix;

    def __eq__(self, other):
        if not isinstance(other, Layer):
            return NotImplemented

        return self.er == other.er and self.ur == other.ur and self.thickness == other.thickness \
               and self.n == other.n and self.crystal == other.crystal

    def __str__(self):
        return f'Layer with\n\ter: {self.er}\n\tur: {self.ur}\n\tL: {self.thickness}\n\tn: {self.n}\n\tcrystal: {self.crystal}'


freeSpaceLayer = Layer(torch.tensor(1, dtype = torch.cdouble),torch.tensor(1, dtype = torch.cdouble))


class LayerStack:
    """
    Class that defines overall geometry in terms of a stack of layers

    :param internal_layers: Layer objects, starting with the top-most layer (reflection region) and ending with the top-most region (substrate)
    :param incident_layer: Semi-infinite layer of incident region. Defaults to free space
    :param transmission_layer: Semi-infinite layer of transmission region. Defaults to free space
    """
    def __init__(self, *internal_layers: Layer,
                 incident_layer: Layer = Layer(er=torch.tensor(1,dtype=torch.cdouble), ur=torch.tensor(1,dtype=torch.cdouble)), transmission_layer: Layer = Layer(er=torch.tensor(1,dtype=torch.cdouble), ur=torch.tensor(1,dtype=torch.cdouble))):
        self.gapLayer = Layer(er=torch.tensor(1,dtype=torch.cdouble), ur=torch.tensor(1,dtype=torch.cdouble))
        self.incident_layer = incident_layer
        self.incident_layer.incident = True
        self.transmission_layer = transmission_layer
        self.transmission_layer.transmission = True

        self.internal_layers = list(internal_layers)
        self._Kx = None
        self._Ky = None

    def __str__(self):
        top_string = f'\nReflection Layer:\n\t' + str(self.incident_layer) + \
                f'\nTransmissionLayer:\n\t' + str(self.transmission_layer) + \
                f'\nInternal Layer Count: {len(self.internal_layers)}\n'
        internal_string = ''
        for layer in self.internal_layers:
            internal_string += str(layer) + '\n'
        return top_string + internal_string

    @property
    def _k_dimension(self) -> int:
        if isinstance(self.Kx, np.ndarray):
            return self.Kx.shape[0]
        else:
            return 1

    @property
    def _s_element_dimension(self) -> int:
        s_dim = self._k_dimension * 2
        return s_dim

    @property
    def all_layers(self) -> List[Layer]:
        return [self.incident_layer, *self.internal_layers, self.transmission_layer]

    @property
    def Kx(self) -> Union[complex, ArrayLike]:
        return self._Kx

    @Kx.setter
    def Kx(self, kx: Union[complex, ArrayLike]):
        self._Kx = kx
        self.gapLayer.Kx = kx
        for layer in self.all_layers:
            layer.Kx = kx

    @property
    def Ky(self) -> Union[complex, ArrayLike]:
        return self._Ky

    @Ky.setter
    def Ky(self, ky: Union[complex, ArrayLike]):
        self._Ky = ky
        self.gapLayer.Ky = ky
        for layer in self.all_layers:
            layer.Ky = ky

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source
        self.gapLayer.source = source
        for layer in self.all_layers:
            layer.source = self.source

    def set_gap_layer(self):
        self.gapLayer.thickness = 0
        if self._k_dimension == 1:
            self.gapLayer.er = 1 + sq(self.Kx) + sq(self.Ky)
            self.gapLayer.ur = 1 + sq(self.Kx) + sq(self.Ky)
            # self.gapLayer.ur = torch.ones(size = self.Kx.shape)
            self.gapLayer.ur = np.ones(shape = self.Kx.shape)

            Qg = self.gapLayer.Q_matrix()

            lambda_gap = self.gapLayer.lambda_matrix()
            print("THIS IS THE FIRST OPTION")
            print("This is the lambda matrix:")
            print(lambda_gap)

        else:
            Kz = self.gapLayer.Kz_gap()
            Qg = self.gapLayer.Q_matrix()
            lambda_gap = complexIdentity(self._k_dimension * 2)
            lambda_gap[:self._k_dimension, :self._k_dimension] = 1j * Kz
            lambda_gap[self._k_dimension:, self._k_dimension:] = 1j * Kz
            print("THIS IS THE SECOND OPTION")

        self.Wg = complexIdentity(self._s_element_dimension)
        print("-"*100)
        print(lambda_gap.dtype)
        print(Qg.dtype)
        print("-"*100)
        self.Vg = Qg @ inv(lambda_gap)

        for layer in self.all_layers:
            layer.Wg = self.Wg
            layer.Vg = self.Vg

    # set all convolution matrices for all interior layers
    def set_convolution_matrices(self, n_harmonics: Union[int, ArrayLike]):
        print("Is this being called?")
        for layer in self.internal_layers:
            print(f"Checking per layer: {torch.is_tensor(layer.er)}")
            layer.set_convolution_matrices(n_harmonics)

    @property
    def crystal(self) -> Union[None, Crystal]:
        for i in range(len(self.internal_layers)):
            if self.internal_layers[i].crystal is not None:
                return self.internal_layers[i].crystal
        return None

    def plot(self, fig: Union[None, Figure] = None, ax: Union[None, Axes] = None) -> Tuple[Figure, Axes]:
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is not None and ax is None:
            ax = fig.add_subplot()

        # z = 0 will be defined at the start of the top-most layer.

        return fig, ax



emptyStack = LayerStack()
