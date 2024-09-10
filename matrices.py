from rcwa.shorthand import *
from autograd import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import torch

USEGRAD = 1

if USEGRAD:
    def s_incident(source, n_harmonics: Union[int, ArrayLike]):
        totalNumberHarmonics = np.prod(n_harmonics)
        return torch.hstack((source.pX * kroneckerDeltaVector(totalNumberHarmonics),
                source.pY * kroneckerDeltaVector(totalNumberHarmonics)))
else:
    def s_incident(source, n_harmonics: Union[int, ArrayLike]):
        totalNumberHarmonics = np.prod(n_harmonics)
        return np.hstack((source.pX * kroneckerDeltaVector(totalNumberHarmonics),
                source.pY * kroneckerDeltaVector(totalNumberHarmonics)))

def S_matrix_transparent(matrixShape: ArrayLike):
    STransparent = complexZeros((2, 2) + matrixShape);
    STransparent[0,1] = complexIdentity(matrixShape[0]);
    STransparent[1,0] = complexIdentity(matrixShape[0]);
    return STransparent;

def redheffer_product(SA: ArrayLike, SB: ArrayLike):
    D = D_matrix_redheffer(SA, SB)
    F = F_matrix(SA, SB)

    S11 = SA[0, 0] + D @ SB[0, 0] @ SA[1, 0];
    S12 = D @ SB[0, 1];
    S21 = F @ SA[1, 0];
    S22 = SB[1, 1] + F @ SA[1, 1] @ SB[0, 1];
    if USEGRAD:
        print("*"*100)
        print("First bit")
        print(SA.shape)
        print(SB.shape) # This is the problem
        print("*"*100)
        print(S11.shape)
        print(S12.shape)
        print(S21.shape)
        print(S22.shape)
        S = torch.stack([torch.stack([S11, S12]), torch.stack([S21, S22])])
    else:
    # x=[[S11, S12], [S21, S22]]
    # print("*"*100)
    # print(x)
    # print("*"*100)
    # S11=S11.detach().numpy()
    # S12=S12.detach().numpy()
    # S21=S21.detach().numpy()
    # S22=S22.detach().numpy()

        S = np.array([[S11, S12], [S21, S22]])
    return S

def omega_squared_matrix(P: ArrayLike, Q: ArrayLike):
    return P @ Q

def A_matrix(Wi, Wj, Vi, Vj):
    print("$"*100)
    print(Wi.shape)
    print(Vi.shape)
    print(Wj.shape)
    print(Vi.shape)
    print("$"*100)
    if USEGRAD:
        x= torch.linalg.inv(Wi) @ Wj + inv(Vi) @ Vj;
        print(x.shape)
        return x
    else:
        return np.linalg.inv(Wi) @ Wj + inv(Vi) @ Vj;

def B_matrix(Wi, Wj, Vi, Vj):
    print("$"*100)
    print(Wi.shape)
    print(Vi.shape)
    print(Wj.shape)
    print(Vi.shape)
    print("$"*100)
    if USEGRAD:
        x= torch.linalg.inv(Wi) @ Wj - inv(Vi) @ Vj;
        print(x.shape)
        return x
    else:
        return np.linalg.inv(Wi) @ Wj - inv(Vi) @ Vj;

def D_matrix(Ai, Bi, Xi):
    print("$"*100)
    print(Ai.shape)
    print(Bi.shape)
    print(Xi.shape)
    print("$"*100)
    # print(f"This is Ai before we invert it:\n{Ai}")
    if USEGRAD:
        AiInverse = torch.linalg.inv(Ai)
    else:
        AiInverse = np.linalg.inv(Ai);
    # print(f"This is Ai after we invert it:\n{AiInverse}")
    # print(Ai)
    # print(Xi)
    # print(Bi)
    # print(AiInverse)
    x=Ai - Xi @ Bi @ AiInverse @ Xi @ Bi;
    print(x.shape)
    return x

def D_matrix_redheffer(SA, SB):
    if USEGRAD:
        return SA[0,1] @ torch.linalg.inv(complexIdentity(SA[0,0].shape[0]) - SB[0,0] @ SA[1,1])
    else:
        return SA[0,1] @ np.linalg.inv(complexIdentity(SA[0,0].shape[0]) - SB[0,0] @ SA[1,1])

def F_matrix(SA, SB):
    if USEGRAD:
        return SB[1,0] @ torch.linalg.inv(complexIdentity(SA[0,0].shape[0]) - SA[1,1] @ SB[0,0])
    else:
        return SB[1,0] @ np.linalg.inv(complexIdentity(SA[0,0].shape[0]) - SA[1,1] @ SB[0,0])


def calculateInternalSMatrixFromRaw(Ai, Bi, Xi, Di):
    if USEGRAD:
        AiInverse = torch.linalg.inv(Ai)
        DiInverse = torch.linalg.inv(Di)
    else:
        AiInverse = np.linalg.inv(Ai)
        DiInverse = np.linalg.inv(Di)

    print("*"*100)
    print("Second bit")
    print(Ai.shape)
    print(Di.shape)
    print("*"*100)

    S11 = DiInverse @ (Xi @ Bi @ AiInverse @ Xi @ Ai - Bi)
    S12 = DiInverse @ Xi @ (Ai - Bi @ AiInverse @ Bi)
    S21 = S12
    S22 = S11

    print("Second occurrence")
    print(S11.shape)
    print(S12.shape)
    print(S21.shape)
    print(S22.shape)
    print(torch.stack([S11, S12]).shape)
    print(torch.concatenate([S11, S12]).shape)
    if USEGRAD:
        S = torch.stack([torch.stack([S11, S12]), torch.stack([S21, S22])])
        # S = torch.stack([[S11, S12], [S21, S22]], dtype = torch.cdouble)
    else:
        S = np.array([[S11, S12],[S21, S22]])
    print(f"This is the resulting shape HERE: {S.shape}")
    return S

def calculateReflectionRegionSMatrixFromRaw(AReflectionRegion, BReflectionRegion):
    A = AReflectionRegion
    B = BReflectionRegion
    AInverse = np.linalg.inv(A)

    S11 = - AInverse @ B
    S12 = 2 * AInverse
    S21 = 0.5 * (A - B @ AInverse @ B)
    S22 = B @ AInverse
    S = np.array([[S11,S12], [S21,S22]])
    return S

def calculateTransmissionRegionSMatrixFromRaw(ATransmissionRegion, BTransmissionRegion): # UNIT TESTS COMPLETE
    A = ATransmissionRegion
    B = BTransmissionRegion
    AInverse = np.linalg.inv(A)

    S11 = B@ AInverse
    S12 = 0.5* (A- (B @ AInverse @ B))
    S21 = 2* AInverse
    S22 = - AInverse @ B
    S = np.array([[S11,S12],[S21,S22]])
    return S


# NOTE - this can only be used for 1D (TMM-type) simulations. rTE/rTM are not meaningful quantities otherwise.
def calculateTEMReflectionCoefficientsFromXYZ(source, rx, ry, rz):
    if isinstance(rx, np.ndarray):
        raise NotImplementedError
    else:
        rxyz = np.array([rx, ry, rz])
        rTEM = source.ATEM @ rxyz
        rTEM[0] = rTEM[0] / source.pTE
        rTEM[1] = rTEM[1] / source.pTM
        return rTEM

if USEGRAD:
    def calculateReflectionCoefficient(S, Kx, Ky, KzReflectionRegion,
            WReflectionRegion, source, numberHarmonics):
        print(f"195: {source}, {numberHarmonics}")
        incidentFieldHarmonics = s_incident(source, numberHarmonics)
        rTransverse = WReflectionRegion @ S[0,0] @ torch.linalg.inv(WReflectionRegion) @ incidentFieldHarmonics

        rx, ry, rz = None, None, None
        if torch.is_tensor(Kx):
            maxIndex = int(rTransverse.shape[0]/2)
            rx = rTransverse[0:maxIndex]
            ry = rTransverse[maxIndex:]
            rz = - torch.linalg.inv(KzReflectionRegion) @ (Kx @ rx + Ky @ ry)
        else:
            rx = rTransverse[0:1]
            ry = rTransverse[1:2]
            rz = - (Kx * rx + Ky * ry) / KzReflectionRegion
        return rx, ry, rz
else:
    def calculateReflectionCoefficient(S, Kx, Ky, KzReflectionRegion,
            WReflectionRegion, source, numberHarmonics):

        incidentFieldHarmonics = s_incident(source, numberHarmonics)
        rTransverse = WReflectionRegion @ S[0,0] @ np.linalg.inv(WReflectionRegion) @ incidentFieldHarmonics

        rx, ry, rz = None, None, None
        if isinstance(Kx, np.ndarray):
            maxIndex = int(rTransverse.shape[0]/2)
            rx = rTransverse[0:maxIndex]
            ry = rTransverse[maxIndex:]
            rz = - np.linalg.inv(KzReflectionRegion) @ (Kx @ rx + Ky @ ry)
        else:
            rx = rTransverse[0]
            ry = rTransverse[1]
            rz = - (Kx * rx + Ky * ry) / KzReflectionRegion
        return rx, ry, rz
if USEGRAD:
    def calculateTransmissionCoefficient(S, Kx, Ky, KzTransmissionRegion,
            WTransmissionRegion, source, numberHarmonics):
        incidentFieldHarmonics = s_incident(source, numberHarmonics)
        tTransverse = WTransmissionRegion @ S[1,0] @ inv(WTransmissionRegion) @ incidentFieldHarmonics

        tx, ty, tz = None, None, None
        if torch.is_tensor(Kx):
            maxIndex = int(tTransverse.shape[0]/2)
            tx = tTransverse[:maxIndex]
            ty = tTransverse[maxIndex:]
            tz = - inv(KzTransmissionRegion) @ (Kx @ tx + Ky @ ty)
        else:
            tx = tTransverse[0:1]
            ty = tTransverse[1:2]
            tz = - (Kx * tx + Ky * ty) / KzTransmissionRegion
        return tx, ty, tz
else:
    def calculateTransmissionCoefficient(S, Kx, Ky, KzTransmissionRegion,
            WTransmissionRegion, source, numberHarmonics):
        incidentFieldHarmonics = s_incident(source, numberHarmonics)
        tTransverse = WTransmissionRegion @ S[1,0] @ inv(WTransmissionRegion) @ incidentFieldHarmonics

        tx, ty, tz = None, None, None
        if isinstance(Kx, np.ndarray):
            maxIndex = int(tTransverse.shape[0]/2)
            tx = tTransverse[:maxIndex]
            ty = tTransverse[maxIndex:]
            tz = - inv(KzTransmissionRegion) @ (Kx @ tx + Ky @ ty)
        else:
            tx = tTransverse[0]
            ty = tTransverse[1]
            tz = - (Kx * tx + Ky * ty) / KzTransmissionRegion
        return tx, ty, tz

def calculateDiffractionReflectionEfficiency(rx, ry, rz, source, KzReflectionRegion, layerStack, numberHarmonics):
    print(f"These are the numbers of harmonics: {numberHarmonics}") # this should be a scalar--we can use np.isscalar for this
    urReflectionRegion = layerStack.incident_layer.ur
    preMatrix = real(-1 /urReflectionRegion * KzReflectionRegion) / \
            real(source.k_incident[2] / urReflectionRegion)
    R = None
    if isinstance(KzReflectionRegion, np.ndarray):
        R = preMatrix @ (sq(np.abs(rx)) + sq(np.abs(ry)) + sq(np.abs(rz)))
        RDimension = int(sqrt(rx.shape[0]))
        if not np.isscalar(numberHarmonics):
            R = R.reshape((RDimension, RDimension))
    else:
        R = -preMatrix * (sq(np.abs(rx)) + sq(np.abs(ry)) + sq(np.abs(rz)))
    return R

def calculateDiffractionTransmissionEfficiency(tx, ty, tz, source, KzTransmissionRegion, layerStack,
                                              numberHarmonics):
    urTransmissionRegion = layerStack.transmission_layer.ur
    urReflectionRegion = layerStack.incident_layer.ur
    preMatrix = real(1 / urTransmissionRegion * KzTransmissionRegion) / \
            real(source.k_incident[2] / urReflectionRegion)

    if isinstance(KzTransmissionRegion, np.ndarray):
        T = preMatrix @ (sq(np.abs(tx)) + sq(np.abs(ty)) + sq(np.abs(tz)))
        TDimension = int(sqrt(tx.shape[0]))
        if not np.isscalar(numberHarmonics):
            T = T.reshape((TDimension, TDimension))
    else:
        T = preMatrix * (sq(np.abs(tx)) + sq(np.abs(ty)) + sq(np.abs(tz)))
    return T

def calculateEz(kx, ky, kz, Ex, Ey):
    Ez = - (kx*Ex + ky*Ey) / kz
    return Ez;

def calculateRT(kzReflectionRegion, kzTransmissionRegion,
        layerStack, ExyzReflected, ExyzTransmitted):
    urTransmissionRegion = layerStack.transmission_layer.ur
    urReflectionRegion = layerStack.incident_layer.ur
    R = sq(norm(ExyzReflected))
    T = sq(norm(ExyzTransmitted))*np.real(kzTransmissionRegion / urTransmissionRegion) / \
            (kzReflectionRegion / urReflectionRegion);

    return (R, T);


class MatrixCalculator:
    """
    Superclass of Layer which is used purely for the calculation of matrices
    """

    def P_matrix(self):
        if isinstance(self.Kx, np.ndarray):
            return self._P_matrix_general()
        else:
            return self._P_matrix_homogenous()

    def _P_matrix_homogenous(self):
        P = complexZeros((2, 2));

        P[0,0] = self.Kx*self.Ky;
        P[0,1] = self.er*self.ur - np.square(self.Kx);
        P[1,0] = sq(self.Ky) - self.er*self.ur
        P[1,1] = - self.Kx*self.Ky;
        P /= self.er;
        return P

    def _P_matrix_general(self):
        # print("Before: ",self.Kx.shape,self.er.shape,self.Ky.shape)
        if not self.er.shape:
            # print(f"This is the permittivity: {self.er}")
            # erInverse = np.ones((4,4), dtype = np.complex_) * 1/self.er
            erInverse = np.ones((self.Kx.shape[0], self.Ky.shape[0]), dtype = np.complex_) / self.er
        else:
            erInverse = np.squeeze(np.linalg.inv([[self.er]])) # fix this is the fix for single
        # erInverse = np.linalg.inv(self.er)
        KMatrixDimension = self.Kx.shape[0]
        matrixShape = (2 *KMatrixDimension, 2 * KMatrixDimension)
        P = complexZeros(matrixShape)

        P[:KMatrixDimension,:KMatrixDimension] = self.Kx @ erInverse @ self.Ky
        P[:KMatrixDimension,KMatrixDimension:] = self.ur - self.Kx @ erInverse @ self.Kx
        P[KMatrixDimension:,:KMatrixDimension] = self.Ky @ erInverse @ self.Ky - self.ur
        P[KMatrixDimension:,KMatrixDimension:] = - self.Ky @ erInverse @ self.Kx
        # print(f"This is P\n{P}")
        return P

    def Q_matrix(self):
        if isinstance(self.Kx, np.ndarray):
            if isinstance(self.er, np.ndarray):
                return self._Q_matrix_general()
            else:
                return self._Q_matrix_semi_infinite()
        else:
            return self._Q_matrix_homogenous()

    def _Q_matrix_homogenous(self):
        Q = complexZeros((2,2));
        Q[0,0] = self.Kx * self.Ky;
        # print(self.er,self.ur,self.Kx,sq(self.Kx))
        Q[0,1] = self.er*self.ur - sq(self.Kx);
        Q[1,0] = sq(self.Ky) - self.er*self.ur;
        Q[1,1] = - self.Kx * self.Ky;
        Q = Q / self.ur;
        return Q;
# if USEGRAD:
#     def _Q_matrix_general(self):
#         print(f"_Q_matrix_general-->ur.shape: {self.ur.shape}")
#         urInverse = torch.linalg.inv(self.ur) # ur 0-dim
#         KMatrixDimension = self.Kx.shape[0]
#         matrixShape = (2 *KMatrixDimension, 2 * KMatrixDimension)
#         Q = complexZeros(matrixShape)
#         # urInverse = np.zeros(shape=(1,1)) + 1
#         Q[:KMatrixDimension,:KMatrixDimension] = self.Kx @ urInverse @ self.Ky
#         Q[:KMatrixDimension,KMatrixDimension:] = self.er - self.Kx @ urInverse @ self.Kx
#         Q[KMatrixDimension:,:KMatrixDimension] = self.Ky @ urInverse @ self.Ky - self.er
#         Q[KMatrixDimension:,KMatrixDimension:] = - self.Ky @ urInverse @ self.Kx
#         return Q
# else:
    def _Q_matrix_general(self):
        # print(f"_Q_matrix_general-->ur.shape: {self.ur.shape}")
        urInverse = np.linalg.inv(self.ur) # ur 0-dim
        KMatrixDimension = self.Kx.shape[0]
        matrixShape = (2 *KMatrixDimension, 2 * KMatrixDimension)
        Q = complexZeros(matrixShape)
        # urInverse = np.zeros(shape=(1,1)) + 1
        Q[:KMatrixDimension,:KMatrixDimension] = self.Kx @ urInverse @ self.Ky
        Q[:KMatrixDimension,KMatrixDimension:] = self.er - self.Kx @ urInverse @ self.Kx
        Q[KMatrixDimension:,:KMatrixDimension] = self.Ky @ urInverse @ self.Ky - self.er
        Q[KMatrixDimension:,KMatrixDimension:] = - self.Ky @ urInverse @ self.Kx
        return Q

    def _Q_matrix_semi_infinite(self):
        KDimension = self.Kx.shape[0]
        Q = complexZeros((KDimension * 2, KDimension*2))
        Q[:KDimension, :KDimension] = self.Kx @ self.Ky
        Q[:KDimension, KDimension:] = self.ur * self.er * complexIdentity(KDimension) - self.Kx @ self.Kx
        Q[KDimension:, :KDimension] = self.Ky @ self.Ky - self.ur*self.er*complexIdentity(KDimension)
        Q[KDimension:, KDimension:] = - self.Ky @ self.Kx
        Q /= self.ur
        return Q

    if USEGRAD:
        def lambda_matrix(self):
            Kz = self.Kz_forward()
            print(f"This is Kz: {Kz}")
            if torch.is_tensor(Kz):
                KzDimension = Kz.shape[0]
                LambdaShape = (KzDimension*2, KzDimension*2)
                Lambda = complexZeros(LambdaShape)
                Lambda[:KzDimension, :KzDimension] = 1j*Kz
                Lambda[KzDimension:, KzDimension:] = 1j*Kz
                print("THIS FIRST THING IS BEING CALLED")
                print("-"*100)
                print(Kz*1j)
                print(KzDimension)
                print(LambdaShape)
                print(complexZeros(LambdaShape))
                print("-"*100)
                return Lambda
            else:
                return complexIdentity(2)* (0 + 1j)*Kz;
    else:
        def lambda_matrix(self):
            Kz = self.Kz_forward() # I am a little unsure about this particular line. Why is Kz_backward never used?
            print(f"This is Kz: {Kz}")
            if isinstance(Kz, np.ndarray):
                KzDimension = Kz.shape[0]
                LambdaShape = (KzDimension*2, KzDimension*2)
                Lambda = complexZeros(LambdaShape)
                Lambda[:KzDimension, :KzDimension] = 1j*Kz
                Lambda[KzDimension:, KzDimension:] = 1j*Kz
                print("-"*100)
                print(Kz*1j)
                print(KzDimension)
                print(LambdaShape)
                print(complexZeros(LambdaShape))
                print("-"*100)
                return Lambda
            else:
                return complexIdentity(2)* (0 + 1j)*Kz;
    if USEGRAD:
        def Kz_backward(self):
            if torch.is_tensor(self.Kx):
                print(torch.is_tensor(self.er))
                print(torch.is_tensor(self.ur))
                return -conj(sqrt(conj(self.er*self.ur)*complexIdentity(self.Kx.shape[0]) - self.Kx @ self.Kx - self.Ky @ self.Ky))
            else:
                return sqrt(self.er*self.ur - sq(self.Kx) - sq(self.Ky))
    else:
        def Kz_backward(self):
            if isinstance(self.Kx, np.ndarray):
                return -conj(sqrt(conj(self.er*self.ur)*complexIdentity(self.Kx.shape[0]) - self.Kx @ self.Kx - self.Ky @ self.Ky))
            else:
                return sqrt(self.er*self.ur - sq(self.Kx) - sq(self.Ky))

    if USEGRAD:
        def Kz_forward(self):
            if torch.is_tensor(self.Kx):
                x= conj(sqrt(conj(self.er*self.ur)*complexIdentity(self.Kx.shape[0]) - self.Kx @ self.Kx - self.Ky @ self.Ky))
                return torch.tensor(x, dtype = torch.cdouble)
            else:
                return torch.tensor(sqrt(self.er*self.ur - sq(self.Kx) - sq(self.Ky)), dtype = torch.cdouble)
    else:
        def Kz_forward(self):
            if isinstance(self.Kx, np.ndarray):
                # print(type(self.er),type(self.ur))
                x= conj(sqrt(conj(self.er*self.ur)*complexIdentity(self.Kx.shape[0]) - self.Kx @ self.Kx - self.Ky @ self.Ky))
                return x
            else:
                return sqrt(self.er*self.ur - sq(self.Kx) - sq(self.Ky))


    def Kz_gap(self):
        if isinstance(self.Kx, np.ndarray):
            return conj(sqrt(complexIdentity(self.Kx.shape[0]) - self.Kx @ self.Kx - self.Ky @ self.Ky))
        else:
            return sqrt(self.er*self.ur - sq(self.Kx) - sq(self.Ky))

    def VWLX_matrices(self):
        if not isinstance(self.Kx, np.ndarray):
            return self._VWLX_matrices_homogenous()
        else:
            return self._VWLX_matrices_general()

    def _VWLX_matrices_homogenous(self):    # TODO: flip
        Kz = self.Kz_forward()
        Q = self.Q_matrix()
        O = self.lambda_matrix()
        if USEGRAD:
            OInverse= torch.linalg.inv(O)
        else:
            OInverse = np.linalg.inv(O)
        W = complexIdentity(2)
        X = matrixExponentiate(O * self.source.k0 * self.thickness) # added the negative to match the homogenous calculation
        print(f"Within _VWLX_matrices_homogenous")
        print(Q.dtype)
        print(W.dtype)
        V = Q @ W @ OInverse
        return (V, W, O, X)

    def _VWLX_matrices_general(self):   # TODO: flip
        P = self.P_matrix()
        Q = self.Q_matrix()
        OmegaSquared = omega_squared_matrix(P, Q)

        if self.homogenous:
            Kz = self.Kz_forward()
            Lambda = self.lambda_matrix()
            if USEGRAD:
                LambdaInverse = torch.linalg.inv(Lambda)
            else:
                LambdaInverse = np.linalg.inv(Lambda)
            W = complexIdentity(2 * Kz.shape[0])
            V = Q @ W @ LambdaInverse
            X = matrixExponentiate(Lambda * self.source.k0 * self.thickness)
            return (V, W, Lambda, X)
        else:
            eigenValues, W = eig(OmegaSquared)
            if USEGRAD:
                Lambda = torch.diag(sqrt(eigenValues))
                LambdaInverse = torch.diag(torch.reciprocal(sqrt(eigenValues)))
            else:
                Lambda = np.diag(sqrt(eigenValues))
                LambdaInverse = np.diag(np.reciprocal(sqrt(eigenValues)))
            V = Q @ W @ LambdaInverse
            X = matrixExponentiate(Lambda * self.source.k0 * self.thickness)
            return (V, W, Lambda, X)

    def S_matrix(self):
        print("WE ARE CREATING THE S_MATRIX")
        if self.thickness > 0:
            print("First option")
            return self._S_matrix_internal()
        elif self.thickness == 0:
            if self.incident:
                print("Incident angle")
                return self._S_matrix_reflection()
            elif self.transmission:
                print("Transmission angle")
                return self._S_matrix_transmission()
            else:
                raise ValueError('''Semi-infinite film appears to be neither incident or transmissive.
                Cannot compute S-matrix''')

    def _S_matrix_internal(self): #NOTE: this is not the issue. It must be in A_matrix, B_matrix, or D_matrix
        (Vi, Wi, _, Xi) = self.VWLX_matrices()
        # print(f"This is the max of Vi:{np.max(Vi)}")
        # print(f"This is the max of Xi:{np.max(Xi)}")
        print(f"Xi: {Xi}")
        print(f"Wi: {Wi}")
        print(f"Vi: {Vi}")
        Ai = A_matrix(Wi, self.Wg, Vi, self.Vg)
        # print(f"This is the max of Ai:{np.max(Ai)}")
        Bi = B_matrix(Wi, self.Wg, Vi, self.Vg)
        # print(f"This is the max of Bi:{np.max(Bi)}")
        Di = D_matrix(Ai, Bi, Xi)
        # print(f"This is the max of Di:{np.max(Di)}")
        print("%"*100)
        print("These are the shapes:")
        print(Ai.shape)
        print(Bi.shape)
        print(Di.shape)
        print("%"*100)
        Si = calculateInternalSMatrixFromRaw(Ai, Bi, Xi, Di);
        print(f"This is the Si shape: {Si.shape}")
        # print(f"This is the max of Si:{np.max(Si)}")
        return Si;

    def _S_matrix_reflection(self): # TODO: flip
        if isinstance(self.Kx, np.ndarray):
            return self._S_matrix_reflection_general()
        else:
            return self._S_matrix_reflection_homogenous()

    def _S_matrix_reflection_homogenous(self):
        (Vi, Wi, _, X) = self.VWLX_matrices()
        Ai = A_matrix(self.Wg, Wi, self.Vg, Vi)
        Bi = B_matrix(self.Wg, Wi, self.Vg, Vi)

        Si = calculateReflectionRegionSMatrixFromRaw(Ai, Bi)
        return Si

    def _S_matrix_reflection_general(self): # TODO: flip
        KDimension = self.Kx.shape[0]
        lambdaRef = complexZeros((KDimension*2, KDimension*2))
        Wi = complexIdentity(KDimension * 2)
        Q = self.Q_matrix()
        # I have no idea why we conjugate ur * er and then conjugate the whole thing.
        Kz = conj(sqrt (conj(self.er * self.ur) * \
                        complexIdentity(KDimension) - self.Kx @ self.Kx - self.Ky @ self.Ky))
        lambdaRef[:KDimension, :KDimension] = 1j*Kz
        lambdaRef[KDimension:, KDimension:] = 1j*Kz
        Vi = Q @ np.linalg.inv(lambdaRef)
        Ai = A_matrix(self.Wg, Wi, self.Vg, Vi)
        Bi = B_matrix(self.Wg, Wi, self.Vg, Vi)

        Sref = calculateReflectionRegionSMatrixFromRaw(Ai, Bi)
        return Sref

    def _S_matrix_transmission(self):   # TODO: flip
        if isinstance(self.Kx, np.ndarray):
            return self._S_matrix_transmission_general()
        else:
            return self._S_matrix_transmission_homogenous()

    def _S_matrix_transmission_homogenous(self):
        (Vi, Wi, _, X) = self.VWLX_matrices()
        Ai = A_matrix(self.Wg, Wi, self.Vg, Vi);
        Bi = B_matrix(self.Wg, Wi, self.Vg, Vi);

        Si = calculateTransmissionRegionSMatrixFromRaw(Ai, Bi);
        return Si;

    def _S_matrix_transmission_general(self):   # TODO: flip
        KDimension = self.Kx.shape[0]
        lambdaRef = complexZeros((KDimension*2, KDimension*2))
        Wi = complexIdentity(KDimension * 2)
        Q = self.Q_matrix()

        # I have no idea why we conjugate ur * er and then conjugate the whole thing.
        Kz = conj(sqrt (conj(self.er * self.ur) * complexIdentity(KDimension) - self.Kx @ self.Kx - self.Ky @ self.Ky))
        lambdaRef[:KDimension, :KDimension] = 1j*Kz
        lambdaRef[KDimension:,KDimension:] = 1j*Kz
        Vi = Q @ np.linalg.inv(lambdaRef)
        Ai = A_matrix(self.Wg, Wi, self.Vg, Vi)
        Bi = B_matrix(self.Wg, Wi, self.Vg, Vi)

        Strn = calculateTransmissionRegionSMatrixFromRaw(Ai, Bi)
        return Strn
