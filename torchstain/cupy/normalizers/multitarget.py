import cupy as cp

class CupyMultiMacenkoNormalizer:
    def __init__(self, norm_mode="avg-post"):
        self.norm_mode = norm_mode
        self.HERef = cp.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = cp.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = cp.transpose(I, (1, 2, 0))
        OD = -cp.log((I.reshape(-1, I.shape[-1]).astype(float) + 1) / Io)
        ODhat = OD[~cp.any(OD < beta, axis=1)]
        return OD, ODhat

    def __find_phi_bounds(self, ODhat, eigvecs, alpha):
        That = cp.dot(ODhat, eigvecs)
        phi = cp.arctan2(That[:, 1], That[:, 0])

        minPhi = cp.percentile(phi, alpha)
        maxPhi = cp.percentile(phi, 100 - alpha)

        return minPhi, maxPhi

    def __find_HE_from_bounds(self, eigvecs, minPhi, maxPhi):
        vMin = cp.dot(eigvecs, [cp.cos(minPhi), cp.sin(minPhi)]).reshape(-1, 1)
        vMax = cp.dot(eigvecs, [cp.cos(maxPhi), cp.sin(maxPhi)]).reshape(-1, 1)

        HE = cp.concatenate([vMin, vMax], axis=1) if vMin[0] > vMax[0] else cp.concatenate([vMax, vMin], axis=1)
        return HE

    def __find_HE(self, ODhat, eigvecs, alpha):
        minPhi, maxPhi = self.__find_phi_bounds(ODhat, eigvecs, alpha)
        return self.__find_HE_from_bounds(eigvecs, minPhi, maxPhi)

    def __find_concentration(self, OD, HE):
        Y = OD.T
        C, _, _, _ = cp.linalg.lstsq(HE, Y, rcond=None)
        return C

    def __compute_matrices_single(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io, beta)

        cov_matrix = cp.cov(ODhat.T)
        eigvals, eigvecs = cp.linalg.eigh(cov_matrix)
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)
        C = self.__find_concentration(OD, HE)
        maxC = cp.array([cp.percentile(C[0, :], 99), cp.percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, Is, Io=240, alpha=1, beta=0.15):
        if self.norm_mode == "avg-post":
            HEs, _, maxCs = zip(*[self.__compute_matrices_single(I, Io, alpha, beta) for I in Is])

            self.HERef = cp.mean(HEs, axis=0)
            self.maxCRef = cp.mean(maxCs, axis=0)
        elif self.norm_mode == "concat":
            ODs, ODhats = zip(*[self.__convert_rgb2od(I, Io, beta) for I in Is])
            OD = cp.vstack(ODs)
            ODhat = cp.vstack(ODhats)

            cov_matrix = cp.cov(ODhat.T)
            eigvals, eigvecs = cp.linalg.eigh(cov_matrix)
            eigvecs = eigvecs[:, [1, 2]]

            HE = self.__find_HE(ODhat, eigvecs, alpha)
            C = self.__find_concentration(OD, HE)
            maxCs = cp.array([cp.percentile(C[0, :], 99), cp.percentile(C[1, :], 99)])

            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == "avg-pre":
            ODs, ODhats = zip(*[self.__convert_rgb2od(I, Io, beta) for I in Is])

            covs = [cp.cov(ODhat.T) for ODhat in ODhats]
            eigvecs = cp.mean([cp.linalg.eigh(cov)[1][:, [1, 2]] for cov in covs], axis=0)

            OD = cp.vstack(ODs)
            ODhat = cp.vstack(ODhats)

            HE = self.__find_HE(ODhat, eigvecs, alpha)
            C = self.__find_concentration(OD, HE)
            maxCs = cp.array([cp.percentile(C[0, :], 99), cp.percentile(C[1, :], 99)])

            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode in ["fixed-single", "stochastic-single"]:
            self.HERef, _, self.maxCRef = self.__compute_matrices_single(Is[0], Io, alpha, beta)
        else:
            raise ValueError("Unknown norm mode")

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices_single(I, Io, alpha, beta)
        C = (self.maxCRef / maxC).reshape(-1, 1) * C

        Inorm = Io * cp.exp(-cp.dot(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = cp.transpose(Inorm, (1, 0)).reshape(h, w, c).astype(cp.int32)

        H, E = None, None

        if stains:
            H = Io * cp.exp(-cp.dot(self.HERef[:, 0].reshape(-1, 1), C[0, :].reshape(1, -1)))
            H[H > 255] = 255
            H = cp.transpose(H, (1, 0)).reshape(h, w, c).astype(cp.int32)

            E = Io * cp.exp(-cp.dot(self.HERef[:, 1].reshape(-1, 1), C[1, :].reshape(1, -1)))
            E[E > 255] = 255
            E = cp.transpose(E, (1, 0)).reshape(h, w, c).astype(cp.int32)

        return Inorm, H, E
