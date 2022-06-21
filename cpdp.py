from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from scipy.spatial.distance import pdist, squareform


class CPDP:
    """
    This Class inspired from Ke Li's research, source code: https://github.com/COLA-Laboratory/icse2020
    """

    class Bruakfilter(object):
        def __init__(self, n_neighbors=10):
            self.n_neighbors = n_neighbors

        def run(self, Xsource: np.array, Ysource: np.array, Xtarget: np.array, Ytarget: np.array):
            Xsource: np.array = np.log(Xsource + 1)
            Xtarget: np.array = np.log(Xtarget + 1)

            if self.n_neighbors > Xsource.shape[0]:
                return 0, 0, 0, 0

            knn = NearestNeighbors()
            knn.fit(Xsource)
            data = []
            ysel = []

            for item in Xtarget:
                tmp = knn.kneighbors(item.reshape(1, -1), self.n_neighbors, return_distance=False)
                tmp = tmp[0]
                for i in tmp:
                    if list(Xsource[i]) not in data:
                        data.append(list(Xsource[i]))
                        ysel.append(Ysource[i])
            Xsource: np.array = np.asanyarray(data)
            Ysource: np.array = np.asanyarray(ysel)

            return Xsource, Ysource, Xtarget, Ytarget

    class DSBF(object):
        def __init__(self, topK=1, neighbors=10):
            self.topK = int(topK)
            self.neighbors = neighbors

        def featureReduction(self, source, target):
            d = pdist(target.T, metric='euclidean')
            D = squareform(d)
            dist = D.copy()
            D = np.zeros(D.shape)

            for i in range(target.shape[1]):
                index = np.argsort(dist[i])
                count = 0
                for j in range(len(index)):
                    if count < self.topK and index[j] != i:
                        D[i][index[j]] = 1
                        count += 1

            V = np.sum(D, axis=0)
            V[V < 1e-6] = 0
            index = np.where(V != 0)
            target = np.delete(target, index, axis=1)
            source = np.delete(source, index, axis=1)

            return source, target

        def outlierRemove(self, target, ys):
            d = pdist(target, metric='euclidean')
            D = squareform(d)
            dist = D.copy()
            D = np.zeros(D.shape)
            for i in range(target.shape[0]):
                index = np.argsort(dist[i])
                count = 0
                for j in range(len(index)):
                    if count < self.topK and index[j] != i:
                        D[i][index[j]] = 1
                        count += 1
            V = np.sum(D, axis=0)
            V[V < 1e-6] = 0
            index = np.where(V == 0)
            target = np.delete(target, index, axis=0)
            ys = np.delete(ys, index, axis=0)
            return target, ys

        def Bruakfilter(self, Xsource, Ysource, Xtarget, Ytarget):
            Xsource = np.log(Xsource + 1)
            Xtarget = np.log(Xtarget + 1)

            if self.neighbors > Xsource.shape[0]:
                return 0, 0, 0, 0

            knn = NearestNeighbors()
            knn.fit(Xsource)
            data = []
            ysel = []

            for item in Xtarget:
                tmp = knn.kneighbors(item.reshape(1, -1), self.neighbors, return_distance=False)
                tmp = tmp[0]
                for i in tmp:
                    if list(Xsource[i]) not in data:
                        data.append(list(Xsource[i]))
                        ysel.append(Ysource[i])
            Xsource = np.asanyarray(data)
            Ysource = np.asanyarray(ysel)

            return Xsource, Ysource, Xtarget, Ytarget

        def run(self, Xsource, Ysource, Xtarget, Ytarget):
            Xsource, Xtarget = self.featureReduction(Xsource, Xtarget)
            if Xsource.shape[1] == 0:
                return 0, 0, 0, 0
            Xsource, Ysource = self.outlierRemove(Xsource, Ysource)
            if len(Xsource) == 0:
                return 0, 0, 0, 0
            Xtarget, Ytarget = self.outlierRemove(Xtarget, Ytarget)
            if len(Xtarget) == 0:
                return 0, 0, 0, 0
            Xsource, Ysource, Xtarget, Ytarget = self.Bruakfilter(Xsource, Ysource, Xtarget, Ytarget)
            if len(Xsource) == 0 or len(Xtarget) == 0:
                return 0, 0, 0, 0
            Xsource, Ysource = self.outlierRemove(Xsource, Ysource)
            if len(Xsource) == 0 or len(Xtarget) == 0:
                return 0, 0, 0, 0

            return Xsource, Ysource, Xtarget, Ytarget