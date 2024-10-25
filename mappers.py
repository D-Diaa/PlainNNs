import logging
import random
from abc import abstractmethod

from factor_analyzer.rotator import Rotator
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.random_projection import GaussianRandomProjection
from BiHalf import train as BiHalfTrain
from hash import *
from utils import grey


class Mapper:
    def __init__(self, points, p=10):
        """
        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of hyperplanes to use for hashing.
        """
        self.p = p
        self.dimensions = points.shape[1]
        self.mn = points.min(axis=0)
        self.mx = points.max(axis=0)
        self.center = self._normalize(points).mean(axis=0)

    def _normalize(self, points):
        # Normalize the points to the unit hypercube
        return (points - self.mn) / (self.mx - self.mn)

    def _center(self, points):
        return points - self.center

    @abstractmethod
    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        pass

    @abstractmethod
    def transform(self, vector):
        """
        Hash a single vector into a binary string.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        pass


class DSHMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the Density Sensitive Hashing (DSH) instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of bits in the binary hash.
        """
        super().__init__(points, p)
        self.model = None

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings using Density Sensitive Hashing.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        X = self._normalize(vectors)
        X_centered = self._center(X)
        self.model, bits, = DSH(X_centered, self.p)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string using Density Sensitive Hashing.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector)
        X_centered = self._center(X)
        Ym = X_centered @ self.model['U'].T
        res = np.tile(self.model['intercept'], (1,))
        bits = (Ym > res).astype(int)
        index = int(''.join(bits[0].astype(str)), 2)
        return index


class RandomProjectionMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the random projection-based SimHash instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of hyperplanes to use for hashing.
        """
        super().__init__(points, p)
        self.transformer = GaussianRandomProjection(n_components=p)

        print(f"Initialized RandomProjectionMapper with k={self.p} hyperplanes.")

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings using random hyperplanes.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        logging.info("RandomProjectionMapper fit_transform only transforms the data.")
        X = self._normalize(vectors)
        X_centered = self._center(X)
        projections = self.transformer.fit_transform(X_centered)
        bits = (projections >= 0).astype(int)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string using random hyperplanes.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector)
        X_centered = self._center(X)
        projection = self.transformer.transform(X_centered.reshape(1, -1))
        bits = (projection >= 0).astype(int)
        index = int(''.join(bits.astype(str)), 2)
        return index


class PCAMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the PCA-optimized SimHash instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of principal components to use for PCA.
        """
        super().__init__(points, p)
        self.pca = PCA(n_components=self.p)
        self.rotated_loadings = None
        print(f"Initialized PCAOptimizedSimHash with k={self.p} hyperplanes.")

    def fit_transform(self, vectors):
        """
        Hashes multiple vectors into binary strings using PCA-based hyperplanes.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        X = self._normalize(vectors)

        X_centered = self._center(X)

        _ = self.pca.fit_transform(X_centered)

        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)

        # Step 2: Apply Varimax Rotation
        rotator = Rotator(method='equamax')
        self.rotated_loadings = rotator.fit_transform(loadings)

        # Step 3: Transform Data
        X_rotated = np.dot(X_centered, self.rotated_loadings)

        bits = (X_rotated >= 0).astype(int)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hashes a single vector into a binary string using PCA-based hyperplanes.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector)
        X_centered = self._center(X)
        X_rotated = np.dot(X_centered, self.rotated_loadings)
        bits = (X_rotated >= 0).astype(int)
        index = int(''.join(bits[0].astype(str)), 2)
        return index

class ITQMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the ITQ instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of hyperplanes to use for hashing.
        """
        super().__init__(points, p)
        self.model = None

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings using ITQ.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        X = self._normalize(vectors)
        X_centered = self._center(X)
        self.model, bits = ITQ(X_centered, self.p, 50)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string using ITQ.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector)
        X_centered = self._center(X)
        V = self.model['pca'].transform(X_centered.reshape(1, -1))
        Z = V @ self.model['R']
        UX = np.zeros_like(Z)
        UX[Z >= 0] = 1
        index = int(''.join(UX[0].astype(str)), 2)
        return index

class ISOHMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the ISOH instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of hyperplanes to use for hashing.
        """
        super().__init__(points, p)
        self.model = None

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings using ISOH.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        X = self._normalize(vectors)
        X_centered = self._center(X)
        self.model, bits = ISOH(X_centered, self.p)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string using ISOH.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector)
        X_centered = self._center(X)
        Y = X_centered @ self.model['pc'] @ self.model['R']  # Shape: (n_samples, maxbits)
        B = (Y > 0).astype(int)  # Binary codes: 1 if Y > 0, else 0
        index = int(''.join(B.astype(str)), 2)
        return index

class SpHMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initializes the Spherical Hashing (SpH) instance.

        Parameters:
        - points (numpy.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of bits in the binary hash.
        """
        super().__init__(points, p)
        self.model = None

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings using Spherical Hashing.

        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        X = self._normalize(vectors)
        X_centered = self._center(X)
        self.model, bits = SpH(X_centered, self.p)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string using Spherical Hashing.

        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        X = self._normalize(vector.reshape(1, -1))
        X_centered = self._center(X)
        bits = SpH_compress(X_centered, self.model)
        index = int(''.join(bits[0].astype(str)), 2)
        raise index

class HilbertMapper(Mapper):
    def __init__(self, points, p=10, grey_code=False):
        """
        Initialize the Hilbert curve mapper and scale points based on input.

        Parameters:
        - points (np.ndarray): The set of points in k-dimensional space used to initialize the curve.
        - p (int): The order of the Hilbert curve (controls granularity of mapping).
        """
        super().__init__(points, p)
        self.hilbert_curve = HilbertCurve(p, self.dimensions, 16)
        self.grey_code = grey_code

    def fit_transform(self, points):
        """
        Map new points using the Hilbert curve based on the initialized min and max values.

        Parameters:
        - points (np.ndarray): The set of points in k-dimensional space.

        Returns:
        - hilbert_indices (np.ndarray): The indices of points on the Hilbert curve.
        """
        # Scale the points to the range [0, 2^p - 1] using previously determined min and max
        points_normalized = self._normalize(points)
        points_scaled = (points_normalized * (2 ** self.p - 1)).astype(int)

        # Map each point to its corresponding Hilbert index
        hilbert_indices = np.array(self.hilbert_curve.distances_from_points([pt.tolist() for pt in points_scaled]))
        # Sorting disables transform
        hilbert_indices = np.argsort(hilbert_indices)
        if self.grey_code:
            hilbert_indices = [grey(i) for i in hilbert_indices]

        return hilbert_indices

    def transform(self, point):
        raise NotImplementedError("Single point queries are not supported for HilbertMapper.")


class RandomMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initialize the Random mapper.

        Parameters:
        - points (np.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of bits in the binary hash.
        """
        super().__init__(points, p)
        self.size = 0

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into random binary strings.
        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        indices = random.sample(range(len(vectors)), len(vectors))
        self.size = len(vectors)
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a random binary string.
        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        self.size += 1
        return self.size - 1

class BiHalfMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initialize the BiHalf mapper.

        Parameters:
        - points (np.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of bits in the binary hash.
        """
        super().__init__(points, p)
        self.size = 0

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings.
        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        bits = BiHalfTrain(vectors, self.p)
        self.size = len(vectors)
        indices = [int(''.join(bit.astype(str)), 2) for bit in bits]
        return indices

    def transform(self, vector):
        raise NotImplementedError("Single point queries are not supported for BiHalfMapper.")
class DefaultMapper(Mapper):
    def __init__(self, points, p=10):
        """
        Initialize the Default mapper.

        Parameters:
        - points (np.ndarray): The set of points in k-dimensional space used to initialize the hyperplanes.
        - p (int): The number of bits in the binary hash.
        """
        super().__init__(points, p)
        self.size = 0

    def fit_transform(self, vectors):
        """
        Hash multiple vectors into binary strings.
        Parameters:
        - vectors (numpy.ndarray): A 2D array where each row is a vector.

        Returns:
        - indices (list): The indices of the vectors in the hypercube.
        """
        indices = np.arange(len(vectors))
        self.size = len(vectors)
        return indices

    def transform(self, vector):
        """
        Hash a single vector into a binary string.
        Parameters:
        - vector (numpy.ndarray): A single vector.

        Returns:
        - index (int): The index of the vector in the hypercube.
        """
        self.size += 1
        return self.size - 1


# Mapper factory method
def create_mapper(mapper_type, points, p=10):
    """
    Factory method to create a new Mapper instance based on the given type.

    Parameters:
    - mapper_type (str): The type of mapper to create ('pca' or 'hilbert').
    - points (np.ndarray): The set of points in k-dimensional space used to initialize the mapper.
    - p (int): The parameter for the mapper (number of hyperplanes for PCA, order for Hilbert).

    Returns:
    - mapper (Mapper): A new instance of the specified Mapper type.
    """
    if mapper_type == 'pca':
        return PCAMapper(points, p)
    elif mapper_type == 'dsh':
        return DSHMapper(points, p)
    elif mapper_type == 'itq':
        return ITQMapper(points, p)
    elif mapper_type == 'isoh':
        return ISOHMapper(points, p)
    elif mapper_type == 'sph':
        return SpHMapper(points, p)
    elif mapper_type == 'bihalf':
        return BiHalfMapper(points, p)
    elif mapper_type == 'random_projection':
        return RandomProjectionMapper(points, p)
    elif 'hilbert' in mapper_type:
        return HilbertMapper(points, p, grey_code='grey' in mapper_type)
    elif mapper_type == 'random':
        return RandomMapper(points, p)
    elif mapper_type == 'default':
        return DefaultMapper(points, p)
    else:
        raise ValueError(f"Invalid mapper type: {mapper_type}")
