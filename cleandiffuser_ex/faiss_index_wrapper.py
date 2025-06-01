import os
import time
import faiss
import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def apply_dim_weights(data: np.ndarray, dim_weights: np.ndarray) -> np.ndarray:
    if dim_weights is None:
        return data
    if dim_weights.shape[0] != data.shape[1]:
        raise ValueError("dim_weights must match data.shape[1].")
    return data * dim_weights


class FaissIndexIVFWrapper:

    def __init__(
        self,
        similarity_metric: str = "l2",
        nlist: int = 1000,
        data = None,
        dim_weights: np.ndarray = None, 
        device: str = "cuda:0",
    ):
        self.num_data, *self.dims = data.shape
        self.dim_flat = int(np.prod(self.dims))

        self.similarity_metric = similarity_metric.lower()
        self.nlist = nlist
        self.nprobe = 5 # max(1, self.nlist // 10)
        self.data_flat = data.reshape(self.num_data, -1).copy(order = 'C')
        self.device = device
        self.dim_weights_flat = dim_weights.reshape(-1) if dim_weights is not None else None

        if device.lower() == "cpu":
            raise ValueError("[FaissIndexIVFPQWrapper] do not support cpu device")
        elif device.lower().startswith("cuda:"):
            try:
                gpu_id_str = device.split(":")[1]
                self.gpu_id = int(gpu_id_str)
            except (IndexError, ValueError):
                raise ValueError(f"wrong device: {device}")
        else:
            raise ValueError(f"wrong device: {device}")

        # GPU resource
        self.gpu_res = faiss.StandardGpuResources()

        # index
        self.index_cpu = None
        self.index = None
        self.size = 0

        self.build_index(self.data_flat)

    def build_index(self, data: np.ndarray):
        if data.ndim != 2 or data.shape[1] != self.dim_flat:
            raise ValueError(f"data.shape={data.shape}, expected (N, {self.dim_flat})")

        # 1) scale if needed
        data_scaled = apply_dim_weights(data, self.dim_weights_flat)

        # 2) metric
        if self.similarity_metric == 'cosine':
            data_scaled = normalize_vectors(data_scaled)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        n_data = data_scaled.shape[0]
        print(f"[FaissIndexIVFFlatWrapper] building IVF-Flat with {n_data} vectors...")

        # 3) quantizer
        if metric == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(self.dim_flat)
        else:
            print(self.dim_flat)
            quantizer = faiss.IndexFlatL2(self.dim_flat)

        # 4) create IVF Flat
        index_ivf = faiss.IndexIVFFlat(quantizer, self.dim_flat, self.nlist, metric)
        self.index_cpu = index_ivf

        # train
        if not index_ivf.is_trained:
            print("[FaissIndexIVFFlatWrapper] Training IVF quantizer...")
            index_ivf.train(data_scaled)
            print("[FaissIndexIVFFlatWrapper] Training done.")

        # add
        index_ivf.add(data_scaled)
        self.size = n_data
        print(f"[FaissIndexIVFFlatWrapper] CPU IVF-Flat index has {self.size} vectors")

        # to GPU
        self.index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, index_ivf)
        self.index.nprobe = self.nprobe
        print(f"[FaissIndexIVFFlatWrapper] Moved IVF-Flat index to GPU. nprobe={self.index.nprobe}")

    def search(self, queries: np.ndarray, k: int):
        queries_flat = queries.reshape(-1, self.dim_flat)

        if self.index is None:
            raise RuntimeError("Index not built or loaded yet.")

        if queries_flat.shape[1] != self.dim_flat:
            raise ValueError(f"queries_flat shape {queries_flat.shape}, expected (B, {self.dim_flat})")

        # scale queries
        queries_flat_scaled = apply_dim_weights(queries_flat, self.dim_weights_flat)

        # cos => normalize
        if self.similarity_metric == 'cosine':
            queries_flat_scaled = normalize_vectors(queries_flat_scaled)

        distances, indices = self.index.search(queries_flat_scaled, k)
        # these distances are the actual (approx for IP? Actually IVFFlat => exact, but L2 => exact)
        return distances, indices

    def get_original_vectors(self, indices: np.ndarray):
        flat_indices = indices.ravel()
        out_flat = self.data_flat[flat_indices]
        B, K = indices.shape
        out = out_flat.reshape(B, K, *self.dims)
        return out


# --- Exact Faiss Wrapper ---
class FaissIndexWrapper:
    """
    A wrapper for exact Faiss indices (IndexFlatL2 or IndexFlatIP) on GPU.

    Performs exact nearest neighbor search, which is accurate but potentially
    slower than approximate methods like IVF for large datasets.
    """
    def __init__(
        self,
        similarity_metric: str = "l2",
        data: np.ndarray = None, # Expects data like (N, H, D) or (N, D)
        dim_weights: np.ndarray = None, # Expects weights like (H, D) or (D)
        device: str = "cuda:0",
    ):
        """
        Initializes the FaissIndexWrapper.

        Args:
            similarity_metric: 'l2' for Euclidean distance or 'cosine'
                               for cosine similarity (uses Inner Product).
            data: The dataset to index, NumPy array of shape (N, ...).
            dim_weights: Optional weights to apply to dimensions before indexing
                         and searching. Shape should match the flattened feature dim.
            device: The GPU device string (e.g., "cuda:0"). CPU not supported here.
        """
        if data is None:
            raise ValueError("Data must be provided.")
        if not isinstance(data, np.ndarray):
             raise TypeError("Data must be a NumPy array.")
        if data.ndim < 2:
             raise ValueError("Data must have at least 2 dimensions (N, D...).")

        self.num_data, *self.dims = data.shape
        # Flatten dimensions beyond the first (N)
        self.dim_flat = int(np.prod(self.dims)) if self.dims else 0
        if self.dim_flat == 0 :
             raise ValueError("Could not determine flattened dimension from data shape.")

        self.similarity_metric = similarity_metric.lower()
        if self.similarity_metric not in ['l2', 'cosine']:
            raise ValueError("similarity_metric must be 'l2' or 'cosine'")

        # Store original data for retrieval (flattened only along feature dims)
        self.original_data = data # Keep original shape for get_original_vectors

        # Store flattened weights if provided
        self.dim_weights_flat = None
        if dim_weights is not None:
            if not isinstance(dim_weights, np.ndarray):
                 raise TypeError("dim_weights must be a NumPy array.")
            self.dim_weights_flat = dim_weights.reshape(-1) # Flatten weights
            if self.dim_weights_flat.shape[0] != self.dim_flat:
                 raise ValueError(f"Flattened dim_weights length ({self.dim_weights_flat.shape[0]}) must match data's flattened feature dimension ({self.dim_flat}).")

        # --- Device Handling ---
        self.device = device
        if device.lower() == "cpu":
            # Although IndexFlat works on CPU, keeping GPU-only like IVF wrapper for consistency
            raise ValueError("[FaissIndexWrapper] This implementation currently requires a GPU device (e.g., 'cuda:0').")
        elif device.lower().startswith("cuda:"):
            try:
                gpu_id_str = device.split(":")[1]
                self.gpu_id = int(gpu_id_str)
            except (IndexError, ValueError):
                raise ValueError(f"Invalid GPU device string format: {device}. Expected 'cuda:ID'.")
        else:
            raise ValueError(f"Unsupported device string: {device}. Expected 'cuda:ID'.")

        # --- Faiss GPU Resources ---
        try:
            self.gpu_res = faiss.StandardGpuResources()
        except AttributeError:
             print("Warning: faiss.StandardGpuResources() not found. Ensure Faiss GPU support is installed.")
             raise

        # --- Index Variables ---
        self.index_cpu = None # Keep reference to CPU index if needed
        self.index = None     # This will hold the GPU index
        self.size = 0

        # --- Build Index ---
        # Reshape data for building: (N, dim_flat)
        data_flat_for_build = data.reshape(self.num_data, self.dim_flat).copy(order='C')
        self.build_index(data_flat_for_build)

    def build_index(self, data_flat: np.ndarray):
        """
        Builds the exact Faiss index on the GPU.

        Args:
            data_flat: The data reshaped to (N, dim_flat).
        """
        start_time = time.time()
        if data_flat.ndim != 2 or data_flat.shape[1] != self.dim_flat:
            raise ValueError(f"Internal Error: data_flat shape {data_flat.shape} is invalid, expected (N, {self.dim_flat})")
        if data_flat.shape[0] != self.num_data:
             raise ValueError(f"Internal Error: data_flat num samples {data_flat.shape[0]} != initial N {self.num_data}")


        # 1) Apply dimension weights if provided
        data_scaled = apply_dim_weights(data_flat, self.dim_weights_flat)

        # 2) Handle metric and normalization
        if self.similarity_metric == 'cosine':
            print("[FaissIndexWrapper] Normalizing vectors for cosine similarity (using IndexFlatIP).")
            data_scaled = normalize_vectors(data_scaled)
            # Use IndexFlatIP for cosine similarity (max inner product on normalized vectors)
            metric = faiss.METRIC_INNER_PRODUCT
            self.index_cpu = faiss.IndexFlatIP(self.dim_flat)
        else: # 'l2'
            print("[FaissIndexWrapper] Using IndexFlatL2 for Euclidean distance.")
            # Use IndexFlatL2 for Euclidean distance
            metric = faiss.METRIC_L2
            self.index_cpu = faiss.IndexFlatL2(self.dim_flat)

        n_data = data_scaled.shape[0]
        print(f"[FaissIndexWrapper] Building Flat index with {n_data} vectors of dim {self.dim_flat}...")

        # 3) Add data to the CPU index (IndexFlat doesn't require training)
        self.index_cpu.add(data_scaled)
        self.size = self.index_cpu.ntotal
        if self.size != n_data:
             print(f"Warning: Index size {self.size} after add differs from input data size {n_data}")

        print(f"[FaissIndexWrapper] CPU Flat index built with {self.size} vectors.")

        # 4) Transfer index to GPU
        try:
            print(f"[FaissIndexWrapper] Moving Flat index to GPU {self.gpu_id}...")
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, self.index_cpu)
            print(f"[FaissIndexWrapper] Index successfully moved to GPU.")
        except Exception as e:
            print(f"Error moving index to GPU: {e}")
            # Clean up CPU index if GPU transfer fails?
            self.index_cpu = None
            self.index = None
            self.size = 0
            raise RuntimeError("Failed to transfer Faiss index to GPU.") from e

        build_time = time.time() - start_time
        print(f"[FaissIndexWrapper] Index build completed in {build_time:.2f} seconds.")


    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs exact nearest neighbor search on the GPU.

        Args:
            queries: NumPy array of queries, shape (B, ...) matching original data dims.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple containing:
            - distances: NumPy array of distances (L2 squared or negative Inner Product), shape (B, k).
            - indices: NumPy array of indices of the nearest neighbors in the original data, shape (B, k).
        """
        if self.index is None:
            raise RuntimeError("Faiss index has not been built or loaded.")
        if k <= 0:
            raise ValueError("k must be > 0")
        if k > self.size:
             print(f"Warning: k={k} is larger than the index size {self.size}. Returning {self.size} neighbors.")
             k = self.size # Adjust k if it exceeds index size

        # Reshape queries to (B, dim_flat)
        num_queries = queries.shape[0]
        try:
            queries_flat = queries.reshape(num_queries, self.dim_flat)
        except ValueError as e:
             raise ValueError(f"Query shape {queries.shape} incompatible with expected feature dimension {self.dim_flat}.") from e


        # Apply dimension weights to queries
        queries_flat_scaled = apply_dim_weights(queries_flat, self.dim_weights_flat)

        # Normalize queries if using cosine similarity
        if self.similarity_metric == 'cosine':
            queries_flat_scaled = normalize_vectors(queries_flat_scaled)

        # Perform the search on the GPU index
        start_time = time.time()
        distances, indices = self.index.search(queries_flat_scaled, k)
        search_time = time.time() - start_time

        # Note: For METRIC_INNER_PRODUCT, distances returned are negative inner products.
        # For METRIC_L2, distances returned are squared L2 distances.
        return distances, indices

    def get_original_vectors(self, indices: np.ndarray) -> np.ndarray:
        """
        Retrieves the original, unflattened vectors corresponding to the given indices.

        Args:
            indices: A NumPy array of indices (e.g., shape (B, k)) as returned by search.

        Returns:
            A NumPy array containing the original vectors, shape (B, k, *self.dims).
        """
        if self.original_data is None:
             raise RuntimeError("Original data was not stored during initialization.")
        if indices.max() >= self.num_data:
            raise IndexError(f"Received index {indices.max()} which is out of bounds for original data size {self.num_data}.")
        if indices.min() < 0:
             # Faiss typically returns -1 for indices if fewer than k neighbors are found (e.g., if k > N)
             # Handle this case gracefully or raise an error depending on desired behavior.
             # For simplicity here, we'll rely on the check k <= self.size in search.
             # If Faiss *could* return -1 (e.g. in range search), filtering would be needed:
             # valid_indices = indices[indices != -1]
             # result = np.full_like(indices, fill_value=-1, dtype=self.original_data.dtype)
             # result[indices != -1] = self.original_data[valid_indices] ... complex reshaping needed.
             # Assuming indices are valid based on search logic for now.
             pass

        # Directly use the indices on the original data array
        # NumPy's advanced indexing handles multi-dimensional index arrays correctly.
        try:
            output_vectors = self.original_data[indices]
        except IndexError as e:
             print(f"Error during indexing into original_data with indices shape {indices.shape}. Max index: {indices.max()}, Data shape: {self.original_data.shape}")
             raise e

        # Expected shape: indices.shape + self.dims
        # e.g., if indices is (B, K), output is (B, K, *self.dims)
        return output_vectors

    def __len__(self) -> int:
        """Returns the number of vectors in the index."""
        return self.size