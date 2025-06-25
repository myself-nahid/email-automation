# --- START OF FILE faiss_index.py ---

import faiss
import numpy as np
import threading
import os
import pickle
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

INDEX_PATH = "data/faiss_index.bin"
ID_MAP_PATH = "data/faiss_id_map.pkl"
lock = threading.Lock()

class FaissIndex:
    def __init__(self, dim=1536):
        self.dim = dim
        self.index = None
        self.id_map = []
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        self.load()

    def load(self):
        with lock:
            if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
                try:
                    self.index = faiss.read_index(INDEX_PATH)
                    if self.index.d != self.dim:
                        logger.warning(f"Index dimension mismatch. Re-initializing.")
                        self._initialize_empty_index()
                        return

                    with open(ID_MAP_PATH, 'rb') as f:
                        self.id_map = pickle.load(f)
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}", exc_info=True)
                    self._initialize_empty_index()
            else:
                logger.info("FAISS index not found. Initializing new index.")
                self._initialize_empty_index()

    def _initialize_empty_index(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = []
        logger.info(f"Initialized new empty FAISS index.")

    def save(self):
        with lock:
            if self.index is None:
                return
            try:
                faiss.write_index(self.index, INDEX_PATH)
                with open(ID_MAP_PATH, 'wb') as f:
                    pickle.dump(self.id_map, f)
                logger.info(f"Saved FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}", exc_info=True)

    def add(self, email_ids: List[str], embeddings_bytes_list: List[bytes]):
        with lock:
            if self.index is None:
                self._initialize_empty_index()

            valid_embeddings_np = []
            valid_ids_for_batch = []
            
            for email_id, emb_bytes in zip(email_ids, embeddings_bytes_list):
                try:
                    embedding_array = np.frombuffer(emb_bytes, dtype=np.float32)
                    if embedding_array.shape[0] != self.dim:
                        continue
                    valid_embeddings_np.append(embedding_array)
                    valid_ids_for_batch.append(email_id)
                except Exception:
                    continue
            
            if not valid_embeddings_np:
                return

            embeddings_to_add = np.array(valid_embeddings_np).astype('float32')
            self.index.add(embeddings_to_add)
            self.id_map.extend(valid_ids_for_batch)

    def search_with_scores(self, query_embedding: np.ndarray, top_k: int = 5):
        with lock:
            if self.index is None or self.index.ntotal == 0:
                return np.array([]), []

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype('float32')
            
            actual_top_k = min(top_k, self.index.ntotal)
            if actual_top_k == 0:
                return np.array([]), []

            distances, indices = self.index.search(query_embedding, actual_top_k)
            
            valid_indices = indices[0][indices[0] != -1]
            retrieved_ids = [self.id_map[i] for i in valid_indices]
            valid_distances = distances[0][indices[0] != -1]
            
            return valid_distances, retrieved_ids

faiss_index = FaissIndex(dim=1536)