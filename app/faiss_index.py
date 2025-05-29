# --- START OF FILE faiss_index.py ---

import faiss
import numpy as np
import threading
import os
import pickle
import sys # Not strictly needed for fix_faiss_compatibility unless you add logic
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# DIM = 1536 # OpenAI embedding dimension - This should be instance variable or passed
INDEX_PATH = "data/faiss_index.bin" # Consider making paths configurable
ID_MAP_PATH = "data/faiss_id_map.pkl"
lock = threading.Lock()

class FaissIndex:
    def __init__(self, dim=1536): # Make dimension configurable
        self.dim = dim
        self.index = None
        self.id_map = []  # list of email_ids in order added
        
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        self.load() # Attempt to load on initialization

    def load(self):
        with lock: # Ensure thread safety during load
            if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
                try:
                    self.index = faiss.read_index(INDEX_PATH)
                    if self.index.d != self.dim:
                        logger.warning(f"Loaded FAISS index dimension {self.index.d} != configured {self.dim}. Re-initializing.")
                        self._initialize_empty_index()
                        return # Stop loading if dim mismatch

                    with open(ID_MAP_PATH, 'rb') as f:
                        self.id_map = pickle.load(f)
                    logger.info(f"Loaded FAISS index from {INDEX_PATH} with {len(self.id_map)} entries. Index ntotal: {self.index.ntotal}")
                except Exception as e:
                    logger.error(f"Error loading FAISS index or ID map from {INDEX_PATH}/{ID_MAP_PATH}: {e}", exc_info=True)
                    self._initialize_empty_index() # Fallback to new empty index
            else:
                logger.info(f"FAISS index file ({INDEX_PATH}) or ID map ({ID_MAP_PATH}) not found. Initializing new index.")
                self._initialize_empty_index()

    def _initialize_empty_index(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = []
        logger.info(f"Initialized new empty FAISS index with dimension {self.dim}.")


    def save(self):
        with lock:
            if self.index is None:
                logger.warning("Attempted to save FAISS index, but it's not initialized.")
                return
            try:
                faiss.write_index(self.index, INDEX_PATH)
                with open(ID_MAP_PATH, 'wb') as f:
                    pickle.dump(self.id_map, f)
                logger.info(f"Saved FAISS index ({self.index.ntotal} vectors, {len(self.id_map)} IDs) to {INDEX_PATH} and {ID_MAP_PATH}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}", exc_info=True)

    def add(self, email_ids: List[str], embeddings_bytes_list: List[bytes]):
        with lock: # Ensure thread safety for add operations
            if self.index is None:
                logger.error("FAISS index not initialized. Cannot add embeddings.")
                self._initialize_empty_index() # Try to init if not already

            if not email_ids or not embeddings_bytes_list:
                logger.warning("Attempted to add empty email_ids or embeddings_bytes_list to FAISS.")
                return
            if len(email_ids) != len(embeddings_bytes_list):
                logger.error(f"Mismatch between number of email_ids ({len(email_ids)}) and embeddings ({len(embeddings_bytes_list)}). Aborting add.")
                return
            
            valid_embeddings_np = []
            valid_ids_for_batch = []
            expected_byte_len = self.dim * 4 # float32 is 4 bytes

            for i, (email_id, emb_bytes) in enumerate(zip(email_ids, embeddings_bytes_list)):
                if not emb_bytes or not isinstance(emb_bytes, bytes) or len(emb_bytes) != expected_byte_len:
                    logger.warning(f"Skipping email_id {email_id}: Invalid embedding bytes (length {len(emb_bytes) if emb_bytes else 'None'}, expected {expected_byte_len}).")
                    continue
                try:
                    embedding_array = np.frombuffer(emb_bytes, dtype=np.float32)
                    if embedding_array.shape[0] != self.dim: # Should be (DIM,)
                        logger.warning(f"Skipping email_id {email_id}: Incorrect embedding dimension after frombuffer (shape {embedding_array.shape}, expected ({self.dim},)).")
                        continue
                    if not np.isfinite(embedding_array).all():
                        logger.warning(f"Skipping email_id {email_id}: Embedding contains NaN or Inf values.")
                        continue
                    valid_embeddings_np.append(embedding_array)
                    valid_ids_for_batch.append(email_id)
                except Exception as e:
                    logger.error(f"Error converting embedding bytes for email_id {email_id}: {e}", exc_info=True)
            
            if not valid_embeddings_np:
                logger.info("No valid embeddings to add to FAISS in this batch.")
                return
            
            try:
                embeddings_to_add_array = np.array(valid_embeddings_np, dtype=np.float32) # Ensure it's a 2D array
                if embeddings_to_add_array.ndim == 1: # Should be (N, DIM)
                     embeddings_to_add_array = embeddings_to_add_array.reshape(-1, self.dim)

                self.index.add(embeddings_to_add_array)
                self.id_map.extend(valid_ids_for_batch)
                logger.info(f"Successfully added {len(valid_ids_for_batch)} embeddings to FAISS. Total index size: {self.index.ntotal}, ID map size: {len(self.id_map)}")
                # Consider if save() should be called here or less frequently by the caller
            except Exception as e:
                logger.error(f"Error during faiss.index.add: {e}", exc_info=True)


    def search_with_scores(self, query_embedding: np.ndarray, top_k: int = 5, account_id: Optional[str] = None):
        # account_id is not used for filtering in this basic FAISS setup.
        # Filtering by account_id should happen at the DB retrieval stage after getting IDs from FAISS.
        with lock: # Ensure thread safety for search
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS search: Index is empty or not initialized.")
                return np.array([]), [] # Return empty list for distances and IDs

            if not isinstance(query_embedding, np.ndarray):
                logger.error(f"FAISS search: query_embedding is not a numpy array (type: {type(query_embedding)}).")
                return np.array([]), []
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1) # Ensure 2D for FAISS search
            if query_embedding.shape[1] != self.dim:
                logger.error(f"FAISS search: query_embedding dimension mismatch (shape {query_embedding.shape}, expected (1, {self.dim})).")
                return np.array([]), []
            
            query_embedding = query_embedding.astype(np.float32) # Ensure correct dtype

            try:
                # Ensure top_k is not greater than the number of items in the index
                actual_top_k = min(top_k, self.index.ntotal)
                if actual_top_k == 0 : return np.array([]), []


                distances, indices = self.index.search(query_embedding, actual_top_k)
                
                # indices[0] contains the array of indices for the first query vector
                # Filter out -1 indices (can happen if k > ntotal, though we guard with actual_top_k)
                valid_faiss_indices = indices[0][indices[0] != -1]
                
                retrieved_email_ids = []
                valid_distances_list = []

                for i, faiss_idx in enumerate(valid_faiss_indices):
                    if 0 <= faiss_idx < len(self.id_map):
                        retrieved_email_ids.append(self.id_map[faiss_idx])
                        # Ensure we only take distances corresponding to valid indices
                        if i < len(distances[0]):
                             valid_distances_list.append(distances[0][i])
                    else:
                        logger.warning(f"FAISS search: Retrieved invalid FAISS index {faiss_idx} which is out of id_map bounds ({len(self.id_map)}).")
                
                return np.array(valid_distances_list, dtype=np.float32), retrieved_email_ids
            except Exception as e:
                logger.error(f"Error during FAISS search: {e}", exc_info=True)
                return np.array([]), []

# Initialize a global instance. Ensure DIM matches your embedding model.
faiss_index = FaissIndex(dim=1536)

# --- END OF FILE faiss_index.py ---