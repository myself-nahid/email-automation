# 1. FIRST - Fix the faiss_index.py file to handle buffer errors better

import faiss
import numpy as np
import threading
import os
import pickle
import sys

DIM = 1536  # OpenAI embedding dimension
INDEX_PATH = "data/faiss_index.bin"
ID_MAP_PATH = "data/faiss_id_map.pkl"
lock = threading.Lock()

class FaissIndex:
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                with open(ID_MAP_PATH, 'rb') as f:
                    self.id_map = pickle.load(f)
                print(f"Loaded FAISS index with {len(self.id_map)} entries")
            else:
                self.index = faiss.IndexFlatL2(DIM)
                self.id_map = []  # list of email_ids in order added
                print("Created new FAISS index")
                self.save()  # Save empty index
        except Exception as e:
            print(f"Error initializing FAISS index: {str(e)}")
            # Fallback to in-memory index
            self.index = faiss.IndexFlatL2(DIM)
            self.id_map = []
            print("Created fallback in-memory FAISS index")

    def save(self):
        with lock:
            try:
                faiss.write_index(self.index, INDEX_PATH)
                with open(ID_MAP_PATH, 'wb') as f:
                    pickle.dump(self.id_map, f)
                print(f"Saved FAISS index with {len(self.id_map)} entries")
            except Exception as e:
                print(f"Error saving FAISS index: {str(e)}")

    def add(self, email_ids, embeddings_bytes_list):
        if not email_ids or not embeddings_bytes_list:
            print("Warning: Attempted to add empty data to FAISS index")
            return
            
        # embeddings_bytes_list: list of bytes from SQLite, convert back to float32 arrays
        try:
            embeddings = []
            valid_ids = []
            
            for i, (email_id, embedding_bytes) in enumerate(zip(email_ids, embeddings_bytes_list)):
                try:
                    # Check if embedding_bytes is valid
                    if not embedding_bytes:
                        print(f"Skipping email {email_id}: empty embedding")
                        continue
                    
                    # Try to convert to numpy array
                    if len(embedding_bytes) == 6144:  # 1536 * 4 bytes (float32)
                        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                    elif len(embedding_bytes) == 12288:  # 1536 * 8 bytes (float64)
                        print(f"Converting float64 to float32 for email {email_id}")
                        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float64).astype(np.float32)
                    else:
                        print(f"Skipping email {email_id}: invalid embedding size {len(embedding_bytes)} bytes")
                        continue
                    
                    # Validate array size
                    if len(embedding_array) != 1536:
                        print(f"Skipping email {email_id}: wrong embedding dimension {len(embedding_array)}")
                        continue
                    
                    # Check for invalid values
                    if not np.isfinite(embedding_array).all():
                        print(f"Skipping email {email_id}: contains invalid values")
                        continue
                    
                    embeddings.append(embedding_array)
                    valid_ids.append(email_id)
                    
                except Exception as e:
                    print(f"Error processing embedding for {email_id}: {str(e)}")
                    continue
            
            if not embeddings:
                print("No valid embeddings to add to FAISS index")
                return
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            print(f"Adding {len(embeddings)} valid embeddings to FAISS index")
            
            with lock:
                self.index.add(embeddings_array)
                self.id_map.extend(valid_ids)
                self.save()
                print(f"Successfully added {len(valid_ids)} embeddings. Total index size: {self.index.ntotal}")
                
        except Exception as e:
            print(f"Error adding to FAISS index: {str(e)}")
            print(f"Embeddings array shape: {embeddings_array.shape if 'embeddings_array' in locals() else 'Not created'}")
            print(f"Embeddings array dtype: {embeddings_array.dtype if 'embeddings_array' in locals() else 'Not created'}")

    def search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            print("Warning: Searching empty FAISS index")
            return []
            
        try:
            # Ensure query embedding is the right format
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            elif not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Ensure it's 2D array for FAISS
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype('float32')
            
            with lock:
                distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
                results = []
                for idx in indices[0]:
                    if idx >= 0 and idx < len(self.id_map):  # Ensure index is valid
                        results.append(self.id_map[idx])
                return results
        except Exception as e:
            print(f"Error searching FAISS index: {str(e)}")
            return []

# Apply Python version-specific fixes
def fix_faiss_compatibility():
    if sys.version_info >= (3, 12):
        # FAISS may have issues with Python 3.12 due to removed features
        print("Warning: FAISS may have compatibility issues with Python 3.12")
        # Any specific fixes could be implemented here if needed

fix_faiss_compatibility()
faiss_index = FaissIndex()