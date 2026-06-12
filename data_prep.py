import json
import pinecone
# from sentence_transformers import SentenceTransformer  # COMMENTED OUT - Using Mistral API instead
from config import Config
import time
import os
import requests


class DataPrepper:
    def __init__(self):
        # ========== COMMENTED OUT - Original embedding model loading ==========
        # print("🔄 Loading sentence-transformers model (1024 dimensions, FREE)...")
        # self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        # print(f"✅ Loaded {Config.EMBEDDING_MODEL}")
        # print(f"   Model dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        # ======================================================================
        
        print("🔄 Initializing Pinecone and Mistral API...")
        
        # Initialize Mistral client for API calls (FIXED - added this)
        self.mistral_api_key = Config.MISTRAL_API_KEY
        
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = pc.Index(Config.PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {Config.PINECONE_INDEX_NAME}")
        
        # Verify dimension matches
        stats = self.index.describe_index_stats()
        print(f"   Pinecone dimension: {stats.dimension}")
    
    def get_embedding(self, text):
        """Generate embedding using Mistral API (1024 dimensions)"""
        # ========== COMMENTED OUT - Original local embedding generation ==========
        # # Generate embedding locally (FREE, no API call!)
        # # For E5 models, add instruction prefix
        # if "e5" in Config.EMBEDDING_MODEL.lower():
        #     text = f"passage: {text}"
        # 
        # embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        # return embedding.tolist()
        # ===========================================================================
        
        # NEW: Using Mistral API for embeddings (FIXED - direct API call)
         # Validate input
        if not text or not isinstance(text, str):
            return None
        
        text = text.strip()
        if len(text) == 0:
            return None
        try:
            url = "https://api.mistral.ai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "mistral-embed",
                "inputs": text
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                print(f"  API Error {response.status_code}: {response.text[:200]}")
                return None

            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
            
        except Exception as e:
            print(f"⚠️ Error generating embedding: {e}")
            return None
    
    def vector_exists(self, vector_id):
        """Check if vector already exists in Pinecone"""
        # FIXED - This should check Pinecone, not generate embeddings
        try:
            result = self.index.fetch(ids=[vector_id])
            exists = vector_id in result['vectors']
            return exists
        except Exception as e:
            return False
    
    def load_policies(self):
        """Read regional policies from existing txt file"""
        print(f"📖 Loading policies from {Config.POLICIES_FILE}...")
        
        if not os.path.exists(Config.POLICIES_FILE):
            raise FileNotFoundError(f"Policy file {Config.POLICIES_FILE} not found!")
        
        with open(Config.POLICIES_FILE, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        # Parse policies by region
        policies = []
        lines = policy_text.strip().split('\n')
        
        for line in lines:
            if 'Zone:' in line:
                parts = line.split('Zone:')
                region = parts[0].strip()
                rule = parts[1].strip()
                policies.append({'region': region, 'rule': rule})
        
        print(f"✅ Loaded {len(policies)} policies")
        return policies
    
    def load_towers(self):
        """Read towers inventory from existing json file"""
        print(f"📖 Loading towers from {Config.TOWERS_FILE}...")
        
        if not os.path.exists(Config.TOWERS_FILE):
            raise FileNotFoundError(f"Towers file {Config.TOWERS_FILE} not found!")
        
        with open(Config.TOWERS_FILE, 'r', encoding='utf-8') as f:
            towers = json.load(f)
        
        print(f"✅ Loaded {len(towers)} towers")
        return towers
    
    def embed_policies(self, policies):
        """Generate embeddings only for missing policies"""
        print("🔄 Checking policies in Pinecone...")
        
        vectors_to_upsert = []
        existing_count = 0
        
        for i, policy in enumerate(policies):
            policy_id = f"policy_{policy['region'].replace('-', '_').replace(' ', '_')}"
            
            # Check if already exists
            if self.vector_exists(policy_id):
                existing_count += 1
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(policies)} policies...")
                continue
            
            print(f"  🔄 Generating embedding for {policy['region']}...")
            
            # ========== COMMENTED OUT - Original E5 model prefix ==========
            # # For E5 models, use appropriate prefix
            # if "e5" in Config.EMBEDDING_MODEL.lower():
            #     search_text = f"passage: Region: {policy['region']}. Policy Rule: {policy['rule']}"
            # else:
            #     search_text = f"Region: {policy['region']}. Policy Rule: {policy['rule']}"
            # ===============================================================
            
            # NEW: Using Mistral API (no prefix needed for mistral-embed)
            search_text = f"Region: {policy['region']}. Policy Rule: {policy['rule']}"
            
            embedding = self.get_embedding(search_text)
            
            if embedding is None:
                print(f"  ⚠️ Skipping {policy['region']} - embedding generation failed")
                continue
            
            vectors_to_upsert.append({
                'id': policy_id,
                'values': embedding,
                'metadata': {
                    'region': policy['region'],
                    'rule': policy['rule'],
                    'type': 'policy'
                }
            })
            
            # Show progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(policies)} policies...")
        
        print(f"  Found {existing_count} existing policies")
        
        # Upsert only new vectors
        if vectors_to_upsert:
            print(f"📤 Upserting {len(vectors_to_upsert)} new policies...")
            batch_size = 50
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                self.index.upsert(vectors=batch)
                print(f"    Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            print(f"✅ Added {len(vectors_to_upsert)} new policies")
        else:
            print("✅ All policies already exist in Pinecone")
        
        return len(vectors_to_upsert)
    
    def embed_towers(self, towers):
        """Generate embeddings only for missing towers"""
        print("🔄 Checking towers in Pinecone...")
        
        vectors_to_upsert = []
        existing_count = 0
        
        for i, tower in enumerate(towers):
            tower_id = f"tower_{tower['tower_id']}"
            
            # Check if already exists
            if self.vector_exists(tower_id):
                existing_count += 1
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(towers)} towers...")
                continue
            
            print(f"  🔄 Generating embedding for tower {tower['tower_id']}...")
            
            # ========== COMMENTED OUT - Original E5 model prefix ==========
            # # For E5 models, use appropriate prefix
            # if "e5" in Config.EMBEDDING_MODEL.lower():
            #     search_text = f"passage: Tower {tower['tower_id']} in {tower['region']}. Max capacity: {tower['max_allowed_weight_kg']}kg. Current load: {tower['current_weight_kg']}kg"
            # else:
            #     search_text = f"Tower {tower['tower_id']} in {tower['region']}. Max capacity: {tower['max_allowed_weight_kg']}kg. Current load: {tower['current_weight_kg']}kg"
            # ===============================================================
            
            # NEW: Using Mistral API (no prefix needed for mistral-embed)
            search_text = f"Tower {tower['tower_id']} in {tower['region']}. Max capacity: {tower['max_allowed_weight_kg']}kg. Current load: {tower['current_weight_kg']}kg"
            
            embedding = self.get_embedding(search_text)
            
            if embedding is None:
                print(f"  ⚠️ Skipping tower {tower['tower_id']} - embedding generation failed")
                continue
            
            vectors_to_upsert.append({
                'id': tower_id,
                'values': embedding,
                'metadata': {
                    'tower_id': tower['tower_id'],
                    'region': tower['region'],
                    'max_allowed_weight_kg': tower['max_allowed_weight_kg'],
                    'current_weight_kg': tower['current_weight_kg'],
                    'type': 'tower'
                }
            })
            
            # Show progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(towers)} towers...")
        
        print(f"  Found {existing_count} existing towers")
        
        # Upsert only new vectors
        if vectors_to_upsert:
            print(f"📤 Upserting {len(vectors_to_upsert)} new towers...")
            batch_size = 50
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                self.index.upsert(vectors=batch)
                print(f"    Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            print(f"✅ Added {len(vectors_to_upsert)} new towers")
        else:
            print("✅ All towers already exist in Pinecone")
        
        return len(vectors_to_upsert)
    
    def get_stats(self):
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            print(f"\n📊 Pinecone Index Stats:")
            print(f"  Total vectors: {stats.total_vector_count}")
            print(f"  Dimension: {stats.dimension}")
            return stats
        except Exception as e:
            print(f"⚠️ Could not get stats: {e}")
            return None
    
    def clear_all_vectors(self):
        """Clear all vectors from index"""
        try:
            self.index.delete(delete_all=True)
            print("✅ Cleared all vectors from Pinecone")
            return True
        except Exception as e:
            print(f"⚠️ Error clearing vectors: {e}")
            return False
    
    def run(self):
        """Main function to prepare all data"""
        print("🚀 Starting data preparation...")
        print(f"📐 Using Mistral API for embeddings (1024 dimensions)")
        # ========== COMMENTED OUT - Original model print ==========
        # print(f"📐 Using embedding model: {Config.EMBEDDING_MODEL}")
        # ===========================================================
        
        # Load data from existing files
        policies = self.load_policies()
        towers = self.load_towers()
        
        # Generate and store embeddings (only for missing ones)
        new_policies = self.embed_policies(policies)
        new_towers = self.embed_towers(towers)
        
        # Show stats
        self.get_stats()
        
        print(f"\n✅ Data preparation complete!")
        print(f"📊 Summary:")
        print(f"  - Total policies: {len(policies)}")
        print(f"  - New policies added: {new_policies}")
        print(f"  - Total towers: {len(towers)}")
        print(f"  - New towers added: {new_towers}")
        print(f"  - Total vectors in Pinecone: {self.index.describe_index_stats().total_vector_count}")

if __name__ == "__main__":
    prepper = DataPrepper()
    
    # Optional: Clear existing vectors if you want to regenerate everything
    # Uncomment the line below if you want to clear all existing vectors
    # prepper.clear_all_vectors()
    
    prepper.run()