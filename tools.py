import json
import pinecone
# from sentence_transformers import SentenceTransformer  # COMMENTED OUT - Using Mistral API instead
from config import Config
import re
import requests  # ADDED for Mistral API calls
import time

class TowerTools:
    def __init__(self):
        print("🔄 Initializing TowerTools...")
        
        # Initialize Pinecone
        self.pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
        
        # ========== COMMENTED OUT - Original sentence-transformers model loading ==========
        # # Load local embedding model (1024 dimensions, FREE)
        # print(f"  Loading embedding model: {Config.EMBEDDING_MODEL}")
        # self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        # print(f"  Model dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        # ===================================================================================
        
        # NEW: Store API key for Mistral calls
        self.mistral_api_key = Config.MISTRAL_API_KEY
        
        # Load towers as backup for exact lookups
        with open(Config.TOWERS_FILE, 'r') as f:
            self.towers = json.load(f)
        
        print("✅ TowerTools initialized")
    
    def get_embedding(self, text, retry_count=0):
        """Generate embedding using Mistral API (1024 dimensions) with retry logic"""
        # ========== COMMENTED OUT - Original local embedding generation ==========
        # # Generate embedding locally (FREE, 1024 dimensions)
        # # For E5 models, add instruction prefix for queries
        # if "e5" in Config.EMBEDDING_MODEL.lower():
        #     text = f"query: {text}"
        # 
        # embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        # return embedding.tolist()
        # ===========================================================================
        
        # NEW: Using Mistral API for embeddings with retry logic
        try:
            # Validate input
            if not text or not isinstance(text, str):
                print(f"⚠️ Error: Invalid text input (empty or wrong type)")
                return None
            
            # Clean and validate text
            text = text.strip()
            if len(text) == 0:
                print(f"⚠️ Error: Empty text after stripping")
                return None
            
            # Truncate if too long
            if len(text) > 8000:
                print(f"  ⚠️ Text too long ({len(text)} chars), truncating to 8000")
                text = text[:8000]
            
            url = "https://api.mistral.ai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            # Working format: "input" as string
            data = {
                "model": "mistral-embed",
                "input": text
            }
            
            # Increase timeout to 60 seconds and add retry
            print(f"  📤 Embedding query (attempt {retry_count + 1}): '{text[:80]}...' (length: {len(text)})")
            
            response = requests.post(url, headers=headers, json=data, timeout=60)  # Increased to 60 seconds
            
            if response.status_code != 200:
                print(f"  ❌ API Error {response.status_code}: {response.text[:200]}")
                # Retry up to 2 times on server errors
                if retry_count < 2 and response.status_code >= 500:
                    print(f"  🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                    return self.get_embedding(text, retry_count + 1)
                return None
            
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
            
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout error (attempt {retry_count + 1})")
            if retry_count < 2:  # Retry up to 2 times on timeout
                print(f"  🔄 Retrying in 3 seconds...")
                time.sleep(3)
                return self.get_embedding(text, retry_count + 1)
            else:
                print(f"  ❌ Failed after {retry_count + 1} attempts")
                return None
        except Exception as e:
            print(f"⚠️ Error generating embedding: {e}")
            return None
    
    def get_tower_by_id(self, tower_id):
        """Fetch tower details using exact ID match"""
        for tower in self.towers:
            if tower['tower_id'] == tower_id:
                return tower
        return None
    
    def check_weight_capacity(self, tower_id, additional_weight):
        """Check if tower can accommodate additional weight"""
        tower = self.get_tower_by_id(tower_id)
        
        if not tower:
            return {
                "error": f"Tower {tower_id} not found in inventory",
                "weight_ok": False
            }
        
        current = tower['current_weight_kg']
        max_weight = tower['max_allowed_weight_kg']
        new_total = current + additional_weight
        is_ok = new_total <= max_weight
        
        return {
            "tower_id": tower_id,
            "region": tower.get('region'),
            "current_weight_kg": current,
            "max_allowed_weight_kg": max_weight,
            "requested_weight_kg": additional_weight,
            "new_total_kg": new_total,
            "weight_ok": is_ok,
            "remaining_capacity_kg": max_weight - new_total if is_ok else 0,
            "excess_kg": new_total - max_weight if not is_ok else 0
        }
    
    def get_regional_policy(self, region, height=None, weight=None):
        """Retrieve policy using semantic search from Pinecone"""
        # Create query for policy search
        query_parts = []
        if region:
            query_parts.append(f"Region: {region}")
        if height:
            query_parts.append(f"height {height} meters")
        if weight:
            query_parts.append(f"weight {weight} kg")
        
        if not query_parts:
            query_text = "general policy"
        else:
            query_text = " ".join(query_parts)
        
        # Validate that we have meaningful query text
        if not query_text or len(query_text.strip()) < 3:
            print(f"⚠️ Query text too short: '{query_text}'")
            query_text = f"Policy for {region}" if region else "general policy"
        
        print(f"  🔍 Getting embedding for: {query_text[:50]}...")
        query_embedding = self.get_embedding(query_text)
        
        if query_embedding is None:
            return {
                "error": "Failed to generate query embedding after retries",
                "region": region,
                "policy_rule": "Unable to generate embedding. Please try again.",
                "violations": ["Embedding generation timed out - network issue"],
                "is_compliant": False
            }
        
        try:
            # Search Pinecone for relevant policy
            results = self.index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True,
                filter={"type": {"$eq": "policy"}}
            )
            
            if not results['matches']:
                return {
                    "error": f"No policy found for region {region}",
                    "region": region,
                    "policy_rule": "No specific policy found in database",
                    "violations": [f"Regional policy for {region} not found"],
                    "is_compliant": False
                }
            
            # Get the most relevant policy
            policy_metadata = results['matches'][0]['metadata']
            
            # Validate against policy
            return self._validate_policy(policy_metadata, height, weight)
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return {
                "error": f"Error retrieving policy: {str(e)}",
                "region": region,
                "policy_rule": "Query failed",
                "violations": ["Could not complete policy check"],
                "is_compliant": False
            }
    
    def _validate_policy(self, policy_metadata, height=None, weight=None):
        """Validate request against policy rules"""
        rule = policy_metadata.get('rule', '')
        region = policy_metadata.get('region', '')
        
        violations = []
        
        # Check height restrictions
        if height is not None and height > 0:
            height_patterns = [
                r'height.*?(\d+)\s*meters',
                r'(\d+)\s*meters.*?height',
                r'maximum.*?height.*?(\d+)',
                r'limited to (\d+) meters'
            ]
            
            for pattern in height_patterns:
                match = re.search(pattern, rule, re.IGNORECASE)
                if match:
                    max_height = int(match.group(1))
                    if height > max_height:
                        violations.append(f"Height {height}m exceeds maximum of {max_height}m in {region}")
                    break
        
        # Check weight restrictions
        if weight is not None and weight > 0:
            weight_patterns = [
                r'no single tenant.*?(\d+)\s*kg',
                r'may not exceed (\d+) kg',
                r'maximum.*?weight.*?(\d+)',
                r'exceed (\d+)kg',
                r'(\d+)\s*kg\s+limit'
            ]
            
            for pattern in weight_patterns:
                match = re.search(pattern, rule, re.IGNORECASE)
                if match:
                    max_weight = int(match.group(1))
                    if weight > max_weight:
                        violations.append(f"Weight {weight}kg exceeds maximum single tenant weight of {max_weight}kg in {region}")
                    break
        
        return {
            "region": region,
            "policy_rule": rule,
            "violations": violations,
            "is_compliant": len(violations) == 0
        }