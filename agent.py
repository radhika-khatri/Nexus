from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
import json
import re
import requests
from tools import TowerTools
from config import Config

class AgentState(TypedDict):
    input_text: str
    extracted_data: Dict[str, Any]
    tower_validation: Dict[str, Any]
    policy_validation: Dict[str, Any]
    final_judgment: Dict[str, Any]
    messages: Annotated[List[Dict[str, str]], operator.add]

class LeaseAgent:
    def __init__(self):
        self.tools = TowerTools()
        self.graph = self._build_graph()
    
    def _call_mistral_api(self, prompt):
        """Direct API call to Mistral for extraction only"""
        headers = {
            "Authorization": f"Bearer {Config.MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": Config.MISTRAL_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                Config.MISTRAL_API_URL,
                headers=headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return json.loads(result['choices'][0]['message']['content'])
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            return None
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract", self.extract_data)
        workflow.add_node("check_tower", self.check_tower)
        workflow.add_node("check_policy", self.check_policy)
        workflow.add_node("make_judgment", self.make_judgment)
        
        # Add edges
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "check_tower")
        workflow.add_edge("check_tower", "check_policy")
        workflow.add_edge("check_policy", "make_judgment")
        workflow.add_edge("make_judgment", END)
        
        return workflow.compile()
    
    def extract_data(self, state: AgentState):
        """Extract lease request details using Mistral API"""
        prompt = f"""Extract key information from this telecom tower lease request. Return ONLY valid JSON, no other text.
        
Request: "{state['input_text']}"

Return JSON with these exact fields:
{{
    "operator": "name of the mobile operator (e.g., Du, Etisalat, Verizon, Vodafone)",
    "equipment_type": "type of equipment being installed",
    "weight_kg": weight as a number (extract like 15kg -> 15),
    "height_meters": height as a number (extract like 40 meters -> 40),
    "tower_id": "tower identifier like TWR-101"
}}

If a field cannot be found, use null for that field."""
        
        extracted = self._call_mistral_api(prompt)
        
        if not extracted:
            # Fallback to regex extraction if API fails
            extracted = self._regex_extraction(state['input_text'])
        
        return {"extracted_data": extracted}
    
    def _regex_extraction(self, text):
        """Fallback extraction using regex patterns"""
        tower_match = re.search(r'TWR-\d{3}', text)
        tower_id = tower_match.group(0) if tower_match else None
        
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text)
        weight = float(weight_match.group(1)) if weight_match else 0
        
        height_match = re.search(r'(\d+(?:\.\d+)?)\s*meters?', text)
        height = float(height_match.group(1)) if height_match else 0
        
        operators = ['Du', 'Etisalat', 'Verizon', 'Vodafone', 'AT&T', 'T-Mobile']
        operator = None
        for op in operators:
            if re.search(rf'\b{op}\b', text, re.IGNORECASE):
                operator = op
                break
        
        equipment_match = re.search(r'(?:mount|install|place)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)', text, re.IGNORECASE)
        equipment = equipment_match.group(1) if equipment_match else "equipment"
        
        return {
            "operator": operator,
            "equipment_type": equipment,
            "weight_kg": weight,
            "height_meters": height,
            "tower_id": tower_id
        }
    
    def check_tower(self, state: AgentState):
        """Validate tower existence and weight capacity"""
        extracted = state['extracted_data']
        
        if not extracted.get('tower_id'):
            return {
                "tower_validation": {
                    "error": "No tower ID specified in request",
                    "weight_ok": False
                }
            }
        
        tower_check = self.tools.check_weight_capacity(
            extracted['tower_id'],
            extracted.get('weight_kg', 0)
        )
        
        return {"tower_validation": tower_check}
    
    def check_policy(self, state: AgentState):
        """Validate regional policy compliance using RAG"""
        extracted = state['extracted_data']
        tower_check = state['tower_validation']
        
        if tower_check.get('error') or not extracted.get('tower_id'):
            return {
                "policy_validation": {
                    "error": "Cannot check policy without valid tower",
                    "is_compliant": False
                }
            }
        
        region = tower_check.get('region')
        
        if not region:
            return {
                "policy_validation": {
                    "error": "Could not determine region for tower",
                    "is_compliant": False
                }
            }
        
        # Use RAG to get relevant policy
        policy_check = self.tools.get_regional_policy(
            region,
            height=extracted.get('height_meters'),
            weight=extracted.get('weight_kg')
        )
        
        return {"policy_validation": policy_check}
    
    def make_judgment(self, state: AgentState):
        """Make final approval/rejection decision"""
        extracted = state['extracted_data']
        tower_check = state['tower_validation']
        policy_check = state['policy_validation']
        
        approved = True
        reasons = []
        
        # Check tower existence
        if tower_check.get('error'):
            approved = False
            reasons.append(tower_check['error'])
        
        # Check weight capacity
        elif not tower_check.get('weight_ok', False):
            approved = False
            excess = tower_check.get('excess_kg', 0)
            if excess > 0:
                reasons.append(
                    f"Weight capacity exceeded: {tower_check['current_weight_kg']}kg + "
                    f"{tower_check['requested_weight_kg']}kg = {tower_check['new_total_kg']}kg > "
                    f"{tower_check['max_allowed_weight_kg']}kg (exceeds by {excess}kg)"
                )
            else:
                reasons.append("Weight capacity check failed")
        
        # Check policy compliance
        if policy_check.get('error'):
            approved = False
            reasons.append(policy_check['error'])
        elif not policy_check.get('is_compliant', False):
            approved = False
            reasons.extend(policy_check.get('violations', ['Policy violation']))
        
        # Prepare final judgment
        judgment = {
            "status": "APPROVED" if approved else "REJECTED",
            "reasons": reasons if reasons else ["All requirements satisfied"],
            "summary": f"Lease request from {extracted.get('operator', 'Unknown operator')} for {extracted.get('tower_id', 'unknown tower')} is {'approved' if approved else 'rejected'}",
            "details": {
                "request": {
                    "operator": extracted.get('operator'),
                    "tower_id": extracted.get('tower_id'),
                    "equipment": extracted.get('equipment_type'),
                    "weight_kg": extracted.get('weight_kg'),
                    "height_meters": extracted.get('height_meters')
                },
                "tower_assessment": {
                    "region": tower_check.get('region'),
                    "current_load_kg": tower_check.get('current_weight_kg'),
                    "max_capacity_kg": tower_check.get('max_allowed_weight_kg'),
                    "requested_addition_kg": tower_check.get('requested_weight_kg'),
                    "new_total_kg": tower_check.get('new_total_kg'),
                    "within_capacity": tower_check.get('weight_ok', False)
                },
                "policy_assessment": {
                    "applicable_rule": policy_check.get('policy_rule'),
                    "compliant": policy_check.get('is_compliant', False),
                    "violations": policy_check.get('violations', [])
                }
            }
        }
        
        return {"final_judgment": judgment}
    
    def process_request(self, input_text):
        """Process a lease request through the agent"""
        initial_state = {
            "input_text": input_text,
            "extracted_data": {},
            "tower_validation": {},
            "policy_validation": {},
            "final_judgment": {},
            "messages": []
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return result['final_judgment']
        except Exception as e:
            return {
                "status": "ERROR",
                "reasons": [f"Processing error: {str(e)}"],
                "summary": "Failed to process lease request",
                "details": {"error": str(e)}
            }