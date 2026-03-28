"""Generation schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class GenerateRequest(BaseModel):
    twin_id: str
    context: str = Field(min_length=1, max_length=16000)
    guardrails: dict = {}
    mode: str = Field(default="CONVERSATION", pattern="^(CONVERSATION|TRAINING|REFINEMENT|ASSISTANT)$")
    conversation_history: Optional[List[Dict]] = None
    deployment_scope: str = "TRAINING_AREA"


class GuardrailChecks(BaseModel):
    personality_consistency: str = "PASSED"
    content_safety: str = "PASSED"
    brand_compliance: str = "PASSED"
    deployment_scope: str = "PASSED"
    legal_compliance: str = "PASSED"
    knowledge_accuracy: str = "PASSED"


class GenerateResponse(BaseModel):
    response_text: str
    personality_consistency_score: Optional[float] = None
    mood_state: dict = {}
    guardrail_checks: GuardrailChecks = GuardrailChecks()
    tokens_used: dict = {}
    metadata: dict = {}
