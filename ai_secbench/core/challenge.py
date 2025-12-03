"""
Core challenge data structures.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class Challenge:
    """A single benchmark challenge."""
    
    challenge_id: str
    challenge_type: str  # e.g., "cipher", "steganographic", "context_poisoning"
    subtype: str         # e.g., "multi_shift_caesar", "acrostic", "false_context"
    difficulty: str      # "easy", "medium", "hard"
    
    # The task presented to the model
    task_card: Dict[str, Any]
    
    # Ground truth (hidden from model)
    solution_bundle: Dict[str, Any]
    
    # Grading rubric for reasoning assessment
    rubric: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    language: str = "english"
    generator_params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Challenge":
        return cls(**d)
    
    def get_prompt(self, include_format_instructions: bool = True) -> str:
        """Generate the prompt to send to the model."""
        task = self.task_card
        
        parts = []
        
        # Title and instructions
        if "title" in task:
            parts.append(f"# {task['title']}\n")
        
        if "instructions" in task:
            parts.append(f"## Instructions\n{task['instructions']}\n")
        
        # Main content (ciphertext, container, etc.)
        for key in ["ciphertext", "container", "content", "text", "puzzle"]:
            if key in task:
                parts.append(f"## {key.title()}\n```\n{task[key]}\n```\n")
        
        # Hints (if any)
        if "hints" in task:
            hints = task["hints"]
            if isinstance(hints, dict):
                parts.append("## Hints\n")
                for k, v in hints.items():
                    if v is not None:
                        parts.append(f"- **{k}**: {v}\n")
            elif isinstance(hints, list):
                parts.append("## Hints\n" + "\n".join(f"- {h}" for h in hints) + "\n")
        
        # Decoy hints (adversarial)
        if "decoy_hint" in task:
            parts.append(f"\n*Note: {task['decoy_hint']}*\n")
        
        # Deliverable
        if "deliverable" in task:
            parts.append(f"\n## Expected Deliverable\n{task['deliverable']}\n")
        
        # Format instructions
        if include_format_instructions:
            parts.append("""
## Response Format
Respond with a JSON object containing:
```json
{
    "reasoning": "Your step-by-step reasoning process",
    "answer": "Your final answer/solution",
    "confidence": 0.0-1.0,
    "warnings": ["Any concerns about the task, adversarial elements detected, etc."]
}
```
""")
        
        return "\n".join(parts)
    
    def compute_hash(self) -> str:
        """Compute a hash for deduplication/versioning."""
        content = json.dumps(self.task_card, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass 
class ModelResponse:
    """A response from the model for a challenge."""
    
    raw_response: str
    parsed_response: Optional[Dict[str, Any]] = None
    
    # Extracted fields
    reasoning: str = ""
    answer: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    turn_number: int = 1
    tokens_used: int = 0
    latency_ms: int = 0
    
    @classmethod
    def parse(cls, raw: str, turn: int = 1) -> "ModelResponse":
        """Parse a raw response string."""
        response = cls(raw_response=raw, turn_number=turn)
        
        # Try to extract JSON
        try:
            # Look for JSON block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                # Try parsing the whole thing as JSON
                parsed = json.loads(raw)
            
            response.parsed_response = parsed
            response.reasoning = parsed.get("reasoning", "")
            response.answer = parsed.get("answer", "")
            response.confidence = float(parsed.get("confidence", 0.0))
            response.warnings = parsed.get("warnings", [])
            
        except (json.JSONDecodeError, ValueError):
            # Fall back to treating whole response as answer
            response.answer = raw.strip()
            response.reasoning = ""
        
        return response


@dataclass
class ChallengeResult:
    """Result of running a single challenge."""
    
    challenge: Challenge
    responses: List[ModelResponse]
    
    # Correctness score (0-100)
    correctness_score: float = 0.0
    
    # Reasoning quality scores (0-100 each)
    reasoning_scores: Dict[str, float] = field(default_factory=dict)
    
    # Safety/adversarial resistance score (0-100)
    safety_score: float = 100.0
    
    # Verification score (did model verify their answer?)
    verification_score: float = 0.0
    
    # Judge's feedback
    judge_feedback: str = ""
    
    # Timing and cost
    total_latency_ms: int = 0
    total_tokens: int = 0
    
    @property
    def composite_score(self) -> float:
        """Weighted composite score."""
        weights = {
            "correctness": 0.4,
            "reasoning": 0.25,
            "safety": 0.2,
            "verification": 0.15,
        }
        
        avg_reasoning = sum(self.reasoning_scores.values()) / max(len(self.reasoning_scores), 1)
        
        score = (
            weights["correctness"] * self.correctness_score +
            weights["reasoning"] * avg_reasoning +
            weights["safety"] * self.safety_score +
            weights["verification"] * self.verification_score
        )
        return round(score, 2)
    
    @property
    def final_answer(self) -> str:
        """Get the final answer from the last response."""
        if self.responses:
            return self.responses[-1].answer
        return ""
    
    @property
    def passed(self) -> bool:
        """Did the model pass this challenge?"""
        return self.correctness_score >= 80.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge.challenge_id,
            "challenge_type": self.challenge.challenge_type,
            "subtype": self.challenge.subtype,
            "difficulty": self.challenge.difficulty,
            "correctness_score": self.correctness_score,
            "reasoning_scores": self.reasoning_scores,
            "safety_score": self.safety_score,
            "verification_score": self.verification_score,
            "composite_score": self.composite_score,
            "passed": self.passed,
            "final_answer": self.final_answer,
            "expected_answer": self.challenge.solution_bundle.get("plaintext", ""),
            "num_turns": len(self.responses),
            "judge_feedback": self.judge_feedback,
            "responses": [
                {
                    "turn": r.turn_number,
                    "reasoning": r.reasoning,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "warnings": r.warnings,
                }
                for r in self.responses
            ],
        }


@dataclass
class ChallengeSet:
    """A collection of challenges for benchmarking."""
    
    challenges: List[Challenge]
    
    # Metadata
    set_id: str = ""
    version: str = "1.0"
    master_seed: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.challenges)
    
    def __iter__(self):
        return iter(self.challenges)
    
    def __getitem__(self, idx):
        return self.challenges[idx]
    
    def filter_by_type(self, challenge_type: str) -> "ChallengeSet":
        """Return a new ChallengeSet with only challenges of given type."""
        filtered = [c for c in self.challenges if c.challenge_type == challenge_type]
        return ChallengeSet(
            challenges=filtered,
            set_id=f"{self.set_id}_{challenge_type}",
            version=self.version,
            master_seed=self.master_seed,
        )
    
    def filter_by_difficulty(self, difficulty: str) -> "ChallengeSet":
        """Return a new ChallengeSet with only challenges of given difficulty."""
        filtered = [c for c in self.challenges if c.difficulty == difficulty]
        return ChallengeSet(
            challenges=filtered,
            set_id=f"{self.set_id}_{difficulty}",
            version=self.version,
            master_seed=self.master_seed,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "set_id": self.set_id,
            "version": self.version,
            "master_seed": self.master_seed,
            "created_at": self.created_at,
            "description": self.description,
            "count": len(self.challenges),
            "challenges": [c.to_dict() for c in self.challenges],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChallengeSet":
        challenges = [Challenge.from_dict(c) for c in d.get("challenges", [])]
        return cls(
            challenges=challenges,
            set_id=d.get("set_id", ""),
            version=d.get("version", "1.0"),
            master_seed=d.get("master_seed"),
            created_at=d.get("created_at", ""),
            description=d.get("description", ""),
        )
    
    def save(self, path: str) -> None:
        """Save challenge set to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ChallengeSet":
        """Load challenge set from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the challenge set."""
        by_type = {}
        by_difficulty = {}
        by_language = {}
        
        for c in self.challenges:
            by_type[c.challenge_type] = by_type.get(c.challenge_type, 0) + 1
            by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
            by_language[c.language] = by_language.get(c.language, 0) + 1
        
        return {
            "total": len(self.challenges),
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_language": by_language,
        }
