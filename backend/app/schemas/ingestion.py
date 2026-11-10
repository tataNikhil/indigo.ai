"""Pydantic schemas: the 'firewall' that every request and LLM reply must pass."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

Intent = Literal[
    "greeting",
    "order_tracking",
    "payment_issue",
    "refund_request",
    "account_access",
    "fraud_report",
    "loan_or_credit",
    "general_question",
    "unknown",
]
RiskLevel = Literal["low", "medium", "high", "critical"]
RiskFlag = Literal[
    "pii_present",
    "financial_loss",
    "fraud_suspected",
    "prompt_injection",
    "abusive_language",
    "legal_threat",
]
PriorityLevel = Literal["low", "medium", "high", "urgent"]

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

SLA_HOURS = {"urgent": 1, "high": 4, "medium": 24, "low": 72}


class IngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(
        min_length=1,
        max_length=8000,
        description="Raw customer support / financial text.",
        examples=["Rs 4,999 was debited twice from my account. Please refund urgently!"],
    )
    channel: Literal["email", "chat", "phone_transcript", "web_form"] = "chat"


class BatchIngestionRequest(RootModel[list[IngestionRequest]]):
    @field_validator("root")
    @classmethod
    def batch_size(cls, items: list[IngestionRequest]) -> list[IngestionRequest]:
        if not 1 <= len(items) <= 10:
            raise ValueError("A batch must contain between 1 and 10 messages.")
        return items


class RiskProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: RiskLevel
    flags: list[RiskFlag] = Field(default_factory=list)
    reason: str = Field(min_length=3, max_length=300)

    @field_validator("flags")
    @classmethod
    def unique_flags(cls, flags: list[str]) -> list[str]:
        return sorted(set(flags))

    @model_validator(mode="after")
    def level_matches_flags(self) -> "RiskProfile":
        if "fraud_suspected" in self.flags and RISK_ORDER[self.level] < RISK_ORDER["high"]:
            raise ValueError("risk level must be 'high' or 'critical' when fraud is suspected")
        return self


class LLMPriority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: PriorityLevel
    requires_human: bool = Field(description="True if an agent must review this.")


class LLMClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    risk: RiskProfile
    priority: LLMPriority
    summary: str = Field(min_length=5, max_length=300)

    @field_validator("summary")
    @classmethod
    def single_line_summary(cls, value: str) -> str:
        return " ".join(value.split())


class PriorityParameters(LLMPriority):
    sla_hours: int = Field(ge=1)


class TokenMetadata(BaseModel):
    estimated_input_tokens: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    model: str
    latency_ms: int = Field(ge=0)


class IngestionResult(BaseModel):
    intent: Intent
    confidence: float
    risk: RiskProfile
    priority: PriorityParameters
    summary: str
    token_metadata: TokenMetadata


class BatchIngestionResult(RootModel[list[IngestionResult]]):
    pass
