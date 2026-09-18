"""Inspect structured output using synthetic facts only; never print credentials."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from cardiac_agent.explanation import QwenClient, evidence_claims
from cardiac_agent.schemas import MedicalKnowledgePacket

load_dotenv(sys.argv[1], override=False)
packet = MedicalKnowledgePacket.model_validate_json(Path(sys.argv[2]).read_text())
if not packet.synthetic:
    raise ValueError("This probe accepts synthetic fixtures only")
client = QwenClient()
print(json.dumps({"synthetic_response": client.organize(evidence_claims(packet)),
                  "telemetry": client.telemetry}, ensure_ascii=False))
