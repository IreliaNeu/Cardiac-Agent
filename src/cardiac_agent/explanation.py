import json
import os
import time

import httpx

from .schemas import Claim, Explanation, MedicalKnowledgePacket


def evidence_claims(packet: MedicalKnowledgePacket) -> list[Claim]:
    m = packet.measurements
    facts = {
        "area_ed_px": str(m.area_ed_px), "area_es_px": str(m.area_es_px),
        "fac_percent": f"{m.fac_percent:.4f}", "area_ratio": f"{m.area_ratio:.6f}",
        "rwma_status": packet.rwma.status,
        "deformation_method": m.deformation_method,
    }
    if m.deformation_mean is not None and m.deformation_var is not None:
        facts.update(deformation_mean=f"{m.deformation_mean:.6f}",
                     deformation_var=f"{m.deformation_var:.6f}",
                     deformation_unit=m.deformation_unit or "unspecified")
    if packet.rwma.status == "predicted":
        facts.update(rwma_label=str(packet.rwma.label),
                     rwma_probability=f"{packet.rwma.probability:.6f}")
    return [Claim(evidence_id=k, value=v) for k, v in facts.items()]


def realize(claims: list[Claim]) -> str:
    templates = {
        "area_ed_px": "ED cavity area: {} pixels.",
        "area_es_px": "ES cavity area: {} pixels.",
        "fac_percent": "Two-frame cavity area reduction: {}% (not LVEF).",
        "area_ratio": "ES/ED cavity area ratio: {}.",
        "rwma_status": "RWMA prediction status: {}.",
        "deformation_method": "Difference/motion method: {}.",
        "deformation_mean": "Mean map magnitude within the ED cavity: {}.",
        "deformation_var": "Map magnitude variance within the ED cavity: {} (squared units).",
        "deformation_unit": "Map magnitude unit: {}.",
        "rwma_label": "Model RWMA class (0=normal, 1=RWMA): {}.",
        "rwma_probability": "Model probability of RWMA: {}.",
    }
    return " ".join(templates[c.evidence_id].format(c.value) for c in claims)


class QwenClient:
    def __init__(self, transport=None):
        self.model = os.getenv("CARDIAC_QWEN_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.base_url = os.getenv("CARDIAC_BASE_URL", "https://api.siliconflow.cn/v1")
        self.key = os.getenv(os.getenv("CARDIAC_API_KEY_ENV", "SILICONFLOW_API_KEY"), "")
        self.transport = transport
        self.telemetry = {}

    def organize(self, claims: list[Claim]) -> str:
        return self.complete(
            "Organize these verified cardiac findings into a logical reporting order. "
            "Return ONLY an ordered JSON object mapping every evidence_id to its value. "
            "Preserve all keys and value strings exactly. Do not add diagnoses, "
            "measurements, commentary, markdown, or other keys.",
            [c.model_dump() for c in claims], max_tokens=600)

    def complete(self, system: str, context, max_tokens: int = 1800) -> str:
        self.telemetry = {}
        if not self.key:
            raise RuntimeError("missing_api_key")
        started = time.monotonic()
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        with httpx.Client(timeout=float(os.getenv("CARDIAC_TIMEOUT", "60")),
                          transport=self.transport, trust_env=False) as client:
            for attempt in range(3):
                self.telemetry["attempts"] = attempt + 1
                try:
                    response = client.post(self.base_url.rstrip("/") + "/chat/completions",
                                           headers={"Authorization": "Bearer " + self.key},
                                           json=payload)
                except httpx.TransportError:
                    self.telemetry["latency_seconds"] = time.monotonic() - started
                    if attempt == 2:
                        raise RuntimeError("transport_error") from None
                    time.sleep(2 ** attempt)
                    continue
                self.telemetry.update(status_code=response.status_code,
                                      latency_seconds=time.monotonic() - started)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"http_{response.status_code}")
                try:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError
                    usage = data.get("usage") or {}
                    if isinstance(usage, dict):
                        self.telemetry["usage"] = {k: usage[k] for k in
                            ("prompt_tokens", "completion_tokens", "total_tokens")
                            if isinstance(usage.get(k), int) and usage[k] >= 0}
                    return content
                except (ValueError, KeyError, IndexError, TypeError):
                    raise RuntimeError("invalid_completion") from None
        raise RuntimeError("exhausted_retries")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def explain(packet: MedicalKnowledgePacket, client: QwenClient | None = None) -> Explanation:
    claims = evidence_claims(packet)
    mode, status = "template", "not_requested"
    if client:
        try:
            response = json.loads(client.organize(claims), object_pairs_hook=unique_object)
            expected = {c.evidence_id: c.value for c in claims}
            if isinstance(response, dict) and set(response) == set(expected):
                proposed = [Claim(evidence_id=k, value=v) for k, v in response.items()]
            elif isinstance(response, dict) and set(response) == {"claims"}:
                proposed = [Claim.model_validate(c) for c in response["claims"]]
            else:
                raise ValueError("invalid_contract")
            if len(proposed) != len(claims) or {c.evidence_id: c.value for c in proposed} != expected:
                raise ValueError("unverified_claims")
            claims, mode, status = proposed, "qwen", "success"
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            mode = "template_fallback"
            # Never persist provider error bodies, credentials, or arbitrary response text.
            status = str(exc) if type(exc) is RuntimeError else "invalid_claim_contract"
    text = realize(claims) + " Full-cycle regional wall motion was not assessed."
    if packet.synthetic:
        text = "Synthetic pipeline fixture. " + text
    if packet.measurements.qc_flags:
        text += " QC: " + "; ".join(packet.measurements.qc_flags) + "."
    return Explanation(mode=mode, text=text, claims=claims,
                       verified=True, api_status=status)
