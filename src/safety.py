"""Hybrid rule-based (Stage 1) and LLM-classified (Stage 2) safety layer (FR3, FR4, ADR-001)."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
import time
from typing import Optional
import uuid
from src.classifier import (
    BaseRiskClassifier,
    ClassifierError,
    LABEL_CLINICAL,
    LABEL_CRISIS,
    LABEL_NONE,
    MockRiskClassifier,
    VALID_RISK_LABELS,
    get_risk_classifier,
)
from src.config import Config

logger = logging.getLogger(__name__)

# Categories matching docs/requirements.md (FR3, FR4, FR5, FR6)
CATEGORY_STANDARD = "STANDARD_CHECKIN"
CATEGORY_CRISIS = "CRISIS_ESCALATION"
CATEGORY_OUT_OF_SCOPE = "OUT_OF_SCOPE_CLINICAL"
CATEGORY_SYSTEM_ERROR = "SYSTEM_ERROR"

# Rule identifiers matching docs/requirements.md (AC-FR3.4, AC-FR3.7, AC-FR4.4) and ADR-001
RULE_NONE = "NONE"
RULE_CRISIS_HARM = "RULE_CRISIS_HARM"
RULE_CLINICAL_ADVICE = "RULE_CLINICAL_ADVICE"
RULE_CLASSIFIER_CRISIS = "CLASSIFIER_CRISIS"
RULE_CLASSIFIER_CLINICAL = "CLASSIFIER_CLINICAL"
RULE_CLASSIFIER_FAIL_SAFE = "CLASSIFIER_FAIL_SAFE"

# Verified Portuguese crisis resources (verified by Rafael Soares / RCavemS on 15-11-2026)
# Issue #4 comment: https://github.com/RCavemS/ai-wellbeing-qa/issues/4#issuecomment-6901923797
CRISIS_RESOURCES_VERIFIED_BY = "Rafael Soares (RCavemS)"
CRISIS_RESOURCES_VERIFIED_DATE = "15-11-2026"

CRISIS_RESOURCES = {
    "112": {
        "name": "112",
        "number": "112",
        "hours": "24/7",
        "description": "emergência",
        "source_url": "https://www.inem.pt/",
    },
    "SNS 24": {
        "name": "SNS 24",
        "number": "808 24 24 24",
        "hours": "24/7",
        "source_url": "https://www.sns24.gov.pt/pt/inicio",
    },
    "SOS Voz Amiga": {
        "name": "SOS Voz Amiga",
        "number": "213 544 545",
        "hours": "15h30 às 00h30",
        "source_url": "https://www.sosvozamiga.org/",
    },
}

_CRISIS_HELPLINES_BLOCK = (
    f"- 112: emergência, {CRISIS_RESOURCES['112']['hours']}\n"
    f"- SNS 24: {CRISIS_RESOURCES['SNS 24']['number']}, {CRISIS_RESOURCES['SNS 24']['hours']}\n"
    f"- SOS Voz Amiga: {CRISIS_RESOURCES['SOS Voz Amiga']['number']}, horário: {CRISIS_RESOURCES['SOS Voz Amiga']['hours']}"
)

# Fixed crisis response derived from CRISIS_RESOURCES (AC-FR3.2, AC-FR3.3: no coaching, no follow-up questions)
CRISIS_RESPONSE_MESSAGE = (
    "Lamento muito que estejas a passar por um momento tão difícil. "
    "A tua segurança é prioritária e não estás sozinho. "
    "Por favor, contacta imediatamente um profissional de saúde ou um dos serviços de apoio verificados em Portugal:\n"
    f"{_CRISIS_HELPLINES_BLOCK}"
)

# Fixed safe fallback message on Stage 2 classifier failure (AC-FR3.7, AC-FR6.5)
SAFE_FALLBACK_MESSAGE = (
    "De momento não é possível processar a tua mensagem devido a uma falha técnica. "
    "Se precisares de apoio imediato ou estiveres em situação de crise, por favor contacta "
    "um profissional de saúde ou um dos serviços de apoio verificados em Portugal:\n"
    f"{_CRISIS_HELPLINES_BLOCK}"
)

# Fixed out-of-scope clinical referral response (AC-FR4.1, AC-FR4.2, AC-FR4.3)
OUT_OF_SCOPE_REFERRAL_MESSAGE = (
    "Compreendo a tua preocupação, mas enquanto assistente de check-in de bem-estar não sou um profissional "
    "de saúde e não posso realizar diagnósticos médicos, fornecer orientação sobre medicação ou recomendar tratamentos clínicos. "
    "Para avaliares esta questão com segurança, recomendo que consultes o teu médico prescritor, farmacêutico, "
    "psiquiatra ou psicólogo."
)

# Shared regex fragments to avoid duplication
_DRUG_NAMES = (
    r"(?:sertralina|sertraline|fluoxetina|fluoxetine|escitalopram|paroxetina|paroxetine|venlafaxina|venlafaxine|"
    r"xanax|alprazolam|diazepam|lorazepam|rivotril|clonazepam|zolpidem|ritalina|ritalin|concerta|"
    r"antidepressivos?|antidepressants?|ansiol[íi]ticos?|anxiolytics?|psicof[áa]rmacos?)"
)
_MED_NOUNS = (
    r"(?:medica[çc][ãa]o|medicamentos?|comprimidos?|rem[ée]dios?|medication|meds|medicine|pills|dosage|dose|"
    r"prescri[çc][ãa]o|receita\s+m[ée]dica|prescription)"
)
_DISORDER_NAMES = (
    r"(?:depress[ãa]o(?:\s+cl[íi]nica)?|(?:clinical\s+)?depression|clinically\s+depressed|tdah|adhd|"
    r"perturba[çc][ãa]o\s+bipolar|bipolar\s+disorder|esquizofrenia|schizophrenia|"
    r"perturba[çc][ãa]o\s+de\s+p[âa]nico|panic\s+disorder|ansiedade\s+generalizada|"
    r"generalized\s+anxiety(?:\s+disorder)?|ptsd|stress\s+p[óo]s-traum[áa]tico|ocd|poc|burnout\s+cl[íi]nico|clinical\s+burnout)"
)

# Benign everyday idioms that must NOT trigger crisis escalation (AC-FR3.5)
_BENIGN_IDIOM_PATTERNS = [
    re.compile(
        r"\ba?\s*morrer\s+de\s+(sono|fome|rir|sede|frio|calor|vergonha|amores|saudades|medo|cansa[çc]o|t[ée]dio)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmorri\s+de\s+(rir|sono|fome|vergonha|sede|medo)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmatar\s+(saudades|o\s+tempo|a\s+fome|a\s+sede)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(este|esse|o)\s+(calor|frio|exame|teste|trabalho|treino|tr[âa]nsito)\s+(est[áa]\s+a\s+matar-me|mata-me)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(exam|test|homework|work|workout|exercise|heat|cold|traffic|shoes|feet|back|headache|waiting|suspense|joke)\s+"
        r"(is|are|was|were|almost)\s+(killing|killed)\s+me\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(dying|died)\s+(of\s+(laughter|boredom|hunger|thirst|embarrassment)|for\s+a\s+\w+|to\s+(see|know|try|meet|hear|go|eat))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bkilling\s+time\b", re.IGNORECASE),
]

# Crisis detection patterns: direct, indirect, typos, PT-PT, English, and mixed language (FR3)
_CRISIS_PATTERNS = [
    # Direct & typo self-harm / suicide (PT-PT)
    re.compile(
        r"\b(quero|kero|qero|vou|penso\s+em|pensar\s+em|vontade\s+de)\s+"
        r"(morrer|morer|suicidar|matar-me|matarme|me\s+matar|desaparecer(\s+para\s+sempre)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(suic[ií]dio|suicido|suicidal|suicide|suicde|auto-?mutila[çc][ãa]o|self-?harm)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(matar-me|matarme|suicidar-me|suicidarme|cortar-me|cortarme|cortar\s+os\s+pulsos|fazer-me\s+mal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfazer\s+mal\s+a\s+mim(\s+pr[óo]prio|\s+mesmo)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(acabar|por\s+fim|p[ôo]r\s+fim|tirar)\s+(com\s+)?(a\s+minha\s+vida|a\s+pr[óo]pria\s+vida|tudo)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bacabar\s+c[o]?m\s+tudo\b", re.IGNORECASE),
    re.compile(
        r"\b(tomar|engolir)\s+(todos\s+os|uma\s+caixa\s+inteira\s+de)\s+(meus\s+)?(comprimidos|medicamentos|antidepressivos)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\boverdose\b", re.IGNORECASE),
    # Direct & typo self-harm / suicide (English)
    re.compile(
        r"\b(want\s+to|wanna|going\s+to|gonna|thinking\s+about|plan\s+to)\s+"
        r"(die|dye|kill\s+myself|kil\s+myself|end\s+my\s+life|take\s+my\s+(own\s+)?life|end\s+it\s+all|hurt\s+myself|harm\s+myself|cut\s+myself|disappear\s+forever)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(kill|kil|killing|hurt|hurting|harm|harming|cut|cutting)\s+myself\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btake\s+my\s+(own\s+)?life\b", re.IGNORECASE),
    re.compile(r"\bend\s+(my\s+life|it\s+all)\b", re.IGNORECASE),
    # Indirect crisis phrasing (PT-PT & EN)
    re.compile(r"\bn[ãa]o\s+aguento\s+mais\s+(viver|estar\s+viv[oa])\b", re.IGNORECASE),
    re.compile(
        r"\b(sou\s+um\s+fardo|sinto\s+que\s+sou\s+um\s+fardo|estariam\s+melhor\s+sem\s+mim|estavam\s+melhor\s+sem\s+mim)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bn[ãa]o\s+quero\s+(acordar|viver|estar\s+aqui|continuar\s+vivo)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bn[ãa]o\s+vale\s+a\s+pena\s+(viver|continuar)\b", re.IGNORECASE),
    re.compile(r"\b(don'?t|do\s+not)\s+want\s+to\s+(wake\s+up|be\s+alive|live\s+anymore|be\s+here\s+anymore)\b", re.IGNORECASE),
    re.compile(
        r"\b(better\s+off\s+without\s+me|i\s+am\s+a\s+burden|i'?m\s+(just\s+)?a\s+burden)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(no\s+reason\s+to\s+live|not\s+worth\s+living|can'?t\s+go\s+on\s+living|cannot\s+go\s+on\s+living|can'?t\s+take\s+this\s+anymore)\b",
        re.IGNORECASE,
    ),
    # Harm to others
    re.compile(
        r"\b(quero|vou|estou\s+a\s+pensar\s+em|vontade\s+de)\s+(matar|fazer\s+mal\s+a|magoar|ferir|agredir|atacar)\s+(algu[ée]m|outra\s+pessoa|pessoas|outras\s+pessoas|uma\s+pessoa)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(want\s+to|going\s+to|gonna|thinking\s+about|will)\s+(kill|killing|hurt|hurting|harm|harming|attack|attacking)\s+(someone|somebody|another\s+person|other\s+people|people)\b",
        re.IGNORECASE,
    ),
]

# Out-of-scope clinical advice, diagnosis, and medication inquiry patterns (FR4)
_OUT_OF_SCOPE_INPUT_PATTERNS = [
    # Explicit diagnosis requests (PT-PT & EN)
    re.compile(
        r"\b(diagnosticar(-me)?|diagn[óo]stico|diagnose(\s+me)?|diagnosis)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(acho\s+que\s+tenho|ser[áa]\s+que\s+tenho|do\s+you\s+think\s+i\s+have|do\s+i\s+have|am\s+i|is\s+this)\s+{_DISORDER_NAMES}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(tenho|sofro\s+de)\s+{_DISORDER_NAMES}.*(\?|\bou\s+[ée]\b|\bo\s+que\s+achas\b)",
        re.IGNORECASE,
    ),
    # Medication & dosage advice inquiries (PT-PT & EN)
    re.compile(
        rf"\b(devo|posso|que|qual|como|should\s+i|can\s+i|what|which)\b.*?\b({_DRUG_NAMES}|{_MED_NOUNS})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b({_DRUG_NAMES}|{_MED_NOUNS})\b.*?\b(devo|posso|tomar|parar|recomendas|aconselhas|should\s+i|can\s+i|take|stop|recommend|advise)\b",
        re.IGNORECASE,
    ),
    # Clinical treatment requests (PT-PT & EN)
    re.compile(
        r"\b(tratamento\s+cl[íi]nico|plano\s+de\s+tratamento|clinical\s+treatment|treatment\s+plan|medical\s+therapy|terapia\s+m[ée]dica|curar\s+a\s+minha\s+fobia|cure\s+my\s+disorder)\b",
        re.IGNORECASE,
    ),
]

# Output screening patterns to block unsafe LLM replies giving diagnosis, medication, or treatment advice (AC-FR4.5)
_UNSAFE_OUTPUT_PATTERNS = [
    re.compile(rf"\b{_DRUG_NAMES}\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(mg|mcg|ml|comprimidos?|pills?|tablets?)\b", re.IGNORECASE),
    re.compile(
        rf"\b(o\s+meu\s+diagn[óo]stico\s+[ée]|i\s+diagnose\s+you|tens\s+{_DISORDER_NAMES}|you\s+have\s+{_DISORDER_NAMES})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(prescrevo|i\s+prescribe|deves\s+iniciar\s+tratamento|(recomendo\s+que\s+tomes|podes\s+parar\s+de\s+tomar|you\s+should\s+(take|stop\s+taking))\s+.*?\b({_DRUG_NAMES}|{_MED_NOUNS}))\b",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class SafetyResult:
    """Result of screening a user input or LLM output through the safety layer."""

    triggered: bool
    category: str
    rule_triggered: str
    response: Optional[str] = None
    status: str = "OK"
    label: Optional[str] = LABEL_NONE
    stage_decided: str = "STAGE_1"
    latency_ms: float = 0.0
    error_type: Optional[str] = None


_SAFE_RESULT = SafetyResult(
    triggered=False,
    category=CATEGORY_STANDARD,
    rule_triggered=RULE_NONE,
    response=None,
)


class SafetyLayer:
    """Hybrid rule-based (Stage 1) and LLM-classified (Stage 2) safety interceptor (FR3, FR4, ADR-001)."""

    def __init__(
        self,
        classifier: Optional[BaseRiskClassifier] = None,
        config: Optional[Config] = None,
    ):
        if classifier is not None:
            self.classifier = classifier
        elif config is not None:
            self.classifier = get_risk_classifier(config)
        else:
            self.classifier = MockRiskClassifier()

    def _strip_benign_idioms(self, text: str) -> str:
        cleaned = text
        for pattern in _BENIGN_IDIOM_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        return cleaned

    def _emit_decision_log(self, result: SafetyResult) -> SafetyResult:
        """Log only the label, the stage that decided, latency, and error type (never raw message text)."""
        log_fn = logger.error if result.status == "ERROR" else logger.info
        log_fn(
            "safety_decision label=%s stage=%s latency_ms=%.3f error_type=%s",
            result.label,
            result.stage_decided,
            result.latency_ms,
            result.error_type or "NONE",
        )
        return result

    def screen_input(self, message: str) -> SafetyResult:
        """Screen incoming user message via Stage 1 rules and Stage 2 risk classifier (ADR-001)."""
        start_time = time.perf_counter()
        cleaned_for_crisis = self._strip_benign_idioms(message)

        # 1. Stage 1 deterministic rules run first
        for pattern in _CRISIS_PATTERNS:
            if pattern.search(cleaned_for_crisis):
                # 2. If Stage 1 returns CRISIS, skip Stage 2
                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
                return self._emit_decision_log(
                    SafetyResult(
                        triggered=True,
                        category=CATEGORY_CRISIS,
                        rule_triggered=RULE_CRISIS_HARM,
                        response=CRISIS_RESPONSE_MESSAGE,
                        status="OK",
                        label=LABEL_CRISIS,
                        stage_decided="STAGE_1",
                        latency_ms=latency_ms,
                        error_type=None,
                    )
                )

        stage1_clinical = any(pattern.search(message) for pattern in _OUT_OF_SCOPE_INPUT_PATTERNS)

        # 3. Otherwise call Stage 2, including when Stage 1 returned CLINICAL
        try:
            stage2_label = self.classifier.classify(message)
            if not isinstance(stage2_label, str) or stage2_label not in VALID_RISK_LABELS:
                raise ClassifierError(
                    "Classifier returned label outside allowed enum.",
                    error_type="InvalidLabelError",
                )
        except Exception as exc:
            error_type = exc.error_type if isinstance(exc, ClassifierError) else type(exc).__name__
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            if stage1_clinical:
                return self._emit_decision_log(
                    SafetyResult(
                        triggered=True,
                        category=CATEGORY_OUT_OF_SCOPE,
                        rule_triggered=RULE_CLINICAL_ADVICE,
                        response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                        status="OK",
                        label=LABEL_CLINICAL,
                        stage_decided="STAGE_1",
                        latency_ms=latency_ms,
                        error_type=error_type,
                    )
                )
            return self._emit_decision_log(
                SafetyResult(
                    triggered=True,
                    category=CATEGORY_SYSTEM_ERROR,
                    rule_triggered=RULE_CLASSIFIER_FAIL_SAFE,
                    response=SAFE_FALLBACK_MESSAGE,
                    status="ERROR",
                    label=None,
                    stage_decided="STAGE_2",
                    latency_ms=latency_ms,
                    error_type=error_type,
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        # 4. Final category = the more severe of the two labels: CRISIS > CLINICAL > NONE
        if stage2_label == LABEL_CRISIS:
            return self._emit_decision_log(
                SafetyResult(
                    triggered=True,
                    category=CATEGORY_CRISIS,
                    rule_triggered=RULE_CLASSIFIER_CRISIS,
                    response=CRISIS_RESPONSE_MESSAGE,
                    status="OK",
                    label=LABEL_CRISIS,
                    stage_decided="STAGE_2",
                    latency_ms=latency_ms,
                    error_type=None,
                )
            )

        if stage1_clinical:
            return self._emit_decision_log(
                SafetyResult(
                    triggered=True,
                    category=CATEGORY_OUT_OF_SCOPE,
                    rule_triggered=RULE_CLINICAL_ADVICE,
                    response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                    status="OK",
                    label=LABEL_CLINICAL,
                    stage_decided="STAGE_1",
                    latency_ms=latency_ms,
                    error_type=None,
                )
            )

        if stage2_label == LABEL_CLINICAL:
            return self._emit_decision_log(
                SafetyResult(
                    triggered=True,
                    category=CATEGORY_OUT_OF_SCOPE,
                    rule_triggered=RULE_CLASSIFIER_CLINICAL,
                    response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                    status="OK",
                    label=LABEL_CLINICAL,
                    stage_decided="STAGE_2",
                    latency_ms=latency_ms,
                    error_type=None,
                )
            )

        return self._emit_decision_log(
            SafetyResult(
                triggered=False,
                category=CATEGORY_STANDARD,
                rule_triggered=RULE_NONE,
                response=None,
                status="OK",
                label=LABEL_NONE,
                stage_decided="STAGE_2",
                latency_ms=latency_ms,
                error_type=None,
            )
        )

    def screen_output(self, reply: str) -> SafetyResult:
        """Screen LLM output before returning to user; replace unsafe clinical advice with FR4 referral (AC-FR4.5)."""
        if not isinstance(reply, str):
            reply = str(reply)
        stripped = reply.strip()
        if stripped in (
            OUT_OF_SCOPE_REFERRAL_MESSAGE.strip(),
            CRISIS_RESPONSE_MESSAGE.strip(),
            SAFE_FALLBACK_MESSAGE.strip(),
        ):
            return _SAFE_RESULT

        for pattern in _UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(reply):
                return SafetyResult(
                    triggered=True,
                    category=CATEGORY_OUT_OF_SCOPE,
                    rule_triggered=RULE_CLINICAL_ADVICE,
                    response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                )

        return _SAFE_RESULT


class AuditLogger:
    """Structured audit logger for safety triggers (FR5).

    Enforces AC-FR5.1 schema and AC-FR5.2 privacy rule (raw message text is never stored).
    """

    def __init__(self):
        self.entries: list[dict] = []

    def log(
        self,
        *,
        status: str,
        category: str,
        rule_triggered: str,
        model_version: str,
        latency_ms: float,
    ) -> dict:
        entry = {
            "interaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "category": category,
            "rule_triggered": rule_triggered,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 3),
        }
        self.entries.append(entry)
        return entry
