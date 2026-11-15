"""Deterministic rule-based safety layer for crisis escalation (FR3) and clinical scope boundaries (FR4)."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Optional
import uuid

# Categories matching docs/requirements.md (FR3, FR4, FR5, FR6)
CATEGORY_STANDARD = "STANDARD_CHECKIN"
CATEGORY_CRISIS = "CRISIS_ESCALATION"
CATEGORY_OUT_OF_SCOPE = "OUT_OF_SCOPE_CLINICAL"

# Rule identifiers
RULE_NONE = "NONE"
RULE_CRISIS_HARM = "RULE_CRISIS_HARM"
RULE_CLINICAL_ADVICE = "RULE_CLINICAL_ADVICE"
RULE_OUTPUT_CLINICAL_ADVICE = "RULE_OUTPUT_CLINICAL_ADVICE"

# Verified Portuguese crisis resources (verified by Rafael Soares / RCavemS on 15-11-2026)
# Issue #4 comment: https://github.com/RCavemS/ai-wellbeing-qa/issues/4#issuecomment-6901923797
CRISIS_RESOURCES_VERIFIED_BY = "Rafael Soares (RCavemS)"
CRISIS_RESOURCES_VERIFIED_DATE = "15-11-2026"

CRISIS_RESOURCES = {
    "112": {
        "name": "112",
        "number": "112",
        "hours": "24/7",
        "description": "Emergency services",
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

# Fixed crisis response (AC-FR3.2, AC-FR3.3: exact numbers, no coaching, no follow-up questions)
CRISIS_RESPONSE_MESSAGE = (
    "Lamento muito que estejas a passar por um momento tão difícil. "
    "A tua segurança é prioritária e não estás sozinho. "
    "Por favor, contacta imediatamente um profissional de saúde ou um dos serviços de apoio verificados em Portugal:\n"
    "- 112: emergência, 24/7\n"
    "- SNS 24: 808 24 24 24, 24/7\n"
    "- SOS Voz Amiga: 213 544 545, horário: 15h30 às 00h30"
)

# Fixed out-of-scope clinical referral response (AC-FR4.1, AC-FR4.2, AC-FR4.3)
OUT_OF_SCOPE_REFERRAL_MESSAGE = (
    "Compreendo a tua preocupação, mas enquanto assistente de check-in de bem-estar não sou um profissional "
    "de saúde e não posso realizar diagnósticos médicos, prescrever medicação ou recomendar tratamentos clínicos. "
    "Para avaliares esta questão com segurança, recomendo que consultes um médico, psiquiatra ou psicólogo, "
    "ou que contactes a linha SNS 24 (808 24 24 24)."
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
    re.compile(r"\bn[ãa]o\s+aguento\s+mais(\s+viver|\s+isto|\s+nada)?\b", re.IGNORECASE),
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
        r"\b(quero|vou)\s+(matar|fazer\s+mal\s+a)\s+(algu[ée]m|outra\s+pessoa|pessoas)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(want\s+to|going\s+to)\s+(kill|hurt|harm)\s+(someone|other\s+people)\b",
        re.IGNORECASE,
    ),
]

# Out-of-scope clinical advice, diagnosis, and medication patterns for input screening (FR4)
_OUT_OF_SCOPE_INPUT_PATTERNS = [
    # Diagnosis requests (PT-PT & EN)
    re.compile(
        r"\b(diagnosticar(-me)?|diagn[óo]stico|diagnose(\s+me)?|diagnosis)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(acho\s+que\s+tenho|ser[áa]\s+que\s+tenho|tenho|sofro\s+de|do\s+you\s+think\s+i\s+have|do\s+i\s+have|am\s+i\s+suffering\s+from|am\s+i|is\s+this)\s+"
        r"(depress[ãa]o(\s+cl[íi]nica)?|(clinical\s+)?depression|clinically\s+depressed|tdah|adhd|perturba[çc][ãa]o\s+bipolar|bipolar\s+disorder|esquizofrenia|schizophrenia|"
        r"perturba[çc][ãa]o\s+de\s+p[âa]nico|panic\s+disorder|ansiedade\s+generalizada|generalized\s+anxiety|ptsd|stress\s+p[óo]s-traum[áa]tico|ocd|poc)\b",
        re.IGNORECASE,
    ),
    # Specific psychiatric medications & pharmaceutical queries (PT-PT & EN)
    re.compile(
        r"\b(sertralina|sertraline|fluoxetina|fluoxetine|escitalopram|paroxetina|paroxetine|venlafaxina|venlafaxine|"
        r"xanax|alprazolam|diazepam|lorazepam|rivotril|clonazepam|zolpidem|ritalina|ritalin|concerta|"
        r"antidepressivos?|antidepressants?|ansiol[íi]ticos?|anxiolytics?|psicof[áa]rmacos?)\b",
        re.IGNORECASE,
    ),
    # Medication / dosage advice requests (PT-PT & EN)
    re.compile(
        r"\b(medica[çc][ãa]o|medicamentos?|comprimidos?|rem[ée]dios?|medication|meds|medicine|pills|dosage|dose|prescri[çc][ãa]o|receita\s+m[ée]dica|prescription)\b"
        r".*?\b(devo|posso|tomar|parar|recomendas|aconselhas|should\s+i|can\s+i|take|stop|recommend|advise)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(devo|posso|que|qual|should\s+i|can\s+i|what|which)\b"
        r".*?\b(medica[çc][ãa]o|medicamentos?|comprimidos?|rem[ée]dios?|medication|meds|medicine|pills|dosage|dose|prescri[çc][ãa]o|receita\s+m[ée]dica|prescription)\b",
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
    re.compile(
        r"\b(sertralina|sertraline|fluoxetina|fluoxetine|escitalopram|paroxetina|paroxetine|venlafaxina|venlafaxine|"
        r"xanax|alprazolam|diazepam|lorazepam|rivotril|clonazepam|zolpidem|ritalina|ritalin|concerta|"
        r"antidepressivos?|antidepressants?|ansiol[íi]ticos?|anxiolytics?|psicof[áa]rmacos?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+\s*(mg|mcg|ml|comprimidos?|pills?|tablets?)\b", re.IGNORECASE),
    re.compile(
        r"\b(o\s+meu\s+diagn[óo]stico\s+[ée]|i\s+diagnose\s+you|tens\s+(depress[ãa]o(\s+cl[íi]nica)?|perturba[çc][ãa]o|tdah)|"
        r"you\s+have\s+(generalized\s+anxiety\s+disorder|clinical\s+depression|bipolar\s+disorder|adhd|panic\s+disorder))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(prescrevo|i\s+prescribe|deves\s+iniciar\s+tratamento|recomendo\s+que\s+tomes|podes\s+parar\s+de\s+tomar|you\s+should\s+(take|stop\s+taking))\b",
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


class SafetyLayer:
    """Deterministic rule-based interceptor for FR3 (Crisis) and FR4 (Out-of-Scope Clinical)."""

    def _strip_benign_idioms(self, text: str) -> str:
        cleaned = text
        for pattern in _BENIGN_IDIOM_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        return cleaned

    def screen_input(self, message: str) -> SafetyResult:
        """Screen incoming user message before any LLM call (FR3 takes precedence over FR4)."""
        cleaned_for_crisis = self._strip_benign_idioms(message)

        for pattern in _CRISIS_PATTERNS:
            if pattern.search(cleaned_for_crisis):
                return SafetyResult(
                    triggered=True,
                    category=CATEGORY_CRISIS,
                    rule_triggered=RULE_CRISIS_HARM,
                    response=CRISIS_RESPONSE_MESSAGE,
                )

        for pattern in _OUT_OF_SCOPE_INPUT_PATTERNS:
            if pattern.search(message):
                return SafetyResult(
                    triggered=True,
                    category=CATEGORY_OUT_OF_SCOPE,
                    rule_triggered=RULE_CLINICAL_ADVICE,
                    response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                )

        return SafetyResult(
            triggered=False,
            category=CATEGORY_STANDARD,
            rule_triggered=RULE_NONE,
            response=None,
        )

    def screen_output(self, reply: str) -> SafetyResult:
        """Screen LLM output before returning to user; replace unsafe clinical advice with FR4 referral (AC-FR4.5)."""
        stripped = reply.strip()
        if stripped in (OUT_OF_SCOPE_REFERRAL_MESSAGE.strip(), CRISIS_RESPONSE_MESSAGE.strip()):
            return SafetyResult(
                triggered=False,
                category=CATEGORY_STANDARD,
                rule_triggered=RULE_NONE,
                response=None,
            )

        for pattern in _UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(reply):
                return SafetyResult(
                    triggered=True,
                    category=CATEGORY_OUT_OF_SCOPE,
                    rule_triggered=RULE_OUTPUT_CLINICAL_ADVICE,
                    response=OUT_OF_SCOPE_REFERRAL_MESSAGE,
                )

        return SafetyResult(
            triggered=False,
            category=CATEGORY_STANDARD,
            rule_triggered=RULE_NONE,
            response=None,
        )


class AuditLogger:
    """Structured audit logger for assistant interactions and safety triggers (FR5).

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

