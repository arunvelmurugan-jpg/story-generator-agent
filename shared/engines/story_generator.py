"""Config-driven Story Generator Engine.

Ports the original /run endpoint logic from main.py exactly:
  - Refine mode: single LLM call to refine one story
  - Generate mode: parallel per-epic LLM calls via asyncio.gather
    + US governance scoring call
  All original helper functions (_build_prd_context_block,
  _build_epic_prompt, _generate_for_epic, _run_us_governance_sync)
  are preserved verbatim. Only their import mechanism changes — we
  load llm_client.py, prompt.py, us_governance_prompt.py and
  domain_rules.py via importlib so the original files remain untouched.

Business logic is 100% preserved from the original agent:
  - Accepts epics, capabilities, domain, refine, story_to_refine,
    user_instruction, grounded, fe_only, prd_sections
  - Returns stories, us_governance, input_tokens, output_tokens
"""

import asyncio
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Locate agent root (phtnai-agentic-ai/) ───────────────────────────────────
# This file lives at shared/engines/story_generator.py — three levels up.
_AGENT_ROOT = Path(__file__).parent.parent.parent

MAX_CONCURRENT = 5  # overridden from env in __init__


# ── importlib loader helper ───────────────────────────────────────────────────
def _load_module(name: str, file: Path):
    """Load a module from an absolute path. Returns the module object."""
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Load prompt.py (STORY_GENERATOR_PROMPT) ───────────────────────────────────
_STORY_GENERATOR_PROMPT: str = ""
try:
    _pm = _load_module("_sg_prompt", _AGENT_ROOT / "prompt.py")
    _STORY_GENERATOR_PROMPT = getattr(_pm, "STORY_GENERATOR_PROMPT", "")
    logger.info("[StoryGeneratorEngine] Loaded STORY_GENERATOR_PROMPT from prompt.py")
except Exception as e:
    logger.warning(f"[StoryGeneratorEngine] Could not load prompt.py: {e}")

# ── Load us_governance_prompt.py (US_GOVERNANCE_SCORING_PROMPT) ───────────────
_US_GOVERNANCE_SCORING_PROMPT: str = ""
try:
    _gm = _load_module("_sg_gov_prompt", _AGENT_ROOT / "us_governance_prompt.py")
    _US_GOVERNANCE_SCORING_PROMPT = getattr(_gm, "US_GOVERNANCE_SCORING_PROMPT", "")
    logger.info("[StoryGeneratorEngine] Loaded US_GOVERNANCE_SCORING_PROMPT from us_governance_prompt.py")
except Exception as e:
    logger.warning(f"[StoryGeneratorEngine] Could not load us_governance_prompt.py: {e}")

# ── Load domain_rules.py (get_domain_guidance) ────────────────────────────────
_get_domain_guidance = None
try:
    _dm = _load_module("_sg_domain_rules", _AGENT_ROOT / "domain_rules.py")
    _get_domain_guidance = getattr(_dm, "get_domain_guidance", None)
    logger.info("[StoryGeneratorEngine] Loaded get_domain_guidance from domain_rules.py")
except Exception as e:
    logger.warning(f"[StoryGeneratorEngine] Could not load domain_rules.py: {e}")

# ── Load llm_client.py (get_llm_client) ──────────────────────────────────────
# Uses the OpenAI SDK directly — same as the working main.py.
# Two strategies so it works locally and inside the container.
_get_llm_client = None
try:
    _lm = _load_module("_sg_llm_client", _AGENT_ROOT / "llm_client.py")
    _get_llm_client = getattr(_lm, "get_llm_client", None)
    logger.info("[StoryGeneratorEngine] Loaded get_llm_client from llm_client.py (importlib)")
except Exception as e:
    logger.warning(f"[StoryGeneratorEngine] importlib load of llm_client.py failed: {e} — trying direct import")

if _get_llm_client is None:
    try:
        if str(_AGENT_ROOT) not in sys.path:
            sys.path.insert(0, str(_AGENT_ROOT))
        import importlib as _il
        _lm2 = _il.import_module("llm_client")
        _get_llm_client = getattr(_lm2, "get_llm_client", None)
        logger.info("[StoryGeneratorEngine] Loaded get_llm_client via direct import fallback")
    except Exception as e2:
        logger.error(f"[StoryGeneratorEngine] Both load strategies failed for llm_client.py: {e2}")


# ── Pure helper functions (identical to main.py) ──────────────────────────────

def _build_prd_context_block(prd_sections: Optional[list]) -> str:
    """Build a compact PRD context block from key sections."""
    if not prd_sections:
        return ""
    key_sections = {"s7": "USER PERSONAS", "s8": "FUNCTIONAL REQUIREMENTS", "s11": "USER FLOWS"}
    parts = []
    for s in prd_sections:
        sid = s.get("id", "")
        if sid in key_sections and s.get("content", "").strip():
            label = key_sections[sid]
            content = s["content"][:5000]
            parts.append(f"── {label} ({sid}) ──\n{content}")
    if not parts:
        return ""
    return (
        "═══ PRD CONTEXT (use for persona names, flow paths, FR traceability) ═══\n"
        + "\n\n".join(parts)
        + "\n═══ END PRD CONTEXT ═══\n\n"
    )


def _build_epic_prompt(
    epic: dict,
    epic_index: int,
    grounded_instruction: str,
    fe_only_instruction: str,
    domain_guidance: str,
    prd_sections: Optional[list] = None,
) -> str:
    """Build user prompt for a single epic."""
    feature_count = len(epic.get("features", []))
    prd_block = _build_prd_context_block(prd_sections)
    return (
        f"Generate user stories for this SINGLE epic:\n\n"
        f"{prd_block}"
        f"{grounded_instruction}"
        f"{fe_only_instruction}"
        f"{domain_guidance}\n\n"
        f"DOMAIN-SPECIFIC GUIDELINES:\n"
        f"- Include acceptance criteria that verify compliance requirements\n"
        f"- Add security-focused acceptance criteria where applicable\n\n"
        f"Epic:\n{json.dumps(epic, indent=2)}\n\n"
        f"RULES:\n"
        f"- Generate EXACTLY {feature_count} stories (one per feature in this epic).\n"
        f"- Use placeholder IDs starting from US-001 (they will be renumbered later).\n"
        f"- Set epicId to \"{epic.get('id', '')}\" for every story.\n"
        f"- Story title must match the feature name (strip FE/BE prefix).\n"
        f"- Each story MUST have 4-8 acceptance criteria. Include at least 1 positive, 1 negative, and 1 edge AC, plus additional criteria covering all meaningful scenarios.\n"
        f"Return ONLY valid JSON matching the schema."
    )


async def _generate_for_epic(
    epic: dict,
    epic_index: int,
    grounded_instruction: str,
    fe_only_instruction: str,
    domain_guidance: str,
    semaphore: asyncio.Semaphore,
    story_prompt: str,
    prd_sections: Optional[list] = None,
) -> Tuple[list, int, int]:
    """Generate stories for a single epic (runs in thread pool since LLM is sync)."""
    async with semaphore:
        epic_id = epic.get("id", f"epic_{epic_index}")
        feature_count = len(epic.get("features", []))
        logger.info(f"[EPIC-START] epic={epic_id} features={feature_count}")
        t0 = time.time()

        prompt = _build_epic_prompt(
            epic, epic_index, grounded_instruction, fe_only_instruction,
            domain_guidance, prd_sections
        )
        llm = _get_llm_client()

        loop = asyncio.get_event_loop()
        result, in_tok, out_tok = await loop.run_in_executor(
            None, llm.complete, story_prompt, prompt
        )

        stories = result.get("stories", [])
        for s in stories:
            s["epicId"] = epic_id

        logger.info(
            f"[EPIC-DONE] epic={epic_id} stories={len(stories)} "
            f"tokens_in={in_tok} tokens_out={out_tok} "
            f"elapsed={round(time.time()-t0, 2)}s"
        )
        return stories, in_tok, out_tok


def _run_us_governance_sync(
    all_stories: list,
    epics: list,
    prd_sections: Optional[list],
    gov_prompt: str,
) -> Tuple[Optional[dict], int, int]:
    """Single LLM pass to score the full story set."""
    if not all_stories:
        return None, 0, 0
    llm = _get_llm_client()
    user_prompt = (
        f"EPICS (backlog context):\n{json.dumps(epics, indent=2)}\n\n"
        f"GENERATED USER STORIES:\n{json.dumps(all_stories, indent=2)}\n\n"
    )
    if prd_sections:
        trimmed = prd_sections[:25]
        user_prompt += (
            f"PRD SECTIONS (traceability context, truncated if long):\n"
            f"{json.dumps(trimmed, indent=2)}\n\n"
        )
    user_prompt += (
        "Return ONLY the JSON object with overallScore, passThreshold, "
        "subMetrics (prdEpicCoverage, acQuality, investScore, clarityScore)."
    )
    try:
        result, in_tok, out_tok = llm.complete(gov_prompt, user_prompt)
        if isinstance(result, dict) and result.get("overallScore") is not None:
            return result, in_tok, out_tok
        logger.warning("[US_GOV] LLM returned unexpected shape")
    except Exception as e:
        logger.warning(f"[US_GOV] scoring failed: {e}")
    return None, 0, 0


# ── Engine class ──────────────────────────────────────────────────────────────

class StoryGeneratorEngine:
    """Config-driven engine — all logic from original main.py /run endpoint."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        if _get_llm_client is None:
            raise RuntimeError(
                "[StoryGeneratorEngine] Could not load original llm_client.py — "
                "check that llm_client.py exists at the agent root and is importable."
            )
        self._llm_factory = _get_llm_client
        self._max_concurrent = int(
            config.get("execution_profile", {}).get("max_concurrent_epics", MAX_CONCURRENT)
        )
        self._story_prompt = _STORY_GENERATOR_PROMPT
        self._gov_prompt = _US_GOVERNANCE_SCORING_PROMPT

    async def execute(
        self, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Main dispatch: refine or generate, exactly as original main.py."""
        t0 = time.time()

        epics: list = input_data.get("epics", [])
        capabilities: list = input_data.get("capabilities", [])
        domain: str = input_data.get("domain", "")
        refine: bool = input_data.get("refine", False)
        story_to_refine: dict = input_data.get("story_to_refine", {})
        user_instruction: str = input_data.get("user_instruction", "")
        grounded: bool = input_data.get("grounded", True)
        fe_only: bool = input_data.get("fe_only", False)
        prd_sections: Optional[list] = input_data.get("prd_sections", None)

        mode = "refine" if refine else "generate"
        logger.info(
            f"[REQ] mode={mode} epics={len(epics)} domain={domain} "
            f"grounded={grounded} fe_only={fe_only}"
        )

        # ── Refine mode ───────────────────────────────────────────────────────
        if refine and story_to_refine:
            user_prompt = (
                f"You are refining a single user story based on the user's instruction.\n\n"
                f"STORY TO REFINE:\n{json.dumps(story_to_refine, indent=2)}\n\n"
                f"USER INSTRUCTION: {user_instruction}\n\n"
                f"Return a JSON object with a 'stories' array containing ONLY the refined story. "
                f"Keep the same id and epicId. Apply the user's instruction to modify the story title, "
                f"asA, iWant, soThat, or acceptance criteria as requested. "
                f"Return ONLY valid JSON matching the schema."
            )
            llm = self._llm_factory()
            result, in_tok, out_tok = llm.complete(self._story_prompt, user_prompt)
            stories = result.get("stories", [])
            logger.info(
                f"[RES] mode=refine stories={len(stories)} "
                f"tokens_in={in_tok} tokens_out={out_tok} "
                f"elapsed={round(time.time()-t0, 2)}s"
            )
            return (
                {"stories": stories, "us_governance": None, "input_tokens": in_tok, "output_tokens": out_tok},
                {},
            )

        # ── Generate mode — parallel per-epic ─────────────────────────────────
        domain_guidance = ""
        if domain and _get_domain_guidance:
            try:
                domain_guidance = _get_domain_guidance(domain)
            except Exception as e:
                logger.warning(f"[StoryGeneratorEngine] get_domain_guidance error: {e}")
                domain_guidance = f"Domain: {domain}"

        grounded_instruction = (
            "GROUNDED GENERATION MODE (STRICT):\n"
            "- Generate user stories ONLY based on the provided epics and features.\n"
            "- Do NOT add stories for functionality not present in the input epics/features.\n"
            "- Acceptance criteria must reflect ONLY what the feature explicitly requires.\n"
            "- Do NOT use general knowledge to invent additional requirements or stories.\n\n"
        ) if grounded else (
            "ENHANCED GENERATION MODE:\n"
            "- Use the provided epics/features as the primary source, but you MAY enhance stories with your expertise.\n"
            "- Add stories for common edge cases, error handling, and UX best practices even if not explicitly listed.\n"
            "- Enrich acceptance criteria with industry-standard validations and checks.\n"
            "- Suggest additional stories that would improve the overall product quality.\n\n"
        )

        fe_only_instruction = (
            "FRONT-END STORIES ONLY MODE:\n"
            "- Generate ONLY front-end related user stories.\n"
            "- Focus on: UI rendering, user interactions, form validation (client-side), navigation, responsiveness, accessibility, visual feedback, loading states.\n"
            "- EXCLUDE stories about: API endpoints, database operations, server-side logic, authentication backends, data migrations, cron jobs.\n"
            "- The 'asA' role should be an end-user or front-end developer, NOT a backend engineer or system admin.\n"
            "- Acceptance criteria should be verifiable from the UI perspective.\n\n"
        ) if fe_only else ""

        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks = [
            _generate_for_epic(
                epic, i, grounded_instruction, fe_only_instruction,
                domain_guidance, semaphore, self._story_prompt, prd_sections
            )
            for i, epic in enumerate(epics)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_stories: list = []
        total_in = 0
        total_out = 0
        failed_epics: list = []

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                epic_id = epics[i].get("id", f"epic_{i}")
                logger.error(f"[EPIC-FAIL] epic={epic_id} error={res}")
                failed_epics.append(epic_id)
                continue
            stories, in_tok, out_tok = res
            all_stories.extend(stories)
            total_in += in_tok
            total_out += out_tok

        # Renumber story IDs sequentially: US-101, US-102, ...
        for idx, story in enumerate(all_stories):
            story["id"] = f"US-{101 + idx}"
            for ac_idx, ac in enumerate(story.get("ac", [])):
                ac["id"] = f"AC-{ac_idx + 1:02d}"

        if failed_epics and not all_stories:
            raise Exception(f"All epic generations failed: {failed_epics}")

        if failed_epics:
            logger.warning(
                f"[PARTIAL] {len(failed_epics)} epics failed: {failed_epics}, "
                f"{len(all_stories)} stories generated from remaining epics"
            )

        us_gov: Optional[dict] = None
        if all_stories:
            loop = asyncio.get_event_loop()
            gov, gin, gout = await loop.run_in_executor(
                None,
                lambda: _run_us_governance_sync(
                    all_stories, epics, prd_sections, self._gov_prompt
                ),
            )
            us_gov = gov
            total_in += gin
            total_out += gout
            if us_gov:
                logger.info(f"[US_GOV] overallScore={us_gov.get('overallScore')}")

        logger.info(
            f"[RES] stories={len(all_stories)} "
            f"epics_ok={len(epics)-len(failed_epics)}/{len(epics)} "
            f"tokens_in={total_in} tokens_out={total_out} "
            f"elapsed={round(time.time()-t0, 2)}s"
        )
        return (
            {
                "stories": all_stories,
                "us_governance": us_gov,
                "input_tokens": total_in,
                "output_tokens": total_out,
            },
            {},
        )
