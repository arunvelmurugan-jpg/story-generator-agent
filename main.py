"""
Agent 5 — Story Generator
Port: 5005
Generates INVEST-compliant user stories from epics/features.
Uses asyncio.gather to process epics in parallel for speed.
"""
import os
import json
import time
import asyncio
import logging

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Any, Optional
from base_agent import create_agent_app
from llm_client import get_llm_client
from prompt import STORY_GENERATOR_PROMPT
from us_governance_prompt import US_GOVERNANCE_SCORING_PROMPT
from domain_rules import get_domain_guidance

import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("agent_story_generator")

PORT = int(os.getenv("PORT", "5005"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_EPICS", "5"))

app, router = create_agent_app(
    agent_id="stories",
    agent_name="Story Generator",
    description="Generates INVEST-compliant user stories with acceptance criteria.",
    port=PORT,
    root_path="/sub-ba-story-generate",
    skills=[{
        "id": "stories_run",
        "name": "Generate Stories",
        "description": "Create user stories from epics with AC and quality scores",
    }],
)


class StoryRequest(BaseModel):
    epics: list[dict] = []
    capabilities: list[dict] = []
    domain: str = ""
    refine: bool = False
    story_to_refine: dict = {}
    user_instruction: str = ""
    grounded: bool = True
    fe_only: bool = False
    prd_sections: Optional[list[dict]] = None
    # Scope Creep: incremental mode fields
    mode: str = "full"  # "full" or "incremental"
    existing_stories: list[dict] = []
    last_story_id: str = "US-100"


class StoryResponse(BaseModel):
    stories: list[dict] = []
    us_governance: Optional[dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0


def _build_prd_context_block(prd_sections: Optional[list[dict]]) -> str:
    """Build a compact PRD context block from key sections."""
    if not prd_sections:
        return ""
    key_sections = {"s7": "USER PERSONAS", "s8": "FUNCTIONAL REQUIREMENTS", "s11": "USER FLOWS"}
    parts = []
    for s in prd_sections:
        sid = s.get("id", "")
        if sid in key_sections and s.get("content", "").strip():
            label = key_sections[sid]
            content = s["content"][:5000]  # cap per section
            parts.append(f"── {label} ({sid}) ──\n{content}")
    if not parts:
        return ""
    return (
        "═══ PRD CONTEXT (use for persona names, flow paths, FR traceability) ═══\n"
        + "\n\n".join(parts)
        + "\n═══ END PRD CONTEXT ═══\n\n"
    )


def _build_epic_prompt(epic: dict, epic_index: int, grounded_instruction: str, fe_only_instruction: str, domain_guidance: str, prd_sections: Optional[list[dict]] = None) -> str:
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


async def _generate_for_epic(epic: dict, epic_index: int, grounded_instruction: str, fe_only_instruction: str, domain_guidance: str, semaphore: asyncio.Semaphore, prd_sections: Optional[list[dict]] = None) -> tuple[list[dict], int, int]:
    """Generate stories for a single epic (runs in thread pool since LLM client is sync)."""
    async with semaphore:
        epic_id = epic.get("id", f"epic_{epic_index}")
        feature_count = len(epic.get("features", []))
        logger.info(f"[EPIC-START] epic={epic_id} features={feature_count}")
        t0 = time.time()

        prompt = _build_epic_prompt(epic, epic_index, grounded_instruction, fe_only_instruction, domain_guidance, prd_sections)
        llm = get_llm_client()

        loop = asyncio.get_event_loop()
        result, in_tok, out_tok = await loop.run_in_executor(
            None, llm.complete, STORY_GENERATOR_PROMPT, prompt
        )

        stories = result.get("stories", [])
        # Ensure epicId is correct on every story
        for s in stories:
            s["epicId"] = epic_id

        logger.info(f"[EPIC-DONE] epic={epic_id} stories={len(stories)} tokens_in={in_tok} tokens_out={out_tok} elapsed={round(time.time()-t0, 2)}s")
        return stories, in_tok, out_tok


def _run_us_governance_sync(
    all_stories: list[dict],
    epics: list[dict],
    prd_sections: Optional[list[dict]],
) -> tuple[Optional[dict[str, Any]], int, int]:
    """Single LLM pass to score the full story set."""
    if not all_stories:
        return None, 0, 0
    llm = get_llm_client()

    # Pre-compute story stats to anchor LLM analysis
    # NOTE: stories use "ac" as the acceptance criteria field name
    total_stories = len(all_stories)
    stories_without_ac = sum(1 for s in all_stories if len(s.get("ac", [])) < 3)
    stories_with_generic_persona = sum(
        1 for s in all_stories
        if (s.get("asA") or "").strip().lower() in ("user", "system", "", "the user", "a user")
    )
    stories_missing_so_that = sum(
        1 for s in all_stories
        if not (s.get("soThat") or "").strip()
        or (s.get("soThat") or "").strip() == (s.get("iWant") or "").strip()
    )
    stories_with_low_quality = sum(
        1 for s in all_stories
        if isinstance(s.get("quality"), (int, float)) and s["quality"] < 60
    )
    avg_ac_count = (
        sum(len(s.get("ac", [])) for s in all_stories) / total_stories
        if total_stories else 0
    )
    # Count stories missing negative/edge ACs
    stories_missing_negative_ac = sum(
        1 for s in all_stories
        if not any(ac.get("type") == "negative" for ac in s.get("ac", []))
    )
    stories_missing_edge_ac = sum(
        1 for s in all_stories
        if not any(ac.get("type") == "edge" for ac in s.get("ac", []))
    )
    # Count stories where ACs have vague given/when/then
    VAGUE_TERMS = {"works correctly", "loads fast", "user-friendly", "appropriate", "handles errors",
                   "behaves as expected", "suitable response", "properly", "as expected", "correctly"}
    stories_with_vague_ac = sum(
        1 for s in all_stories
        if any(
            any(vague in (ac.get(field) or "").lower() for vague in VAGUE_TERMS)
            for ac in s.get("ac", [])
            for field in ("given", "when", "then")
        )
    )
    total_features = sum(len(e.get("features", [])) for e in epics)
    coverage_gap = max(0, total_features - total_stories)

    stats_block = (
        f"PRE-COMPUTED BACKLOG STATISTICS (use these to calibrate your scoring):\n"
        f"NOTE: Acceptance criteria in each story are stored under the key 'ac' (not 'acceptanceCriteria').\n"
        f"  Total stories generated: {total_stories}\n"
        f"  Total epic features (expected story count): {total_features}\n"
        f"  Feature coverage gap (features with no story): {coverage_gap}\n"
        f"  Average AC count per story: {avg_ac_count:.1f}\n"
        f"  Stories with fewer than 3 ACs (critical AC gap): {stories_without_ac}\n"
        f"  Stories missing at least one NEGATIVE AC: {stories_missing_negative_ac}\n"
        f"  Stories missing at least one EDGE AC: {stories_missing_edge_ac}\n"
        f"  Stories with vague AC language: {stories_with_vague_ac}\n"
        f"  Stories with generic persona ('User'/'System'): {stories_with_generic_persona}\n"
        f"  Stories with missing or duplicate soThat: {stories_missing_so_that}\n"
        f"  Stories with individual quality score < 60: {stories_with_low_quality}\n\n"
    )

    user_prompt = (
        f"{stats_block}"
        f"EPICS (backlog context):\n{json.dumps(epics, indent=2)}\n\n"
        f"GENERATED USER STORIES:\n{json.dumps(all_stories, indent=2)}\n\n"
    )
    if prd_sections:
        trimmed = prd_sections[:25]
        user_prompt += f"PRD SECTIONS (traceability context):\n{json.dumps(trimmed, indent=2)}\n\n"
    user_prompt += (
        "Apply the deduction-based scoring model from the system prompt. "
        "Check EVERY story. Return ONLY the JSON object with the exact schema specified."
    )
    try:
        result, in_tok, out_tok = llm.complete(US_GOVERNANCE_SCORING_PROMPT, user_prompt)
        if isinstance(result, dict) and result.get("overallScore") is not None:
            return result, in_tok, out_tok
        logger.warning("[US_GOV] LLM returned unexpected shape")
    except Exception as e:
        logger.warning(f"[US_GOV] scoring failed: {e}")
    return None, 0, 0


@router.post("/run", response_model=StoryResponse)
async def run_agent(req: StoryRequest):
    t0 = time.time()
    mode = 'refine' if req.refine else 'generate'
    logger.info(f"[REQ] mode={mode} epics={len(req.epics)} domain={req.domain} grounded={req.grounded} fe_only={req.fe_only} incremental={req.mode=='incremental'}")
    try:
        # Scope Creep: Handle incremental mode
        if req.mode == "incremental" and req.existing_stories:
            logger.info(f"[INCREMENTAL MODE] existing_stories={len(req.existing_stories)} last_story_id={req.last_story_id}")
            
            # Extract last story ID number
            try:
                last_id_num = int(req.last_story_id.split('-')[1]) if '-' in req.last_story_id else 100
            except (IndexError, ValueError):
                last_id_num = 100
            
            # Identify features that already have stories
            existing_feature_ids = set()
            for story in req.existing_stories:
                feature_id = story.get('featureId') or story.get('feature_id')
                if feature_id:
                    existing_feature_ids.add(feature_id)
            
            # Extract NEW features from epics (features without stories)
            new_features = []
            for epic in req.epics:
                epic_id = epic.get('id', '')
                for feature in epic.get('features', []):
                    feature_id = feature.get('id', '')
                    if feature_id and feature_id not in existing_feature_ids:
                        new_features.append({
                            'id': feature_id,
                            'name': feature.get('name', ''),
                            'epicId': epic_id,
                            'epicName': epic.get('name', '')
                        })
            
            logger.info(f"[INCREMENTAL] Found {len(new_features)} new features to generate stories for")
            
            if not new_features:
                logger.info("[INCREMENTAL] No new features found, returning empty story list")
                return StoryResponse(stories=[], us_governance=None, input_tokens=0, output_tokens=0)
            
            # Generate stories ONLY for new features
            domain_guidance = get_domain_guidance(req.domain) if req.domain else ""
            
            user_prompt = (
                f"Generate user stories for NEW features being added to an existing project.\n\n"
                f"EXISTING STORY COUNT: {len(req.existing_stories)}\n"
                f"LAST STORY ID: {req.last_story_id}\n"
                f"NEXT STORY ID SHOULD START AT: US-{last_id_num + 1}\n\n"
                f"NEW FEATURES TO GENERATE STORIES FOR:\n{json.dumps(new_features, indent=2)}\n\n"
                f"Domain: {req.domain}\n"
                f"{domain_guidance}\n\n"
                f"INCREMENTAL STORY GENERATION RULES:\n"
                f"- Generate ONLY stories for the NEW features listed above\n"
                f"- Start story IDs from US-{last_id_num + 1} and continue sequentially\n"
                f"- Match the style and quality of the existing stories\n"
                f"- Include 4-8 acceptance criteria per story\n"
                f"- Mark each story with is_incremental: true\n"
                f"- Ensure story IDs are unique and sequential\n\n"
                f"Return ONLY valid JSON with the stories array containing ONLY the new stories."
            )
            
            llm = get_llm_client()
            result, in_tok, out_tok = llm.complete(STORY_GENERATOR_PROMPT, user_prompt)
            new_stories = result.get("stories", [])
            
            # Ensure story IDs are correctly numbered
            for idx, story in enumerate(new_stories):
                story['id'] = f"US-{last_id_num + idx + 1}"
                story['is_incremental'] = True
            
            logger.info(f"[INCREMENTAL RES] new_stories={len(new_stories)} tokens_in={in_tok} tokens_out={out_tok}")
            return StoryResponse(stories=new_stories, us_governance=None, input_tokens=in_tok, output_tokens=out_tok)
        
        if req.refine and req.story_to_refine:
            # AI Edit mode — refine a single story based on user instruction
            user_prompt = (
                f"You are refining a single user story based on the user's instruction.\n\n"
                f"STORY TO REFINE:\n{json.dumps(req.story_to_refine, indent=2)}\n\n"
                f"USER INSTRUCTION: {req.user_instruction}\n\n"
                f"Return a JSON object with a 'stories' array containing ONLY the refined story. "
                f"Keep the same id and epicId. Apply the user's instruction to modify the story title, "
                f"asA, iWant, soThat, or acceptance criteria as requested. "
                f"Return ONLY valid JSON matching the schema."
            )
            llm = get_llm_client()
            result, in_tok, out_tok = llm.complete(STORY_GENERATOR_PROMPT, user_prompt)
            stories = result.get("stories", [])
            logger.info(f"[RES] mode=refine stories={len(stories)} tokens_in={in_tok} tokens_out={out_tok} elapsed={round(time.time()-t0, 2)}s")
            return StoryResponse(stories=stories, us_governance=None, input_tokens=in_tok, output_tokens=out_tok)

        # ── Parallel per-epic generation ──────────────────────────────
        domain_guidance = get_domain_guidance(req.domain) if req.domain else ""

        grounded_instruction = ""
        if req.grounded:
            grounded_instruction = (
                "GROUNDED GENERATION MODE (STRICT):\n"
                "- Generate user stories ONLY based on the provided epics and features.\n"
                "- Do NOT add stories for functionality not present in the input epics/features.\n"
                "- Acceptance criteria must reflect ONLY what the feature explicitly requires.\n"
                "- Do NOT use general knowledge to invent additional requirements or stories.\n\n"
            )
        else:
            grounded_instruction = (
                "ENHANCED GENERATION MODE:\n"
                "- Use the provided epics/features as the primary source, but you MAY enhance stories with your expertise.\n"
                "- Add stories for common edge cases, error handling, and UX best practices even if not explicitly listed.\n"
                "- Enrich acceptance criteria with industry-standard validations and checks.\n"
                "- Suggest additional stories that would improve the overall product quality.\n\n"
            )

        fe_only_instruction = ""
        if req.fe_only:
            fe_only_instruction = (
                "FRONT-END STORIES ONLY MODE:\n"
                "- Generate ONLY front-end related user stories.\n"
                "- Focus on: UI rendering, user interactions, form validation (client-side), navigation, responsiveness, accessibility, visual feedback, loading states.\n"
                "- EXCLUDE stories about: API endpoints, database operations, server-side logic, authentication backends, data migrations, cron jobs.\n"
                "- The 'asA' role should be an end-user or front-end developer, NOT a backend engineer or system admin.\n"
                "- Acceptance criteria should be verifiable from the UI perspective.\n\n"
            )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [
            _generate_for_epic(epic, i, grounded_instruction, fe_only_instruction, domain_guidance, semaphore, req.prd_sections)
            for i, epic in enumerate(req.epics)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results preserving epic order, renumber IDs sequentially
        all_stories: list[dict] = []
        total_in = 0
        total_out = 0
        failed_epics: list[str] = []

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                epic_id = req.epics[i].get("id", f"epic_{i}")
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
            # Renumber AC IDs within each story
            for ac_idx, ac in enumerate(story.get("ac", [])):
                ac["id"] = f"AC-{ac_idx + 1:02d}"

        if failed_epics and not all_stories:
            raise Exception(f"All epic generations failed: {failed_epics}")

        if failed_epics:
            logger.warning(f"[PARTIAL] {len(failed_epics)} epics failed: {failed_epics}, {len(all_stories)} stories generated from remaining epics")

        us_gov: Optional[dict[str, Any]] = None
        if all_stories:
            loop = asyncio.get_event_loop()
            gov, gin, gout = await loop.run_in_executor(
                None,
                lambda: _run_us_governance_sync(all_stories, req.epics, req.prd_sections),
            )
            us_gov = gov
            total_in += gin
            total_out += gout
            if us_gov:
                logger.info(f"[US_GOV] overallScore={us_gov.get('overallScore')}")

        logger.info(f"[RES] stories={len(all_stories)} epics_ok={len(req.epics)-len(failed_epics)}/{len(req.epics)} tokens_in={total_in} tokens_out={total_out} elapsed={round(time.time()-t0, 2)}s")
        return StoryResponse(stories=all_stories, us_governance=us_gov, input_tokens=total_in, output_tokens=total_out)

    except Exception as e:
        logger.error(f"Story Generator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Register all router routes on app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
