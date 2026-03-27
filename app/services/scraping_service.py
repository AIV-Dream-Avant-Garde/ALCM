"""
Scraping Service — Cross-platform discovery and content ingestion.

Moved from platform's research_agent_service.py. Uses the LLM provider
abstraction for search-grounded queries to gather public information
about a person during onboarding.

IMPORTANT: Research must be factual, professional, and objective.
No gossip, rumors, fan opinions, or negative public perception.
Focus on verifiable facts: career, work, achievements, stated values.

Scraped data reliability weight: 0.6 (public content is curated,
reliable for public positions, less reliable for private behavior).
"""

import asyncio
import json
import logging
from typing import Optional, Dict
from functools import lru_cache

from .llm_provider import get_llm_provider, AIProvider

logger = logging.getLogger(__name__)


class ScrapingService:
    """Gathers public information about a person via search-grounded LLM queries."""

    def __init__(self, provider: AIProvider):
        self.llm = provider

    async def research_person(
        self,
        name: str,
        category: Optional[str] = None,
        social_handles: Optional[dict] = None,
        bio: Optional[str] = None,
    ) -> dict:
        """Run parallel research queries and return structured identity data.

        Returns a dict with keys: personality, knowledge, social_media,
        commercial, identity — suitable for populating a TwinProfile.
        """
        context_parts = [f"Researching: {name}"]
        if category:
            context_parts.append(f"Category: {category}")
        if bio:
            context_parts.append(f"Bio: {bio}")
        if social_handles:
            handles_str = ", ".join(f"{k}: {v}" for k, v in social_handles.items() if v)
            if handles_str:
                context_parts.append(f"Social handles: {handles_str}")
        context = "\n".join(context_parts)

        # Run 4 searches in parallel
        tasks = [
            self._search_career(name, category, context),
            self._search_social(name, social_handles, context),
            self._search_interviews(name, category, context),
            self._search_commercial(name, category, context),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        career_data = results[0] if not isinstance(results[0], Exception) else ""
        social_data = results[1] if not isinstance(results[1], Exception) else ""
        values_data = results[2] if not isinstance(results[2], Exception) else ""
        commercial_data = results[3] if not isinstance(results[3], Exception) else ""

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Research query {i} failed: {r}")

        return await self._synthesize(
            name, category, bio, career_data, social_data, values_data, commercial_data
        )

    async def _search_career(self, name: str, category: Optional[str], context: str) -> str:
        cat = f"as a {category}" if category else ""
        prompt = (
            f"Find factual career information about {name} {cat}. "
            f"Focus on: professional background, notable works and projects, "
            f"career milestones, awards and nominations, collaborations. "
            f"Only include verified, factual information."
        )
        return await self.llm.generate(prompt=prompt, context=context, temperature=0.3)

    async def _search_social(self, name: str, handles: Optional[dict], context: str) -> str:
        handles_info = ""
        if handles:
            handles_info = " ".join(f"{p}: @{h}" for p, h in handles.items() if h)
        prompt = (
            f"Find the official social media profiles and online presence for {name}. "
            f"{handles_info} "
            f"What platforms are they active on? What type of content do they post? "
            f"Focus only on what they themselves post — not fan commentary."
        )
        return await self.llm.generate(prompt=prompt, context=context, temperature=0.3)

    async def _search_interviews(self, name: str, category: Optional[str], context: str) -> str:
        cat = f"({category})" if category else ""
        prompt = (
            f"Find interviews, public statements, and quotes from {name} {cat}. "
            f"What causes do they advocate for? What values have they expressed? "
            f"Only include things {name} has said themselves — not third-party opinions."
        )
        return await self.llm.generate(prompt=prompt, context=context, temperature=0.3)

    async def _search_commercial(self, name: str, category: Optional[str], context: str) -> str:
        cat = f"in {category}" if category else ""
        prompt = (
            f"Find verified brand partnerships, business ventures for {name} {cat}. "
            f"What companies have they worked with? What ventures have they launched? "
            f"Only include confirmed, factual partnerships."
        )
        return await self.llm.generate(prompt=prompt, context=context, temperature=0.3)

    async def _synthesize(
        self, name: str, category: Optional[str], bio: Optional[str],
        career_data: str, social_data: str, values_data: str, commercial_data: str,
    ) -> dict:
        """Synthesize raw research into structured identity profile data."""
        prompt = f"""You are building an identity profile for {name} to power their digital twin.

CRITICAL RULES:
- Include ONLY verified, factual information
- NO gossip, rumors, fan opinions, or negative public perception
- NO speculation about personal life unless publicly stated by {name}
- Focus on professional identity, stated values, and creative output
- If information is uncertain, omit it rather than guess

Research - Career & Work:
{career_data[:2500]}

Research - Social Media:
{social_data[:2500]}

Research - Interviews & Values:
{values_data[:2500]}

Research - Commercial Activity:
{commercial_data[:2500]}

Return ONLY valid JSON with this structure:
{{
  "personality": {{
    "traits": ["8-12 personality traits based on their public persona"],
    "communication_style": "detailed description of how they communicate",
    "values": ["6-10 core values they have expressed"],
    "interests": ["personal interests they have publicly shared"],
    "humor_style": "description of their sense of humor if observable"
  }},
  "knowledge": {{
    "expertise": ["areas of professional expertise"],
    "career_highlights": ["major career milestones"],
    "awards": ["awards, nominations, recognitions"],
    "collaborations": ["notable collaborations"],
    "topics": ["topics they discuss publicly"]
  }},
  "social_media": {{
    "platforms": ["platforms with approximate follower counts if known"],
    "content_style": "type of content they create",
    "content_themes": ["recurring themes"],
    "posting_frequency": "how active they are"
  }},
  "commercial": {{
    "brand_partnerships": ["confirmed partnerships"],
    "business_ventures": ["ventures they have launched"],
    "industry_focus": ["industries they operate in"],
    "collaboration_style": "how they approach partnerships"
  }},
  "identity": {{
    "public_bio": "2-3 sentence factual summary",
    "known_for": ["what they are most recognized for"],
    "causes": ["causes they support"],
    "creative_philosophy": "their stated approach to their craft"
  }}
}}"""

        context = f"Name: {name}, Category: {category or 'Unknown'}, Bio: {bio or 'Not provided'}"
        response = await self.llm.generate(prompt=prompt, context=context, temperature=0.2)

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse synthesis JSON for {name}")
            return {
                "personality": {"raw_research": values_data[:500]},
                "knowledge": {"raw_research": career_data[:500]},
                "social_media": {"raw_research": social_data[:500]},
                "commercial": {"raw_research": commercial_data[:500]},
                "_synthesis_failed": True,
            }


@lru_cache()
def get_scraping_service() -> ScrapingService:
    return ScrapingService(get_llm_provider())
