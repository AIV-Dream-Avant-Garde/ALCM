"""RAG retrieval service — keyword and vector similarity search.

Retrieves relevant knowledge base entries for prompt assembly.
Uses pgvector cosine similarity when embeddings are available,
falls back to conviction-based retrieval.
"""
import logging
from typing import List
from uuid import UUID

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_entry import RagEntry
from app.services.embedding_service import generate_embedding

logger = logging.getLogger(__name__)


# Keywords that indicate fear/need-relevant conversational context
FEAR_CONTEXT_SIGNALS = [
    "afraid", "fear", "scared", "anxious", "worry", "nervous", "avoid",
    "dread", "concern", "risk", "danger", "threat", "uncomfortable",
    "stressed", "overwhelmed", "panic", "phobia",
]
NEED_CONTEXT_SIGNALS = [
    "need", "want", "desire", "motivated", "driven", "fulfilled",
    "purpose", "meaning", "autonomy", "recognition", "security",
    "why do you", "what drives", "what matters", "important to you",
    "decision", "choose", "priority", "value",
]


async def retrieve_relevant_entries(
    twin_id: UUID,
    query: str,
    limit: int = 5,
    db: AsyncSession = None,
) -> List[RagEntry]:
    """Retrieve the most relevant RAG entries for a query.

    Strategy:
    1. Detect if query touches fear/need topics → boost those categories
    2. If embeddings available → vector similarity search
    3. Otherwise → conviction-based retrieval
    4. Merge fear/need-specific entries when contextually relevant
    """
    entries = []

    # Check if query context warrants fear/need entries
    query_lower = query.lower()
    needs_fear = any(signal in query_lower for signal in FEAR_CONTEXT_SIGNALS)
    needs_need = any(signal in query_lower for signal in NEED_CONTEXT_SIGNALS)

    # Retrieve fear/need entries specifically when contextually relevant
    if needs_fear or needs_need:
        targeted_categories = []
        if needs_fear:
            targeted_categories.append("fear")
        if needs_need:
            targeted_categories.append("need")
        targeted = await _category_search(twin_id, targeted_categories, min(2, limit), db)
        entries.extend(targeted)

    # Fill remaining slots with general retrieval
    remaining = limit - len(entries)
    if remaining > 0:
        query_embedding = await generate_embedding(query)
        if query_embedding is not None:
            general = await _vector_search(twin_id, query_embedding, remaining, db)
            if general:
                # Deduplicate against already-added fear/need entries
                existing_ids = {e.id for e in entries}
                entries.extend([e for e in general if e.id not in existing_ids])
                return entries[:limit]

        general = await _conviction_search(twin_id, remaining, db)
        existing_ids = {e.id for e in entries}
        entries.extend([e for e in general if e.id not in existing_ids])

    return entries[:limit]


async def _vector_search(
    twin_id: UUID,
    query_embedding: list[float],
    limit: int,
    db: AsyncSession,
) -> List[RagEntry]:
    """Search using pgvector cosine similarity.

    Uses raw SQL because SQLAlchemy's pgvector support requires
    the pgvector Python package with Vector type. Since we store
    embeddings as JSONB (array of floats), we cast to vector at query time.
    """
    try:
        # Cast JSONB embedding to vector and compute cosine distance
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        stmt = text("""
            SELECT id, content, topic, category, source_type, conviction,
                   (embedding::text::vector <=> :query_vec::vector) as distance
            FROM rag_entries
            WHERE twin_id = :twin_id
              AND is_active = true
              AND embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT :limit
        """)
        result = await db.execute(
            stmt,
            {"twin_id": str(twin_id), "query_vec": embedding_str, "limit": limit},
        )
        rows = result.fetchall()

        if not rows:
            return []

        # Load full ORM objects for the matched IDs
        ids = [row.id for row in rows]
        orm_result = await db.execute(
            select(RagEntry).where(RagEntry.id.in_(ids))
        )
        entries = orm_result.scalars().all()

        # Sort by the distance order from the vector search
        id_order = {row.id: i for i, row in enumerate(rows)}
        entries.sort(key=lambda e: id_order.get(e.id, 999))

        return list(entries)

    except Exception as e:
        logger.warning(f"Vector search failed (falling back to conviction): {e}")
        return []


async def _category_search(
    twin_id: UUID,
    categories: list[str],
    limit: int,
    db: AsyncSession,
) -> List[RagEntry]:
    """Retrieve entries from specific RAG categories (e.g., fear, need)."""
    result = await db.execute(
        select(RagEntry)
        .where(
            RagEntry.twin_id == twin_id,
            RagEntry.is_active == True,
            RagEntry.category.in_(categories),
        )
        .order_by(RagEntry.conviction.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def _conviction_search(
    twin_id: UUID,
    limit: int,
    db: AsyncSession,
) -> List[RagEntry]:
    """Fallback: retrieve by highest conviction score."""
    result = await db.execute(
        select(RagEntry)
        .where(RagEntry.twin_id == twin_id, RagEntry.is_active == True)
        .order_by(RagEntry.conviction.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def store_entry_with_embedding(
    twin_id: UUID,
    content: str,
    topic: str = None,
    category: str = None,
    source_type: str = None,
    source_ref: str = None,
    conviction: float = 0.5,
    db: AsyncSession = None,
) -> RagEntry:
    """Store a new RAG entry with auto-generated embedding."""
    embedding = await generate_embedding(content)

    entry = RagEntry(
        twin_id=twin_id,
        content=content,
        embedding=embedding,
        topic=topic,
        category=category,
        source_type=source_type,
        source_ref=source_ref,
        conviction=conviction,
    )
    db.add(entry)
    await db.flush()
    return entry
