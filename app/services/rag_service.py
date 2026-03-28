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


async def retrieve_relevant_entries(
    twin_id: UUID,
    query: str,
    limit: int = 5,
    db: AsyncSession = None,
) -> List[RagEntry]:
    """Retrieve the most relevant RAG entries for a query.

    Strategy:
    1. If embeddings are available and query can be embedded → vector similarity search
    2. Otherwise → conviction-based retrieval (highest conviction first)
    """
    # Try vector similarity search
    query_embedding = await generate_embedding(query)
    if query_embedding is not None:
        entries = await _vector_search(twin_id, query_embedding, limit, db)
        if entries:
            return entries

    # Fallback: conviction-based retrieval
    return await _conviction_search(twin_id, limit, db)


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
