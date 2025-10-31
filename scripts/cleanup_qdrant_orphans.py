import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict
from src.services.qdrant_client import qdrant_service
from src.models.database import async_session_factory
from src.models.models import Paper
from sqlalchemy import select

async def get_all_active_paper_ids():
    async with async_session_factory() as session:
        result = await session.execute(select(Paper.id))
        paper_ids = set(row[0] for row in result.fetchall())
        return paper_ids

async def get_all_qdrant_paper_ids():
    await qdrant_service.initialize()
    seen = set()
    offset = None
    # Use small batch size if you expect lots of vectors
    while True:
        points, next_offset = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            limit=1000,
            offset=offset,
            with_payload=["paper_id"]
        )
        for p in points:
            paper_id = p.payload.get('paper_id')
            if paper_id is not None:
                seen.add(paper_id)
        if next_offset is None:
            break
        offset = next_offset
    return seen

async def main():
    print("Fetching active paper IDs from DB...")
    db_paper_ids = await get_all_active_paper_ids()
    print(f"Found {len(db_paper_ids)} paper_ids in DB.")
    print("Scanning Qdrant for stored vectors...")
    qdrant_paper_ids = await get_all_qdrant_paper_ids()
    print(f"Found {len(qdrant_paper_ids)} unique paper_ids in Qdrant.")
    orphans = qdrant_paper_ids - db_paper_ids
    if not orphans:
        print("No orphan vectors found in Qdrant.")
        return
    print(f"Found {len(orphans)} paper_ids to cleanup: {orphans}")
    for pid in orphans:
        print(f"Deleting Qdrant vectors for orphan paper_id={pid} ...")
        await qdrant_service.delete_by_paper_id(pid)
    print("Orphan cleanup complete!")

if __name__ == "__main__":
    asyncio.run(main())
