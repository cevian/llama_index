from typing import List, Any, Dict, Union, Generator

import pytest
import asyncio
import os

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import TimescaleVectorStore
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStoreQuery,
    MetadataFilters,
    ExactMatchFilter,
)

# from testing find install here https://github.com/timescale/python-vector/

TEST_SERVICE_URL = os.environ.get(
    "TEST_TIMESCALE_SERVICE_URL", "postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require")
TEST_TABLE_NAME = "lorem_ipsum"

try:
    from timescale_vector import client  # noqa: F401
    cli = client.Sync(TEST_SERVICE_URL, TEST_TABLE_NAME, 1536)
    with cli.connect() as conn:
        pass

    cli.close()

    timescale_not_available = False
except (ImportError, Exception):
    timescale_not_available = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    conn_ = psycopg2.connect(TEST_SERVICE_URL)  # type: ignore
    return conn_


@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        conn.commit()
    yield
    with conn.cursor() as c:
        # c.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        conn.commit()


@pytest.fixture
def tvs(db: None) -> Any:
    tvs = TimescaleVectorStore.from_params(
        service_url=TEST_SERVICE_URL,
        table_name=TEST_TABLE_NAME,
    )

    yield tvs

    try:
        asyncio.get_event_loop().run_until_complete(tvs.close())
    except RuntimeError:
        asyncio.run(tvs.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0] * 1536,
            node=TextNode(
                text="lorem ipsum",
                id_="aaa",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.1] * 1536,
            node=TextNode(
                text="dolor sit amet",
                id_="bbb",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
                extra_info={"test_key": "test_value"},
            ),
        ),
    ]


@pytest.mark.skipif(timescale_not_available, reason="timescale vector store is not available")
@pytest.mark.asyncio
async def test_instance_creation(db: None) -> None:
    tvs = TimescaleVectorStore.from_params(
        service_url=TEST_SERVICE_URL,
        table_name=TEST_TABLE_NAME,
    )
    assert isinstance(tvs, TimescaleVectorStore)
    await tvs.close()


@pytest.mark.skipif(timescale_not_available, reason="timescale vector store is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_add_to_db_and_query(
    tvs: TimescaleVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)
    q = VectorStoreQuery(query_embedding=[1] * 1536, similarity_top_k=1)
    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(timescale_not_available, reason="timescale vector store is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_add_to_db_and_query_with_metadata_filters(
    tvs: TimescaleVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=[0.5] * 1536, similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(timescale_not_available, reason="timescale vector store is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_add_to_db_query_and_delete(
    tvs: TimescaleVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)

    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"
    tvs.delete("bbb")

    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(timescale_not_available, reason="timescale vector store is not available")
def test_add_to_db_query_and_delete(
    tvs: TimescaleVectorStore, node_embeddings: List[NodeWithEmbedding]
) -> None:
    tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)
    res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"

    tvs.create_index()
    tvs.drop_index()

    tvs.create_index("tsv", max_alpha=1.0, num_neighbors=50)
    tvs.drop_index()

    tvs.create_index("ivfflat", num_lists=20, num_records=1000)
    tvs.drop_index()

    tvs.create_index("hnsw", m=16, ef_construction=64)
    tvs.drop_index()
