from unittest.mock import patch

import pytest

from rag_lib.graph.store import NetworkXGraphStore, create_graph_store


def test_create_graph_store_networkx_provider():
    store = create_graph_store(provider="networkx")
    assert isinstance(store, NetworkXGraphStore)


def test_create_graph_store_neo4j_with_explicit_auth():
    with patch("rag_lib.graph.store.Neo4jGraphStore") as mock_neo4j:
        sentinel_store = object()
        mock_neo4j.return_value = sentinel_store

        store = create_graph_store(
            provider="neo4j",
            uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
            database="analytics",
        )

        assert store is sentinel_store
        mock_neo4j.assert_called_once_with(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
            database="analytics",
        )


def test_create_graph_store_neo4j_settings_fallback(monkeypatch):
    monkeypatch.setenv("GRAPH_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_URI", "bolt://db:7687")
    monkeypatch.setenv("GRAPH_USERNAME", "svc_user")
    monkeypatch.setenv("GRAPH_PASSWORD", "svc_secret")
    monkeypatch.setenv("GRAPH_DATABASE", "graph_db")

    with patch("rag_lib.graph.store.Neo4jGraphStore") as mock_neo4j:
        sentinel_store = object()
        mock_neo4j.return_value = sentinel_store

        store = create_graph_store()

        assert store is sentinel_store
        mock_neo4j.assert_called_once_with(
            uri="bolt://db:7687",
            auth=("svc_user", "svc_secret"),
            database="graph_db",
        )


def test_create_graph_store_auth_overrides_username_password(monkeypatch):
    monkeypatch.setenv("GRAPH_PROVIDER", "neo4j")
    monkeypatch.setenv("GRAPH_USERNAME", "env_user")
    monkeypatch.setenv("GRAPH_PASSWORD", "env_secret")

    with patch("rag_lib.graph.store.Neo4jGraphStore") as mock_neo4j:
        sentinel_store = object()
        mock_neo4j.return_value = sentinel_store

        store = create_graph_store(
            provider="neo4j",
            uri="bolt://localhost:7687",
            auth=("explicit_user", "explicit_secret"),
            username="arg_user",
            password="arg_secret",
        )

        assert store is sentinel_store
        mock_neo4j.assert_called_once_with(
            uri="bolt://localhost:7687",
            auth=("explicit_user", "explicit_secret"),
            database="neo4j",
        )


def test_create_graph_store_unknown_provider():
    with pytest.raises(ValueError, match="Unknown Graph Store provider"):
        create_graph_store(provider="falkordb")


def test_create_graph_store_neo4j_requires_uri():
    with pytest.raises(ValueError, match="requires `uri`"):
        create_graph_store(provider="neo4j", auth=("neo4j", "password"))


def test_create_graph_store_neo4j_requires_auth_or_credentials():
    with pytest.raises(ValueError, match="requires `auth` or `username`/`password`"):
        create_graph_store(provider="neo4j", uri="bolt://localhost:7687")


def test_graph_store_import_surface_exposes_neo4j_graph_store():
    from rag_lib.graph.neo4j_store import Neo4jGraphStore as direct_neo4j_store
    from rag_lib.graph.store import Neo4jGraphStore as store_neo4j_store

    assert store_neo4j_store is direct_neo4j_store
