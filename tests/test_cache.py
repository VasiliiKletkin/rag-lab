import time

from app.cache import LRUCache, answer_key, normalize


def test_normalize_merges_spelling_variants() -> None:
    assert normalize("  Какие  индексы в pgvector? ") == normalize("какие индексы в pgvector")


def test_key_changes_with_index_version() -> None:
    assert answer_key("q", 5, "206:a") != answer_key("q", 5, "207:b")
    assert answer_key("q", 5, "v") != answer_key("q", 3, "v")


def test_lru_evicts_oldest_and_counts() -> None:
    cache: LRUCache[str] = LRUCache(maxsize=2, ttl_seconds=60)
    cache.put("a", "1")
    cache.put("b", "2")
    assert cache.get("a") == "1"  # a теперь свежее b
    cache.put("c", "3")  # вытесняется b

    assert cache.get("b") is None
    assert cache.get("c") == "3"
    assert (cache.stats.hits, cache.stats.misses, cache.stats.size) == (2, 1, 2)


def test_ttl_expires(monkeypatch) -> None:
    cache: LRUCache[str] = LRUCache(maxsize=10, ttl_seconds=10)
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now)
    cache.put("a", "1")
    monkeypatch.setattr(time, "monotonic", lambda: now + 11)

    assert cache.get("a") is None
