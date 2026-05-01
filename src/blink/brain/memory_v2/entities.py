"""Entity registry for Blink continuity memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip().lower() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


@dataclass(frozen=True)
class BrainEntityRecord:
    """One normalized continuity entity."""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases_json: str
    attributes_json: str
    created_at: str
    updated_at: str

    @property
    def aliases(self) -> list[str]:
        """Return normalized aliases."""
        return list(json.loads(self.aliases_json))

    @property
    def attributes(self) -> dict[str, Any]:
        """Return decoded entity attributes."""
        return dict(json.loads(self.attributes_json))


class EntityRegistry:
    """Canonical continuity entity registry."""

    def __init__(self, *, store):
        """Bind the registry to one canonical store."""
        self._store = store

    def ensure_entity(
        self,
        entity_type: str,
        canonical_name: str,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> BrainEntityRecord:
        """Create or update one normalized entity row."""
        normalized_name = " ".join((canonical_name or "").split()).strip()
        if not normalized_name:
            raise ValueError("Entity canonical name must not be empty.")
        existing = self.find_entities_by_name(normalized_name, entity_type=entity_type)
        if existing:
            record = existing[0]
            merged_aliases = sorted({*record.aliases, *(aliases or [])})
            merged_attributes = {**record.attributes, **(attributes or {})}
            if merged_aliases != record.aliases or merged_attributes != record.attributes:
                self._store._conn.execute(
                    """
                    UPDATE entities
                    SET aliases_json = ?, attributes_json = ?, updated_at = ?
                    WHERE entity_id = ?
                    """,
                    (
                        json.dumps(merged_aliases, ensure_ascii=False, sort_keys=True),
                        json.dumps(merged_attributes, ensure_ascii=False, sort_keys=True),
                        self._store._utc_now_for_memory_v2(),
                        record.entity_id,
                    ),
                )
                self._store._conn.commit()
                return self.get_entity(record.entity_id)
            return record

        now = self._store._utc_now_for_memory_v2()
        entity_id = _stable_id("entity", entity_type, normalized_name)
        self._store._conn.execute(
            """
            INSERT OR IGNORE INTO entities (
                entity_id, entity_type, canonical_name, aliases_json, attributes_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                entity_type,
                normalized_name,
                json.dumps(sorted(set(aliases or [])), ensure_ascii=False, sort_keys=True),
                json.dumps(attributes or {}, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        self._store._conn.commit()
        return self.get_entity(entity_id)

    def get_entity(self, entity_id: str) -> BrainEntityRecord:
        """Return one entity by id."""
        row = self._store._conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Missing entity id {entity_id}")
        return BrainEntityRecord(
            entity_id=str(row["entity_id"]),
            entity_type=str(row["entity_type"]),
            canonical_name=str(row["canonical_name"]),
            aliases_json=str(row["aliases_json"]),
            attributes_json=str(row["attributes_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def find_entities_by_name(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> list[BrainEntityRecord]:
        """Find entities by exact canonical name or alias."""
        normalized = " ".join((name or "").split()).strip().lower()
        clauses = []
        params: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        query = "SELECT * FROM entities"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        rows = self._store._conn.execute(query, tuple(params)).fetchall()
        matches: list[BrainEntityRecord] = []
        for row in rows:
            record = BrainEntityRecord(
                entity_id=str(row["entity_id"]),
                entity_type=str(row["entity_type"]),
                canonical_name=str(row["canonical_name"]),
                aliases_json=str(row["aliases_json"]),
                attributes_json=str(row["attributes_json"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
            alias_match = any(alias.lower() == normalized for alias in record.aliases)
            if record.canonical_name.lower() == normalized or alias_match:
                matches.append(record)
        return matches
