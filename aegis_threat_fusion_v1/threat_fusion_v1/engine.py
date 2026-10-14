from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any, Iterable

# Frozen AEGIS architecture weights. These are heuristic weights, not learned probabilities.
WEIGHTS = {
    "weapon_detection": 0.40,
    "crowd_alert": 0.20,
    "fire": 0.25,
    "smoke": 0.25,
    "osint_threat": 0.15,
    "nlp_threat": 0.15,
}
THREAT_KEYWORDS = (
    "threat", "attack", "weapon", "bomb", "gun", "violence",
    "fire", "danger", "explosion", "shooting", "riot", "stampede",
    "stabbing", "explosive", "arson", "crowd crush", "security breach",
)


def parse_timestamp(value: Any) -> datetime:
    """Parse ISO-8601 timestamps; naive timestamps are treated as UTC explicitly."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_coordinates(event: dict[str, Any]) -> tuple[float, float] | None:
    location = event.get("location")
    if not isinstance(location, dict):
        return None
    lat, lon = location.get("lat"), location.get("lon")
    if (isinstance(lat, bool) or not isinstance(lat, (int, float))
            or isinstance(lon, bool) or not isinstance(lon, (int, float))):
        return None
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine distance between latitude/longitude pairs, in metres."""
    lat1, lon1 = a
    lat2, lon2 = b
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6_371_000 * 2 * asin(min(1.0, sqrt(h)))


class ThreatFusionEngine:
    """Fuse normalized AEGIS events deterministically.

    Level boundaries are non-overlapping: LOW < .30, MEDIUM [.30,.60),
    HIGH [.60,.80), CRITICAL [.80,1.00]. A score is a rule-based index, not
    a calibrated probability or proof that a threat is real.
    """

    def __init__(
        self,
        *,
        time_window_seconds: int = 60,
        osint_radius_meters: float = 1000.0,
        crowd_capacity_by_zone: dict[str, int] | None = None,
        threat_keywords: Iterable[str] = THREAT_KEYWORDS,
    ) -> None:
        if time_window_seconds < 0:
            raise ValueError("time_window_seconds must be >= 0")
        if osint_radius_meters < 0:
            raise ValueError("osint_radius_meters must be >= 0")
        self.time_window_seconds = int(time_window_seconds)
        self.osint_radius_meters = float(osint_radius_meters)
        self.crowd_capacity_by_zone = crowd_capacity_by_zone or {}
        self.threat_keywords = tuple(str(word).lower() for word in threat_keywords if str(word).strip())

    @staticmethod
    def _zone(event: dict[str, Any]) -> str | None:
        source = event.get("source")
        if not isinstance(source, dict):
            return None
        zone = source.get("zone")
        return str(zone).strip() if zone is not None and str(zone).strip() else None

    @staticmethod
    def _event_type(event: dict[str, Any]) -> str:
        kind = str(event.get("type", "")).strip().lower()
        aliases = {"weapon": "weapon_detection", "gun": "weapon_detection", "pistol": "weapon_detection",
                   "knife": "weapon_detection", "fire_detection": "fire", "smoke_detection": "smoke",
                   "osint": "osint_threat", "news_threat": "osint_threat"}
        return aliases.get(kind, kind)

    @staticmethod
    def _confidence(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence is missing or not numeric")
        score = float(value)
        if not 0 <= score <= 1:
            raise ValueError("confidence must be within [0, 1]")
        return score

    def _dedupe_key(self, event: dict[str, Any]) -> tuple[Any, ...]:
        kind = self._event_type(event)
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        # Upstream video reports are aggregated per category by the adapter. Track-aware
        # integrations can still suppress repeated frames sharing a track ID.
        if kind in {"weapon_detection", "fire", "smoke"} and details.get("track_id") is not None:
            return (kind, source.get("id"), self._zone(event), str(details["track_id"]))
        return ("event_id", str(event.get("event_id", "")))

    def fuse(
        self,
        events: Iterable[dict[str, Any]],
        *,
        zone: str | None = None,
        reference_time: str | datetime | None = None,
    ) -> dict[str, Any]:
        items = list(events)
        excluded: list[dict[str, Any]] = []
        by_id = {
            item.get("event_id"): item for item in items
            if isinstance(item, dict) and isinstance(item.get("event_id"), str)
        }

        def exclude(event_or_id: dict[str, Any] | str, reason: str) -> None:
            event = event_or_id if isinstance(event_or_id, dict) else by_id.get(event_or_id)
            event_id = event.get("event_id") if isinstance(event, dict) else event_or_id
            entry: dict[str, Any] = {"event_id": str(event_id), "reason": reason}
            if isinstance(event, dict):
                if event.get("source") is not None:
                    entry["source"] = event.get("source")
                if event.get("raw_ref") is not None:
                    entry["raw_ref"] = event.get("raw_ref")
            excluded.append(entry)

        valid: list[tuple[dict[str, Any], datetime]] = []

        for index, event in enumerate(items):
            if not isinstance(event, dict):
                exclude(f"index:{index}", "event must be a JSON object")
                continue
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id.strip():
                exclude(f"index:{index}", "missing event_id")
                continue
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            if details.get("adapter_exclusion_reason"):
                exclude(event, str(details["adapter_exclusion_reason"]))
                continue
            try:
                timestamp = parse_timestamp(event.get("timestamp"))
            except ValueError as exc:
                exclude(event, str(details.get("timestamp_exclusion_reason") or exc))
                continue
            valid.append((event, timestamp))

        if isinstance(reference_time, str):
            reference_dt = parse_timestamp(reference_time)
        elif isinstance(reference_time, datetime):
            reference_dt = reference_time.astimezone(timezone.utc) if reference_time.tzinfo else reference_time.replace(tzinfo=timezone.utc)
        else:
            reference_dt = max((ts for _, ts in valid), default=None)

        if reference_dt is None:
            return self._empty_result(zone, excluded, None)

        eligible: list[tuple[dict[str, Any], datetime]] = []
        for event, timestamp in valid:
            event_id = event["event_id"]
            event_zone = self._zone(event)
            if zone is not None and event_zone != zone:
                exclude(event, f"zone mismatch or missing zone: {event_zone!r}")
                continue
            if zone is None and event_zone is None:
                exclude(event, "missing source.zone; pass zone or populate event zone")
                continue
            delta = abs((timestamp - reference_dt).total_seconds())
            if delta > self.time_window_seconds:
                exclude(event, f"outside ±{self.time_window_seconds}s time window ({delta:.1f}s away)")
                continue
            eligible.append((event, timestamp))

        # Stable deterministic de-duplication: highest-confidence record wins for duplicate tracks.
        chosen: dict[tuple[Any, ...], tuple[dict[str, Any], datetime]] = {}
        for event, timestamp in eligible:
            key = self._dedupe_key(event)
            old = chosen.get(key)
            try:
                confidence = self._confidence(event.get("confidence"))
            except ValueError:
                confidence = -1.0
            if old is None:
                chosen[key] = (event, timestamp)
            else:
                try:
                    old_conf = self._confidence(old[0].get("confidence"))
                except ValueError:
                    old_conf = -1.0
                if confidence > old_conf:
                    exclude(old[0], "duplicate track suppressed; lower-confidence observation")
                    chosen[key] = (event, timestamp)
                else:
                    exclude(event, "duplicate track suppressed")

        deduped = [pair[0] for pair in sorted(chosen.values(), key=lambda pair: (pair[1], pair[0]["event_id"]))]
        # Only camera/video evidence anchors OSINT distance checks. Do not use the OSINT
        # location itself as its own anchor.
        anchors = [parse_coordinates(e) for e in deduped
                   if str(e.get("modality", "")).lower() == "video" and parse_coordinates(e) is not None]

        contributions: list[dict[str, Any]] = []
        for event in deduped:
            eid = event["event_id"]
            kind = self._event_type(event)
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            try:
                confidence = self._confidence(event.get("confidence"))
            except ValueError as exc:
                exclude(event, str(exc))
                continue

            if kind in {"weapon_detection", "fire", "smoke"}:
                if confidence <= 0.60:
                    exclude(event, f"{kind} confidence must be strictly > 0.60")
                    continue
                weight = WEIGHTS[kind]
                subtype = str(event.get("subtype", "")).strip()
                label = subtype or ("weapon" if kind == "weapon_detection" else kind)
                reason = f"{label} detected ({weight:.2f} × {confidence:.2f})"
            elif kind == "crowd_alert":
                count = details.get("count")
                capacity = details.get("capacity")
                event_zone = self._zone(event)
                if capacity is None:
                    capacity = self.crowd_capacity_by_zone.get(event_zone)
                if (isinstance(count, bool) or not isinstance(count, (int, float))
                        or isinstance(capacity, bool) or not isinstance(capacity, (int, float))):
                    exclude(event, "crowd count/capacity missing or invalid")
                    continue
                if count <= capacity:
                    exclude(event, f"crowd not over capacity ({count}/{capacity})")
                    continue
                weight, confidence = WEIGHTS[kind], 1.0
                reason = f"crowd over capacity {count:g}/{capacity:g} ({weight:.2f} × 1.00)"
            elif kind in {"osint_threat", "nlp_threat"}:
                event_coords = parse_coordinates(event)
                if details.get("location_exclusion_reason"):
                    exclude(event, str(details["location_exclusion_reason"]))
                    continue
                if event_coords is None:
                    exclude(event, "missing or invalid OSINT incident coordinates")
                    continue
                if not anchors:
                    exclude(event, "OSINT/NLP needs valid coordinates and a video event location anchor; actual camera coordinates are missing")
                    continue
                nearest = min(distance_m(event_coords, anchor) for anchor in anchors)
                if nearest > self.osint_radius_meters:
                    exclude(event, f"outside OSINT radius ({nearest:.0f}m > {self.osint_radius_meters:.0f}m)")
                    continue
                text_fields = [details.get(k, "") for k in ("text", "title", "description", "category", "evidence", "url", "entities")]
                searchable = " ".join(str(value) for value in text_fields).lower()
                matched = [word for word in self.threat_keywords if word in searchable]
                if not matched:
                    exclude(event, "no configured threat keyword match")
                    continue
                weight = WEIGHTS[kind]
                basis = details.get("confidence_basis", "upstream event confidence")
                reason = f"{kind} nearby ({weight:.2f} × {confidence:.2f}, {nearest:.0f}m; keyword={matched[0]!r}; {basis})"
            else:
                exclude(event, f"unsupported event type: {kind or '<missing>'}")
                continue

            contribution = {
                "event_id": eid,
                "event_type": kind,
                "weight": weight,
                "confidence": confidence,
                "sub_score": round(weight * confidence, 6),
                "reason": reason,
                "source": event.get("source"),
                "raw_ref": event.get("raw_ref"),
                "timestamp": event.get("timestamp"),
            }
            if details.get("timestamp_basis"):
                contribution["timestamp_basis"] = details["timestamp_basis"]
            contributions.append(contribution)

        score = min(1.0, sum(item["sub_score"] for item in contributions))
        level = "LOW" if score < 0.30 else "MEDIUM" if score < 0.60 else "HIGH" if score < 0.80 else "CRITICAL"
        resolved_zone = zone or next((self._zone(e) for e in deduped if self._zone(e)), None)
        return {
            "schema_version": "aegis.threat_fusion.v1",
            "zone": resolved_zone,
            "window": {"seconds": self.time_window_seconds, "reference_timestamp": reference_dt.isoformat().replace("+00:00", "Z")},
            "threat_score": round(score, 6),
            "threat_level": level,
            "score_semantics": "explainable weighted rule score; not a calibrated probability",
            "evidence_count": len(contributions),
            "event_ids": [item["event_id"] for item in contributions],
            "contributions": contributions,
            "reasons": [item["reason"] for item in contributions],
            "excluded_events": excluded,
        }

    @staticmethod
    def _empty_result(zone: str | None, excluded: list[dict[str, str]], reference_dt: datetime | None) -> dict[str, Any]:
        return {
            "schema_version": "aegis.threat_fusion.v1", "zone": zone,
            "window": {"seconds": 0, "reference_timestamp": reference_dt.isoformat() if reference_dt else None},
            "threat_score": 0.0, "threat_level": "LOW",
            "score_semantics": "explainable weighted rule score; not a calibrated probability",
            "evidence_count": 0, "event_ids": [], "contributions": [], "reasons": [], "excluded_events": excluded,
        }
