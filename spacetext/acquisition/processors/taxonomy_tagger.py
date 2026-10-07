"""
Aerospace Taxonomy Tagger
==========================
Maps extracted document sections and abstracts to a structured aerospace
taxonomy hierarchy. Uses AIAA (American Institute of Aeronautics and Astronautics)
subject categories as the canonical taxonomy backbone — 4000+ standardized codes.

For Indian context (ISRO/DRDO), AIAA codes are extended with:
  - ISRO mission families (PSLV, GSLV, GSAT, Chandrayaan, Mangalyaan)
  - DRDO programme areas (IGMDP, AGNI, PRITHVI, AKASH, Hypersonics)

The output of this tagger is the ground-truth label for fine-tuning
and the structural depth metadata for GarudDB's Poincaré index.

Taxonomy structure (depth levels):
  Depth 1 — Domain        : "Space Systems", "Propulsion", "Guidance & Control"
  Depth 2 — Subdomain     : "Launch Vehicles", "Liquid Propellant", "Inertial Navigation"
  Depth 3 — System        : "PSLV", "Vikas Engine", "Ring Laser Gyroscope"
  Depth 4 — Sub-system    : "Stage 2 Separation", "Turbopump Assembly", "Accelerometer"
  Depth 5+ — Component    : leaf nodes (specific specs, materials, parameters)
"""

import re
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# ── Taxonomy Tree ──────────────────────────────────────────────────────────────
# Format: { "node_label": { "parent": str|None, "depth": int, "keywords": [str] } }
AEROSPACE_TAXONOMY: Dict[str, dict] = {

    # ── Depth 1: Domains ──────────────────────────────────────────────────────
    "Space Systems":         {"parent": None, "depth": 1, "keywords": ["space", "spacecraft", "satellite", "orbital"]},
    "Launch Vehicles":       {"parent": None, "depth": 1, "keywords": ["launch", "rocket", "booster", "vehicle", "lv"]},
    "Propulsion":            {"parent": None, "depth": 1, "keywords": ["propulsion", "thrust", "engine", "propellant", "combustion"]},
    "Guidance & Control":    {"parent": None, "depth": 1, "keywords": ["guidance", "navigation", "control", "gnc", "autopilot", "attitude"]},
    "Structures & Materials":{"parent": None, "depth": 1, "keywords": ["structure", "material", "composite", "thermal", "stress", "fatigue"]},
    "Electronics & Avionics":{"parent": None, "depth": 1, "keywords": ["electronics", "avionics", "radar", "sensor", "telemetry", "communication"]},
    "Aerodynamics":          {"parent": None, "depth": 1, "keywords": ["aerodynamic", "mach", "flow", "drag", "lift", "wind tunnel", "cfd"]},
    "Mission & Systems Eng":  {"parent": None, "depth": 1, "keywords": ["mission", "systems engineering", "requirement", "verification", "validation"]},
    "Astronomy & Astrophysics":{"parent": None, "depth": 1, "keywords": ["astronomy", "astrophysics", "telescope", "observation", "stellar", "galactic"]},
    "Defence Systems":       {"parent": None, "depth": 1, "keywords": ["defence", "missile", "weapon", "military", "warhead", "target"]},

    # ── Depth 2: Subdomains ───────────────────────────────────────────────────

    # Under Launch Vehicles
    "PSLV":                  {"parent": "Launch Vehicles", "depth": 2, "keywords": ["pslv", "polar satellite launch"]},
    "GSLV":                  {"parent": "Launch Vehicles", "depth": 2, "keywords": ["gslv", "geosynchronous satellite launch", "cryogenic"]},
    "LVM3":                  {"parent": "Launch Vehicles", "depth": 2, "keywords": ["lvm3", "lvm-3", "launch vehicle mark-3"]},
    "SSLV":                  {"parent": "Launch Vehicles", "depth": 2, "keywords": ["sslv", "small satellite launch"]},
    "Agni":                  {"parent": "Launch Vehicles", "depth": 2, "keywords": ["agni", "agni-v", "agni-vi", "ballistic missile"]},
    "Prithvi":               {"parent": "Launch Vehicles", "depth": 2, "keywords": ["prithvi", "surface-to-surface"]},

    # Under Space Systems
    "Chandrayaan":           {"parent": "Space Systems", "depth": 2, "keywords": ["chandrayaan", "lunar", "moon mission", "vikram", "pragyan"]},
    "Mangalyaan":            {"parent": "Space Systems", "depth": 2, "keywords": ["mangalyaan", "mars orbiter", "mom", "maven"]},
    "Aditya-L1":             {"parent": "Space Systems", "depth": 2, "keywords": ["aditya", "solar mission", "l1", "lagrange"]},
    "GSAT":                  {"parent": "Space Systems", "depth": 2, "keywords": ["gsat", "communication satellite", "geostationary"]},
    "IRNSS/NavIC":           {"parent": "Space Systems", "depth": 2, "keywords": ["irnss", "navic", "navigation satellite", "regional navigation"]},
    "RESOURCESAT":           {"parent": "Space Systems", "depth": 2, "keywords": ["resourcesat", "remote sensing", "cartosat", "risat"]},

    # Under Propulsion
    "Liquid Propulsion":     {"parent": "Propulsion", "depth": 2, "keywords": ["liquid propellant", "liquid engine", "lox", "udmh", "n2o4"]},
    "Solid Propulsion":      {"parent": "Propulsion", "depth": 2, "keywords": ["solid propellant", "solid motor", "htpb", "composite propellant"]},
    "Cryogenic Propulsion":  {"parent": "Propulsion", "depth": 2, "keywords": ["cryogenic", "lh2", "liquid hydrogen", "liquid oxygen engine"]},
    "Electric Propulsion":   {"parent": "Propulsion", "depth": 2, "keywords": ["electric propulsion", "ion thruster", "hall thruster", "plasma"]},
    "Hypersonic Propulsion": {"parent": "Propulsion", "depth": 2, "keywords": ["hypersonic", "scramjet", "ramjet", "air-breathing"]},

    # Under Guidance & Control
    "Inertial Navigation":   {"parent": "Guidance & Control", "depth": 2, "keywords": ["inertial navigation", "ins", "accelerometer", "gyroscope", "imu"]},
    "Attitude Control":      {"parent": "Guidance & Control", "depth": 2, "keywords": ["attitude control", "reaction wheel", "thruster", "momentum", "acs"]},
    "Terminal Guidance":     {"parent": "Guidance & Control", "depth": 2, "keywords": ["terminal guidance", "seeker", "homing", "target tracking"]},

    # Under Defence Systems
    "IGMDP":                 {"parent": "Defence Systems", "depth": 2, "keywords": ["igmdp", "integrated guided missile", "akash", "nag", "trishul"]},
    "Akash Missile":         {"parent": "Defence Systems", "depth": 2, "keywords": ["akash", "surface-to-air", "sam"]},
    "Hypersonic Weapons":    {"parent": "Defence Systems", "depth": 2, "keywords": ["hypersonic glide", "hgv", "hypersonic weapon", "shaurya"]},
    "Radar Systems":         {"parent": "Defence Systems", "depth": 2, "keywords": ["radar", "aesa", "phased array", "ew", "electronic warfare"]},

    # ── Depth 3: Systems (examples — extended at runtime) ─────────────────────
    "Vikas Engine":          {"parent": "Liquid Propulsion", "depth": 3, "keywords": ["vikas", "vikas engine", "ps2"]},
    "CE-7.5":                {"parent": "Cryogenic Propulsion", "depth": 3, "keywords": ["ce-7.5", "cryogenic engine", "ce20", "c25"]},
    "Ring Laser Gyroscope":  {"parent": "Inertial Navigation", "depth": 3, "keywords": ["ring laser gyro", "rlg", "fog", "fibre optic gyro"]},
    "HTPB Propellant":       {"parent": "Solid Propulsion", "depth": 3, "keywords": ["htpb", "solid fuel", "propellant grain"]},
}

# Precompile keyword patterns for speed
_COMPILED: Dict[str, re.Pattern] = {
    label: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in info["keywords"]) + r")\b",
        re.IGNORECASE,
    )
    for label, info in AEROSPACE_TAXONOMY.items()
}


def tag_document(
    title: str,
    abstract: str,
    section_texts: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Main entry point. Returns a list of taxonomy matches, sorted by depth
    (most specific first), each with:
      {
        "label":  str,    # taxonomy node label
        "depth":  int,    # 1–5+
        "parent": str,    # parent label
        "score":  float,  # match confidence (0–1)
        "path":   list,   # full path from root to this node
      }
    """
    corpus = f"{title} {abstract} " + (" ".join(section_texts) if section_texts else "")
    corpus = corpus[:8000]   # cap to avoid runaway matching

    scores: Dict[str, float] = {}

    for label, pattern in _COMPILED.items():
        matches = pattern.findall(corpus)
        if matches:
            # Score = unique keyword hit ratio, weighted by depth (deeper = stronger signal)
            depth = AEROSPACE_TAXONOMY[label]["depth"]
            scores[label] = (len(set(m.lower() for m in matches)) / len(AEROSPACE_TAXONOMY[label]["keywords"])) * depth

    if not scores:
        return [_unknown_tag()]

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for label, score in ranked[:5]:   # top 5 matches
        info = AEROSPACE_TAXONOMY[label]
        results.append({
            "label":  label,
            "depth":  info["depth"],
            "parent": info["parent"],
            "score":  round(min(score, 1.0), 3),
            "path":   _resolve_path(label),
        })

    return results


def get_primary_tag(title: str, abstract: str) -> Tuple[str, int, List[str]]:
    """
    Convenience wrapper. Returns (label, depth, taxonomy_path) for the best match.
    """
    tags = tag_document(title, abstract)
    if tags:
        best = tags[0]
        return best["label"], best["depth"], best["path"]
    return "Unknown", 1, ["Unknown"]


def _resolve_path(label: str) -> List[str]:
    """Walk the taxonomy tree upward to build the full path from root."""
    path = []
    current = label
    visited = set()
    while current and current not in visited:
        visited.add(current)
        path.insert(0, current)
        parent = AEROSPACE_TAXONOMY.get(current, {}).get("parent")
        current = parent
    return path


def _unknown_tag() -> Dict:
    return {
        "label":  "General Aerospace",
        "depth":  1,
        "parent": None,
        "score":  0.0,
        "path":   ["General Aerospace"],
    }
