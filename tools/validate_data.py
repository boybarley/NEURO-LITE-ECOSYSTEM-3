#!/usr/bin/env python3
"""
Neuro-Lite Data Validator (Developer Tool)
──────────────────────────────────────────
Scans crowdsourced data for:
  - Toxic language
  - PII patterns (email, phone, SSN, etc.)
  - Duplicate entries
  - Malformed entries
Zero-Trust ingestion philosophy.

Usage:
    python3 tools/validate_data.py --file data.json
    python3 tools/validate_data.py --scan-db
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "core"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validator")

# ═══════════════════════════════════════════════════════════
# Toxic Language Patterns
# ═══════════════════════════════════════════════════════════

TOXIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(fuck|shit|bitch|asshole|bastard|dick|cunt|damn|crap|piss)\b",
        r"\b(nigger|nigga|faggot|retard|tranny|chink|spic|kike|wetback)\b",
        r"\b(kill\s+(yourself|himself|herself|themselves|your ?self))\b",
        r"\b(suicide|self[- ]?harm|cut\s+(yourself|myself))\b",
        r"\b(bomb\s+making|how\s+to\s+make\s+a\s+bomb|synthesize\s+drugs?)\b",
        r"\b(child\s+porn|cp\b|csam|pedo(phile)?)\b",
        r"\b(hack\s+into|steal\s+password|phishing\s+attack)\b",
        r"\b(white\s+supremac|nazi|holocaust\s+denial|ethnic\s+cleansing)\b",
    ]
]

# ═══════════════════════════════════════════════════════════
# PII Patterns
# ═══════════════════════════════════════════════════════════

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)[2-9]\d{2}[-.\s]?\d{4}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\d{4,14}"),
    "ssn": re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "us_address": re.compile(r"\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)\b", re.IGNORECASE),
}

# ═══════════════════════════════════════════════════════════
# Validator Class
# ═══════════════════════════════════════════════════════════

class DataValidator:
    def __init__(self):
        self.issues: List[Dict] = []
        self.fingerprints: Set[str] = set()
        self.stats = {
            "total": 0,
            "passed": 0,
            "toxic": 0,
            "pii": 0,
            "duplicate": 0,
            "malformed": 0,
            "empty": 0,
        }

    def _fingerprint(self, text: str) -> str:
        """Generate normalized fingerprint for dedup."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _check_toxic(self, text: str) -> List[str]:
        """Check for toxic language patterns."""
        found = []
        for pattern in TOXIC_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found.extend(matches)
        return found

    def _check_pii(self, text: str) -> List[Tuple[str, str]]:
        """Check for PII patterns. Returns list of (type, match)."""
        found = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                found.append((pii_type, match))
        return found

    def validate_entry(self, entry: Dict, index: int) -> bool:
        """
        Validate a single data entry.
        Returns True if entry passes all checks.
        """
        self.stats["total"] += 1

        # Check required fields
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()

        if not question or not answer:
            self.stats["malformed"] += 1
            self.issues.append({
                "index": index,
                "type": "malformed",
                "detail": "Missing question or answer field",
                "severity": "error",
            })
            return False

        if len(question) < 5 or len(answer) < 5:
            self.stats["empty"] += 1
            self.issues.append({
                "index": index,
                "type": "empty",
                "detail": f"Content too short (q={len(question)}, a={len(answer)} chars)",
                "severity": "warning",
            })
            return False

        combined = question + " " + answer

        # Check toxic content
        toxic_matches = self._check_toxic(combined)
        if toxic_matches:
            self.stats["toxic"] += 1
            self.issues.append({
                "index": index,
                "type": "toxic",
                "detail": f"Toxic content detected: {toxic_matches[:3]}",
                "severity": "error",
                "question_preview": question[:60],
            })
            return False

        # Check PII
        pii_matches = self._check_pii(combined)
        if pii_matches:
            self.stats["pii"] += 1
            types = list(set(t for t, _ in pii_matches))
            self.issues.append({
                "index": index,
                "type": "pii",
                "detail": f"PII detected: {types}",
                "severity": "error",
                "question_preview": question[:60],
            })
            return False

        # Check duplicates
        fp = self._fingerprint(question)
        if fp in self.fingerprints:
            self.stats["duplicate"] += 1
            self.issues.append({
                "index": index,
                "type": "duplicate",
                "detail": "Duplicate question detected",
                "severity": "warning",
                "question_preview": question[:60],
            })
            return False
        self.fingerprints.add(fp)

        self.stats["passed"] += 1
        return True

    def validate_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a JSON file of entries.
        Returns (valid_entries, rejected_entries).
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            return [], []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON: %s", e)
            return [], []

        entries = data if isinstance(data, list) else data.get("entries", [])
        if not entries:
            logger.warning("No entries found in file.")
            return [], []

        valid = []
        rejected = []

        for i, entry in enumerate(entries):
            if self.validate_entry(entry, i):
                valid.append(entry)
            else:
                rejected.append(entry)

        return valid, rejected

    def validate_db(self) -> None:
        """Scan existing knowledge.db for issues."""
        from rag_engine import RAGEngine
        rag = RAGEngine()
        entries = rag.export_all()

        logger.info("Scanning %d entries in knowledge.db...", len(entries))

        for i, entry in enumerate(entries):
            self.validate_entry(entry, i)

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("  NEURO-LITE DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"  Total entries scanned:  {self.stats['total']}")
        print(f"  ✅ Passed:              {self.stats['passed']}")
        print(f"  ❌ Toxic content:       {self.stats['toxic']}")
        print(f"  🔒 PII detected:        {self.stats['pii']}")
        print(f"  🔄 Duplicates:          {self.stats['duplicate']}")
        print(f"  ⚠️  Malformed:           {self.stats['malformed']}")
        print(f"  📭 Empty/short:         {self.stats['empty']}")
        print("=" * 60)

        if self.issues:
            print("\nISSUES FOUND:")
            for issue in self.issues[:50]:  # Limit output
                icon = {"error": "❌", "warning": "⚠️"}.get(issue["severity"], "ℹ️")
                preview = issue.get("question_preview", "")
                print(
                    f"  {icon} [{issue['type']:>10}] idx={issue['index']:>4} | "
                    f"{issue['detail'][:60]} | {preview}"
                )
            if len(self.issues) > 50:
                print(f"\n  ... and {len(self.issues) - 50} more issues.")

        passed_pct = (
            (self.stats["passed"] / self.stats["total"] * 100)
            if self.stats["total"] > 0
            else 0
        )
        print(f"\nPASS RATE: {passed_pct:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Neuro-Lite Data Validator")
    parser.add_argument("--file", type=str, help="JSON file to validate")
    parser.add_argument("--scan-db", action="store_true", help="Scan existing knowledge.db")
    parser.add_argument("--output-clean", type=str, help="Save clean (passed) entries to JSON file")

    args = parser.parse_args()

    if not args.file and not args.scan_db:
        parser.print_help()
        sys.exit(1)

    validator = DataValidator()

    if args.scan_db:
        validator.validate_db()
        validator.print_report()
    elif args.file:
        valid, rejected = validator.validate_file(args.file)
        validator.print_report()

        if args.output_clean and valid:
            out = Path(args.output_clean)
            out.write_text(
                json.dumps(
                    {"entries": valid, "count": len(valid)},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            logger.info("Clean data saved to: %s (%d entries)", out, len(valid))

        if valid:
            answer = input(f"\nImport {len(valid)} valid entries to knowledge.db? [y/N]: ").strip().lower()
            if answer == "y":
                from rag_engine import RAGEngine
                rag = RAGEngine()
                count = rag.add_entries_batch(valid)
                logger.info("✅ Imported %d entries to knowledge.db", count)


if __name__ == "__main__":
    main()
