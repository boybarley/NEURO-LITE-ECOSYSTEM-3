#!/usr/bin/env python3
"""
Neuro-Lite Knowledge Distiller (Developer Tool)
────────────────────────────────────────────────
Connects to a premium AI API to generate SOP-style Q&A knowledge.
Stores results into knowledge.db via FTS5.
Supports batch import with retry logic.

Usage:
    python3 tools/distill_knowledge.py --topic "Linux admin" --count 20
    python3 tools/distill_knowledge.py --file topics.txt
    python3 tools/distill_knowledge.py --interactive
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set up path so we can import from core/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "core"))

import httpx

from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("distill")

# Load config
_config_path = BASE_DIR / "config.env"
if _config_path.exists():
    with open(_config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v


DEFAULT_API_URL = os.environ.get("PREMIUM_API_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("PREMIUM_MODEL", "gpt-4o-mini")
DEFAULT_API_KEY = os.environ.get("PREMIUM_API_KEY", "")


def distill_topic(
    topic: str,
    count: int = 10,
    language: str = "english",
    api_key: str = "",
    api_url: str = "",
    model: str = "",
    max_retries: int = 3,
) -> List[Dict[str, str]]:
    """
    Generate Q&A pairs for a given topic using premium AI API.
    Returns list of dicts with keys: topic, question, answer.
    """
    api_key = api_key or DEFAULT_API_KEY
    api_url = api_url or DEFAULT_API_URL
    model = model or DEFAULT_MODEL

    if not api_key:
        logger.error("PREMIUM_API_KEY not set. Configure in config.env or pass --api-key.")
        return []

    prompt = f"""Generate exactly {count} question-and-answer pairs about: "{topic}".
Language: {language}.

Rules:
- Each Q&A must be self-contained and informative.
- Answers should be detailed (2-5 sentences), professional SOP style.
- Questions should be practical and commonly asked.
- Cover different aspects of the topic.
- Output ONLY a valid JSON array of objects with "question" and "answer" fields.

Example format:
[{{"question": "What is X?", "answer": "X is..."}}]"""

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Attempt %d/%d for topic '%s' (%d pairs)...", attempt, max_retries, topic, count)

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a knowledge generation assistant. Output only valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 4096,
                    },
                )

                if response.status_code != 200:
                    logger.warning("API returned %d: %s", response.status_code, response.text[:200])
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        logger.info("Retrying in %ds...", wait)
                        time.sleep(wait)
                        continue
                    return []

                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Strip markdown code fences if present
                if content.startswith("```"):
                    import re
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)

                pairs = json.loads(content)
                if not isinstance(pairs, list):
                    logger.warning("Expected JSON array, got %s", type(pairs).__name__)
                    continue

                # Enrich with topic
                for pair in pairs:
                    pair["topic"] = topic
                    pair["source"] = "ai_distill"

                logger.info("Generated %d Q&A pairs for '%s'", len(pairs), topic)
                return pairs

        except httpx.TimeoutException:
            logger.warning("Request timed out (attempt %d)", attempt)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response: %s", e)
            if attempt < max_retries:
                time.sleep(2)
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error("All %d attempts failed for topic '%s'", max_retries, topic)
    return []


def batch_distill(
    topics: List[str],
    count_per_topic: int = 10,
    language: str = "english",
    api_key: str = "",
) -> List[Dict[str, str]]:
    """Batch distill multiple topics."""
    all_pairs = []
    for i, topic in enumerate(topics, 1):
        logger.info("── Topic %d/%d: %s ──", i, len(topics), topic)
        pairs = distill_topic(
            topic=topic,
            count=count_per_topic,
            language=language,
            api_key=api_key,
        )
        all_pairs.extend(pairs)
        if i < len(topics):
            time.sleep(1)  # Rate limiting
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Neuro-Lite Knowledge Distiller")
    parser.add_argument("--topic", type=str, help="Single topic to distill")
    parser.add_argument("--count", type=int, default=10, help="Q&A pairs per topic (default: 10)")
    parser.add_argument("--language", type=str, default="english", help="Output language")
    parser.add_argument("--file", type=str, help="File with one topic per line")
    parser.add_argument("--api-key", type=str, default="", help="API key (overrides config)")
    parser.add_argument("--output", type=str, default="", help="Also save raw JSON to file")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't save to DB")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    topics = []
    if args.topic:
        topics = [args.topic]
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        topics = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    elif args.interactive:
        print("\n📚 Neuro-Lite Knowledge Distiller — Interactive Mode")
        print("Enter topics one per line. Empty line to start.\n")
        while True:
            line = input("Topic> ").strip()
            if not line:
                break
            topics.append(line)
    else:
        parser.print_help()
        sys.exit(1)

    if not topics:
        logger.error("No topics provided.")
        sys.exit(1)

    logger.info("Distilling %d topic(s), %d pairs each...", len(topics), args.count)

    all_pairs = batch_distill(
        topics=topics,
        count_per_topic=args.count,
        language=args.language,
        api_key=args.api_key,
    )

    if not all_pairs:
        logger.error("No knowledge generated.")
        sys.exit(1)

    logger.info("Total pairs generated: %d", len(all_pairs))

    # Save raw JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False))
        logger.info("Raw JSON saved to: %s", out_path)

    # Save to knowledge DB
    if not args.dry_run:
        rag = RAGEngine()
        count = rag.add_entries_batch(all_pairs)
        logger.info("✅ Saved %d entries to knowledge.db", count)
    else:
        logger.info("Dry run — not saved to DB. Preview:")
        for p in all_pairs[:3]:
            print(f"  Q: {p['question'][:80]}")
            print(f"  A: {p['answer'][:80]}")
            print()

    print(f"\n✅ Done. {len(all_pairs)} Q&A pairs processed.")


if __name__ == "__main__":
    main()
