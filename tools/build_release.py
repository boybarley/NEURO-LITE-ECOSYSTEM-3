#!/usr/bin/env python3
"""
Neuro-Lite Release Builder (Developer Tool)
────────────────────────────────────────────
Creates a deployable release artifact as tar.gz.
Validates all required files exist before packaging.

Usage:
    python3 tools/build_release.py
    python3 tools/build_release.py --output releases/neurolite-v1.0.tar.gz
    python3 tools/build_release.py --include-model
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_release")

BASE_DIR = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════
# Required Files
# ═══════════════════════════════════════════════════════════

REQUIRED_FILES = [
    "config.env",
    "install.sh",
    "modules/01_os_tuning.sh",
    "modules/02_install_deps.sh",
    "modules/03_download_model.sh",
    "modules/04_setup_service.sh",
    "core/main_server.py",
    "core/llm_engine.py",
    "core/context_manager.py",
    "core/post_processor.py",
    "core/rag_engine.py",
    "core/emotional_state.py",
    "core/requirements.txt",
    "webui/index.html",
    "webui/admin.html",
    "tools/distill_knowledge.py",
    "tools/validate_data.py",
    "tools/build_release.py",
]

OPTIONAL_FILES = [
    "data/knowledge.db",
    "README.md",
    "LICENSE",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    "venv",
    "*.pyc",
    ".env",
    "*.tmp",
    "*.log",
    "logs/",
    "models/*.gguf",
    "*.tar.gz",
    "releases/",
]


def validate_required_files() -> bool:
    """Ensure all required files exist."""
    missing = []
    for f in REQUIRED_FILES:
        if not (BASE_DIR / f).exists():
            missing.append(f)

    if missing:
        logger.error("Missing required files:")
        for f in missing:
            logger.error("  ❌ %s", f)
        return False

    logger.info("✅ All %d required files present.", len(REQUIRED_FILES))
    return True


def _should_exclude(name: str) -> bool:
    """Check if a path should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            if pattern.rstrip("/") in name:
                return True
        elif "*" in pattern:
            import fnmatch
            if fnmatch.fnmatch(os.path.basename(name), pattern):
                return True
        else:
            if pattern in name:
                return True
    return False


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release(
    output_path: str = "",
    include_model: bool = False,
    include_knowledge: bool = True,
) -> Optional[Path]:
    """Build the release tar.gz artifact."""

    # Validate
    if not validate_required_files():
        logger.error("Build aborted due to missing files.")
        return None

    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not output_path:
        releases_dir = BASE_DIR / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(releases_dir / f"neurolite-v1.0-{timestamp}.tar.gz")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building release artifact: %s", output.name)

    manifest = {
        "version": "1.0.0",
        "built_at": datetime.utcnow().isoformat(),
        "files": [],
    }

    with tarfile.open(str(output), "w:gz", compresslevel=6) as tar:
        file_count = 0

        for root, dirs, files in os.walk(str(BASE_DIR)):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not _should_exclude(d)]

            for filename in files:
                filepath = Path(root) / filename
                relative = filepath.relative_to(BASE_DIR)
                rel_str = str(relative)

                if _should_exclude(rel_str):
                    continue

                # Skip model files unless --include-model
                if rel_str.startswith("models/") and rel_str.endswith(".gguf"):
                    if not include_model:
                        continue

                # Skip knowledge DB unless requested
                if rel_str == "data/knowledge.db" and not include_knowledge:
                    continue

                arcname = f"neuro-lite/{rel_str}"
                tar.add(str(filepath), arcname=arcname)
                file_count += 1

                manifest["files"].append({
                    "path": rel_str,
                    "size": filepath.stat().st_size,
                    "sha256": compute_sha256(filepath),
                })

        # Add manifest to archive
        manifest["total_files"] = file_count
        manifest_json = json.dumps(manifest, indent=2).encode()
        import io
        info = tarfile.TarInfo(name="neuro-lite/MANIFEST.json")
        info.size = len(manifest_json)
        info.mtime = time.time()
        tar.addfile(info, io.BytesIO(manifest_json))

    output_size = output.stat().st_size
    logger.info(
        "✅ Release built: %s (%d files, %.1f MB)",
        output.name,
        file_count,
        output_size / 1048576,
    )

    # Write manifest alongside
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest: %s", manifest_path.name)

    return output


def main():
    parser = argparse.ArgumentParser(description="Neuro-Lite Release Builder")
    parser.add_argument("--output", type=str, default="", help="Output tar.gz path")
    parser.add_argument("--include-model", action="store_true", help="Include GGUF model in release (large!)")
    parser.add_argument("--no-knowledge", action="store_true", help="Exclude knowledge.db")
    parser.add_argument("--validate-only", action="store_true", help="Only validate files, don't build")

    args = parser.parse_args()

    print("\n🔧 Neuro-Lite Release Builder")
    print("=" * 40)

    if args.validate_only:
        ok = validate_required_files()
        sys.exit(0 if ok else 1)

    result = build_release(
        output_path=args.output,
        include_model=args.include_model,
        include_knowledge=not args.no_knowledge,
    )

    if result:
        print(f"\n✅ Release artifact ready: {result}")
        print(f"   Deploy with: tar xzf {result.name} && cd neuro-lite && sudo bash install.sh")
    else:
        print("\n❌ Build failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
