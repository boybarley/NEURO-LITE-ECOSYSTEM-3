#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Module 03: Download GGUF Model
# ═══════════════════════════════════════════════════════════
set -euo pipefail

source "${NEURO_ROOT}/config.env"

MODEL_DIR="${NEURO_ROOT}/models"
MODEL_FILENAME="qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_FULL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"

MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/${MODEL_FILENAME}"
FALLBACK_URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

EXPECTED_SIZE_MIN=1800000000  # ~1.8GB minimum expected

# ── Skip if exists and valid ──────────────────────────────
if [[ -f "${MODEL_FULL_PATH}" ]]; then
    FILE_SIZE=$(stat -c%s "${MODEL_FULL_PATH}" 2>/dev/null || echo "0")
    if [[ ${FILE_SIZE} -ge ${EXPECTED_SIZE_MIN} ]]; then
        log_info "Model already exists and size is valid ($(( FILE_SIZE / 1048576 )) MB). Skipping."
        exit 0
    else
        log_warn "Model file exists but seems incomplete (${FILE_SIZE} bytes). Re-downloading."
    fi
fi

# ── Download with resume support ──────────────────────────
download_model() {
    local url="$1"
    local dest="$2"

    log_info "Downloading from: ${url}"
    log_info "This will download ~1.9GB. Please be patient..."

    if wget \
        --continue \
        --progress=bar:force:noscroll \
        --timeout=60 \
        --tries=5 \
        --retry-connrefused \
        -O "${dest}" \
        "${url}"; then
        return 0
    fi
    return 1
}

mkdir -p "${MODEL_DIR}"

if download_model "${MODEL_URL}" "${MODEL_FULL_PATH}"; then
    log_ok "Primary download succeeded."
elif download_model "${FALLBACK_URL}" "${MODEL_FULL_PATH}"; then
    log_ok "Fallback download succeeded."
else
    log_err "All download attempts failed."
    log_err "Please download the model manually:"
    log_err "  wget -O '${MODEL_FULL_PATH}' '${MODEL_URL}'"
    exit 1
fi

# ── Verify file size ──────────────────────────────────────
FINAL_SIZE=$(stat -c%s "${MODEL_FULL_PATH}" 2>/dev/null || echo "0")
if [[ ${FINAL_SIZE} -lt ${EXPECTED_SIZE_MIN} ]]; then
    log_err "Downloaded file is too small (${FINAL_SIZE} bytes). May be corrupt."
    exit 1
fi

chown "${REAL_USER}:${REAL_USER}" "${MODEL_FULL_PATH}"
log_ok "Model downloaded: ${MODEL_FILENAME} ($(( FINAL_SIZE / 1048576 )) MB)"
