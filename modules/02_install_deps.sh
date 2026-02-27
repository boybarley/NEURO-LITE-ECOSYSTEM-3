#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Module 02: Install Dependencies
# ═══════════════════════════════════════════════════════════
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────
PACKAGES=(
    python3
    python3-venv
    python3-pip
    python3-dev
    sqlite3
    git
    curl
    wget
    build-essential
    cmake
    cpufrequtils
)

log_info "Updating package index..."
apt-get update -qq > /dev/null 2>&1

INSTALLED=0
for pkg in "${PACKAGES[@]}"; do
    if dpkg -s "${pkg}" &> /dev/null; then
        continue
    fi
    log_info "Installing ${pkg}..."
    apt-get install -y -qq "${pkg}" > /dev/null 2>&1
    ((INSTALLED++))
done

if [[ ${INSTALLED} -gt 0 ]]; then
    log_ok "Installed ${INSTALLED} system packages."
else
    log_info "All system packages already present."
fi

# ── Python virtual environment ────────────────────────────
VENV_DIR="${NEURO_ROOT}/venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    log_info "Creating Python virtual environment..."
    sudo -u "${REAL_USER}" python3 -m venv "${VENV_DIR}"
    log_ok "Virtual environment created at ${VENV_DIR}"
else
    log_info "Virtual environment already exists."
fi

# ── Pip packages ──────────────────────────────────────────
REQUIREMENTS="${NEURO_ROOT}/core/requirements.txt"
if [[ ! -f "${REQUIREMENTS}" ]]; then
    log_err "requirements.txt not found at ${REQUIREMENTS}"
    exit 1
fi

log_info "Installing Python packages (this may take several minutes for llama-cpp-python)..."
sudo -u "${REAL_USER}" bash -c "
    source '${VENV_DIR}/bin/activate'
    pip install --upgrade pip setuptools wheel -q
    CMAKE_ARGS='-DGGML_BLAS=OFF -DGGML_AVX2=ON' pip install -r '${REQUIREMENTS}' -q
"
log_ok "Python packages installed."
