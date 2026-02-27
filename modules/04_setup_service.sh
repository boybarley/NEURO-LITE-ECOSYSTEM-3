#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Module 04: systemd Service Setup
# ═══════════════════════════════════════════════════════════
set -euo pipefail

source "${NEURO_ROOT}/config.env"

SERVICE_NAME="neurolite"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_PYTHON="${NEURO_ROOT}/venv/bin/python3"
SERVER_SCRIPT="${NEURO_ROOT}/core/main_server.py"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Neuro-Lite AI Engine v1.0
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${REAL_USER}
Group=${REAL_USER}
WorkingDirectory=${NEURO_ROOT}
Environment="PATH=${NEURO_ROOT}/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=${NEURO_ROOT}/config.env
ExecStart=${VENV_PYTHON} ${SERVER_SCRIPT}
Restart=always
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5
StandardOutput=append:${NEURO_ROOT}/logs/neurolite.log
StandardError=append:${NEURO_ROOT}/logs/neurolite.log

# ── Resource Guards ───────────────────────────────────────
MemoryMax=3G
CPUQuota=90%
LimitNOFILE=65536

# ── Security Hardening ────────────────────────────────────
ProtectSystem=strict
ReadWritePaths=${NEURO_ROOT}/data ${NEURO_ROOT}/logs ${NEURO_ROOT}/models
PrivateTmp=true
NoNewPrivileges=true

# ── Graceful Shutdown ─────────────────────────────────────
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" > /dev/null 2>&1
systemctl restart "${SERVICE_NAME}"

sleep 2

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    log_ok "Service '${SERVICE_NAME}' is running."
else
    log_warn "Service may need model download to complete first."
    log_info "Check: sudo journalctl -u ${SERVICE_NAME} -f"
fi
