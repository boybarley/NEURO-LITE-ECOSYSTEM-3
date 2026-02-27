#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Module 01: OS Tuning for 4GB RAM constraint
# ═══════════════════════════════════════════════════════════
set -euo pipefail

# ── Swappiness ────────────────────────────────────────────
CURRENT_SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [[ "${CURRENT_SWAPPINESS}" -ne 10 ]]; then
    sysctl -w vm.swappiness=10 > /dev/null
    if ! grep -q "^vm.swappiness" /etc/sysctl.d/99-neurolite.conf 2>/dev/null; then
        echo "vm.swappiness=10" >> /etc/sysctl.d/99-neurolite.conf
    else
        sed -i 's/^vm.swappiness=.*/vm.swappiness=10/' /etc/sysctl.d/99-neurolite.conf
    fi
    log_ok "vm.swappiness set to 10 (was ${CURRENT_SWAPPINESS})"
else
    log_info "vm.swappiness already 10."
fi

# ── VFS cache pressure ───────────────────────────────────
sysctl -w vm.vfs_cache_pressure=50 > /dev/null
if ! grep -q "^vm.vfs_cache_pressure" /etc/sysctl.d/99-neurolite.conf 2>/dev/null; then
    echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.d/99-neurolite.conf
fi
log_ok "vfs_cache_pressure set to 50"

# ── Dirty ratio tuning (reduce write pressure) ───────────
sysctl -w vm.dirty_ratio=10 > /dev/null
sysctl -w vm.dirty_background_ratio=5 > /dev/null
grep -q "^vm.dirty_ratio" /etc/sysctl.d/99-neurolite.conf 2>/dev/null || \
    echo "vm.dirty_ratio=10" >> /etc/sysctl.d/99-neurolite.conf
grep -q "^vm.dirty_background_ratio" /etc/sysctl.d/99-neurolite.conf 2>/dev/null || \
    echo "vm.dirty_background_ratio=5" >> /etc/sysctl.d/99-neurolite.conf
log_ok "Dirty ratios tuned for low-RAM"

# ── CPU Governor ──────────────────────────────────────────
if command -v cpufreq-set &> /dev/null; then
    NCPUS=$(nproc)
    for ((i=0; i<NCPUS; i++)); do
        cpufreq-set -c "$i" -g performance 2>/dev/null || true
    done
    log_ok "CPU governor set to performance (${NCPUS} cores)"
elif [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
    for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -w "${gov_file}" ]]; then
            echo "performance" > "${gov_file}" 2>/dev/null || true
        fi
    done
    log_ok "CPU governor set to performance via sysfs"
else
    log_warn "CPU frequency scaling not available — skipping governor."
fi

# ── Transparent Hugepages (detect, disable if problematic) ─
THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"
if [[ -f "${THP_PATH}" ]]; then
    CURRENT_THP=$(cat "${THP_PATH}")
    if echo "${CURRENT_THP}" | grep -q '\[always\]'; then
        echo "madvise" > "${THP_PATH}" 2>/dev/null || true
        log_ok "Transparent hugepages set to madvise (was always)"
    else
        log_info "Transparent hugepages already safe: ${CURRENT_THP}"
    fi
else
    log_info "Transparent hugepages sysfs not found — skipping."
fi

# ── Swap Creation (2GB if absent) ─────────────────────────
SWAP_COUNT=$(swapon --show --noheadings | wc -l)
if [[ "${SWAP_COUNT}" -eq 0 ]]; then
    SWAPFILE="/swapfile_neurolite"
    if [[ ! -f "${SWAPFILE}" ]]; then
        log_info "Creating 2GB swap at ${SWAPFILE}..."
        dd if=/dev/zero of="${SWAPFILE}" bs=1M count=2048 status=progress 2>/dev/null
        chmod 600 "${SWAPFILE}"
        mkswap "${SWAPFILE}" > /dev/null
        log_ok "Swap file created."
    fi
    swapon "${SWAPFILE}"
    if ! grep -q "${SWAPFILE}" /etc/fstab; then
        echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    fi
    log_ok "2GB swap activated and persisted."
else
    SWAP_SIZE=$(free -m | awk '/^Swap:/ {print $2}')
    log_info "Swap already active: ${SWAP_SIZE}MB — skipping creation."
fi

log_ok "OS tuning complete."
