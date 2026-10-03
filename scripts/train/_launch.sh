#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/data.sh"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/services.sh"

configure_environment
configure_data
configure_reward_services
check_launch_config

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
    check_training_inputs
    check_sandbox
    check_model
    prepare_output_directories

    trap cleanup_local_stem_verifier EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    ensure_stem_verifier
fi

build_train_command "$@"
print_launch_summary
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    exec "${train_command[@]}"
fi
"${train_command[@]}" 2>&1 | tee "${LOG_FILE}"
