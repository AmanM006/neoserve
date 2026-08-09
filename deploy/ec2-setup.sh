#!/usr/bin/env bash
# NeoServe AWS Graviton4 provisioning + bootstrap.
#
# Subcommands:
#   provision   Launch a Graviton4 (c8g) spot instance from your laptop (needs AWS CLI).
#   bootstrap   Run ON the Arm host: install vLLM (aarch64) + quant + benchmark deps.
#   teardown    Terminate the provisioned instance.
#
# Cost control: uses spot by default; always run `teardown` when done. A c8g.4xlarge
# spot is typically well under $0.30/hr; a full mock->real sweep is low tens of $.
#
# Usage:
#   ./ec2-setup.sh provision --type c8g.4xlarge --key my-key --sg sg-xxxx --subnet subnet-xxxx
#   ssh ubuntu@<ip> 'bash -s' < ec2-setup.sh bootstrap     # or scp + run on host
#   ./ec2-setup.sh teardown
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STATE_FILE=".neoserve-ec2.json"
# Ubuntu 24.04 arm64 SSM parameter (resolves to a current AMI in the region).
AMI_SSM="/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id"

log() { printf '\033[1;32m[neoserve]\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------- #
provision() {
  local type="c8g.4xlarge" key="" sg="" subnet="" spot="true"
  while [[ $# -gt 0 ]]; do case "$1" in
    --type) type="$2"; shift 2;;
    --key) key="$2"; shift 2;;
    --sg) sg="$2"; shift 2;;
    --subnet) subnet="$2"; shift 2;;
    --on-demand) spot="false"; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac; done
  [[ -n "$key" && -n "$sg" ]] || { echo "need --key and --sg (and ideally --subnet)"; exit 1; }

  local ami; ami="$(aws ssm get-parameters --region "$REGION" --names "$AMI_SSM" \
      --query 'Parameters[0].Value' --output text)"
  log "AMI $ami  type $type  spot=$spot"

  local market=()
  [[ "$spot" == "true" ]] && market=(--instance-market-options '{"MarketType":"spot"}')
  local net=(); [[ -n "$subnet" ]] && net=(--subnet-id "$subnet")

  local id
  id="$(aws ec2 run-instances --region "$REGION" --image-id "$ami" \
      --instance-type "$type" --key-name "$key" --security-group-ids "$sg" \
      "${net[@]}" "${market[@]}" \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=project,Value=neoserve}]' \
      --query 'Instances[0].InstanceId' --output text)"
  log "launched $id; waiting for running ..."
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$id"
  local ip
  ip="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
  echo "{\"instance_id\":\"$id\",\"public_ip\":\"$ip\",\"type\":\"$type\",\"region\":\"$REGION\"}" > "$STATE_FILE"
  log "instance $id @ $ip  (state -> $STATE_FILE)"
  log "next: scp this repo over, then run: ./deploy/ec2-setup.sh bootstrap"
}

# --------------------------------------------------------------------------- #
bootstrap() {
  log "installing system deps (Ubuntu arm64)"
  sudo apt-get update -y
  sudo apt-get install -y python3-pip python3-venv build-essential cmake git \
      google-perftools libtcmalloc-minimal4 linux-tools-common numactl jq

  log "building mimalloc (best low-concurrency allocator win)"
  if [[ ! -f /usr/local/lib/libmimalloc.so ]]; then
    git clone --depth 1 https://github.com/microsoft/mimalloc /tmp/mimalloc || true
    cmake -S /tmp/mimalloc -B /tmp/mimalloc/out -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build /tmp/mimalloc/out -j >/dev/null
    sudo cp /tmp/mimalloc/out/libmimalloc.so* /usr/local/lib/ && sudo ldconfig
  fi

  log "python env + inference stack (aarch64 wheels)"
  python3 -m venv ~/neoserve-venv
  # shellcheck disable=SC1090
  source ~/neoserve-venv/bin/activate
  pip install --upgrade pip
  # vLLM aarch64 CPU backend (oneDNN + Arm Compute Library) now ships wheels.
  pip install "vllm>=0.11.0" "llmcompressor>=0.4.0" "transformers>=4.45" \
      "datasets>=3.0" "lm-eval>=0.4.5"
  # NeoServe harness deps
  pip install -r requirements.txt

  log "verifying Arm ISA features (expect: fp, asimd, asimddp/i8mm, sve; SME2 absent on Graviton4)"
  grep -m1 -o 'Features.*' /proc/cpuinfo || true

  cat <<'NOTE'

[neoserve] Optional but high-value: install Arm Performix (apx) for PMU top-down.
  See https://learn.arm.com/install-guides/performix/  (Apache-2.0: github.com/arm/performix)
  Then run the harness with:  python -m harness.runner --real --instance c8g.4xlarge

[neoserve] To grant PMU access for Performix/perf:
  echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
NOTE
  log "bootstrap complete. activate with: source ~/neoserve-venv/bin/activate"
}

# --------------------------------------------------------------------------- #
teardown() {
  [[ -f "$STATE_FILE" ]] || { echo "no $STATE_FILE"; exit 1; }
  local id; id="$(jq -r .instance_id "$STATE_FILE")"
  log "terminating $id"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$id" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$id"
  rm -f "$STATE_FILE"
  log "done."
}

cmd="${1:-}"; shift || true
case "$cmd" in
  provision) provision "$@";;
  bootstrap) bootstrap "$@";;
  teardown) teardown "$@";;
  *) echo "usage: $0 {provision|bootstrap|teardown} [args]"; exit 1;;
esac
