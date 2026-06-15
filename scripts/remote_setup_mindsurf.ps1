param(
    [string]$HostAlias = "mindsurf",
    [string]$RemoteRoot = "/home/oscar/global-campus-ai-algorithm"
)

$ErrorActionPreference = "Stop"

$remoteCommand = @"
set -e
user=`$(whoami)
if [ "`$user" != "oscar" ]; then
  echo "Refusing to run outside oscar account. Current user: `$user" >&2
  exit 2
fi
mkdir -p "$RemoteRoot"
cd "$RemoteRoot"
echo "user=`$user"
echo "project_dir=`$(pwd)"
if [ -d /home/oscar/minimind ]; then
  echo "minimind_dir=/home/oscar/minimind"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"@

ssh $HostAlias $remoteCommand
