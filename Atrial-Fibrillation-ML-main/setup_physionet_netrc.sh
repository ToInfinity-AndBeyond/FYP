#!/usr/bin/env bash
set -euo pipefail
read -rp "PhysioNet username [mc1920]: " user
user="${user:-mc1920}"
read -rsp "PhysioNet password: " pass
echo
cat > "${HOME}/.netrc" <<NETRC
machine physionet.org
login ${user}
password ${pass}
NETRC
chmod 600 "${HOME}/.netrc"
echo "Wrote ${HOME}/.netrc"
