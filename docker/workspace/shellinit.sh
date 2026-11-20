# OpenFOAM's environment and its RunFunctions (runApplication, runParallel,
# restore0Dir, ...) for every bash. Read by /etc/profile.d/foamd.sh in login shells
# and through $BASH_ENV in any other. Idempotent and cheap: the bashrc costs ~180 ms,
# so it is sourced only when a parent has not already done it.
if [ -z "${WM_PROJECT_DIR:-}" ]; then
  source /openfoam/bashrc 2>/dev/null || source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || true
fi
{ [ -r "${WM_PROJECT_DIR:-}/bin/tools/RunFunctions" ] && source "${WM_PROJECT_DIR}/bin/tools/RunFunctions"; } 2>/dev/null || true
