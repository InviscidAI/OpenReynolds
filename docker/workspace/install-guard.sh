#!/bin/sh
tool=$(basename "$0")
blocked=0
case "$tool" in
  pip|pip3)
    case "$1" in install|download|wheel) blocked=1 ;; esac ;;
  apt|apt-get)
    for a in "$@"; do
      case "$a" in install|update|upgrade|full-upgrade|download|build-dep) blocked=1 ;; esac
    done ;;
esac
if [ "$blocked" = 1 ]; then
  echo "reynolds: '$tool $*' needs the network, which is sealed in this sandbox." >&2
  echo "What you want is almost certainly already here -- see /work/.toolbox/ENVIRONMENT.md" >&2
  echo "(PDF -> pdftoppm/pdftotext; gifs/mp4 -> imageio[ffmpeg]; meshing -> gmsh; solvers, HiSA -- all on PATH)." >&2
  exit 1
fi
case "$tool" in
  pip|pip3) exec python3 -m pip "$@" ;;
  *)        exec "/usr/bin/$tool" "$@" ;;
esac
