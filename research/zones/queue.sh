#!/bin/zsh
# usage: queue.sh ZONE:VARIANT ...   (runs sequentially, waits for data)
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
for job in "$@"; do
  z=${job%%:*}; v=${job##*:}
  until $PY -I ready.py $z; do sleep 60; done
  echo "$(date +%T) start $z $v"
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} $PY -I bt.py $z $v ${MAXIT:-300} >| ../data/zones/out/bt_${z}_${v}.log 2>&1
  echo "$(date +%T) end $z $v rc=$? $(tail -1 ../data/zones/out/bt_${z}_${v}.log)"
done
