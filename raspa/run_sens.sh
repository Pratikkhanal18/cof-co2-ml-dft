#!/bin/bash
ROOT=$(realpath ${COF_ROOT:-.})
cid=$1; v=$2; cyc=$3; out=$ROOT/raspa/sens/done_$v; mkdir -p $out
[ -f $out/$cid.data ] && exit 0
d=$(python3 $ROOT/raspa/sens_job.py $cid $v $cyc) || exit 0
cd $d && s=$(date +%s) && timeout 20000 simulate simulation.input > run.log 2>&1
cp Output/System_0/*.data $out/$cid.data 2>/dev/null && echo "$cid $v $(( $(date +%s)-s ))s" >> $ROOT/raspa/sens/times.txt || echo "$cid $v fail" >> $ROOT/raspa/sens/times.txt
cd / && rm -rf $d
