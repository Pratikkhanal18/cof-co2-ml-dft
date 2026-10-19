#!/bin/bash
COF_ROOT=${COF_ROOT:-$(pwd)}  # run from the repository root or export COF_ROOT
cid=$1; base=$COF_ROOT/raspa/widom
[ -f $base/done/$cid.data ] && exit 0
[ -f $COF_ROOT/eqeq/out/$cid.json ] || { echo "$cid nocharge" >> $base/skipped.txt; exit 0; }
mkdir -p $base/done
d=$(python3 $COF_ROOT/raspa/make_job.py $cid $base/run 1500) || exit 0
cd $d && timeout 7200 simulate simulation.input > run.log 2>&1
cp Output/System_0/*.data $base/done/$cid.data 2>/dev/null || echo "$cid fail" >> $base/skipped.txt
cd $base && rm -rf $d
