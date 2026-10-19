#!/bin/bash
COF_ROOT=${COF_ROOT:-$(pwd)}  # run from the repository root or export COF_ROOT
# usage: run_gcmc.sh CID T P tag
cid=$1; T=$2; P=$3; tag=$4; base=$COF_ROOT/raspa/gcmc
mkdir -p $base/out
[ -f $base/out/${cid}_${tag}.data ] && exit 0
d=$base/run_${cid}_${tag}; rm -rf $d; mkdir -p $d
python3 - <<PY
import sys; sys.path.insert(0,'$COF_ROOT/raspa')
from make_gcmc import *
d,uc=prep('$cid','$d')
open(d+'/he.input','w').write(he_input('$cid',uc))
open(d+'/uc','w').write(' '.join(map(str,uc)))
PY
cd $d/$cid
if [ ! -f $base/out/${cid}_He.txt ]; then
  mkdir he && cp *.def *.cif he/ && cd he && cp ../he.input simulation.input && simulate simulation.input >/dev/null 2>&1
  grep "\[helium\] Average Widom Rosenbluth-weight" Output/System_0/*.data | awk '{print $5}' > $base/out/${cid}_He.txt; cd ..
fi
vf=$(cat $base/out/${cid}_He.txt)
python3 -c "
import sys; sys.path.insert(0,'$COF_ROOT/raspa')
from make_gcmc import gcmc_input
uc=open('uc').read().split()
open('simulation.input','w').write(gcmc_input('$cid',uc,$T,$P,$vf))"
timeout 20000 simulate simulation.input > log 2>&1
cp Output/System_0/*.data $base/out/${cid}_${tag}.data
cd $base && rm -rf $d
