COF_ROOT=${COF_ROOT:-$(pwd)}  # run from the repository root or export COF_ROOT
cd $COF_ROOT/zeo
for cid in $(cat $COF_ROOT/raspa/widom/order.txt); do
  [ -f $cid.sa ] && continue
  cp $COF_ROOT/data/CURATED-COFs/cifs/$cid.cif .
  timeout 1200 network -ha -res $cid.res -sa 1.65 1.65 2000 $cid.sa -vol 1.3 1.3 20000 $cid.vol $cid.cif > /dev/null 2>&1
  rm -f $cid.cif
done
echo ZEO_DONE
