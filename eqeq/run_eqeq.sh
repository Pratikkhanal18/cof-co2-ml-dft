COF_ROOT=${COF_ROOT:-$(pwd)}  # run from the repository root or export COF_ROOT
cd $COF_ROOT/eqeq
for cid in $(tail -n +2 $COF_ROOT/data/CURATED-COFs/cof-frameworks.csv | cut -d, -f1 | grep -v -E "^(21691N2|20480N3|20481N3)$"); do
  [ -f out/$cid.json ] && continue
  mkdir -p tmp_$cid out; cp $COF_ROOT/data/CURATED-COFs/cifs/$cid.cif ionizationdata.dat chargecenters.dat tmp_$cid/
  (cd tmp_$cid && bash -c "ulimit -v 2500000; timeout 1800 eqeq $cid.cif" > log 2>&1; mv $cid.cif_EQeq_ewald_1.20_-2.00.json ../out/$cid.json 2>/dev/null; mv $cid.cif_EQeq_ewald_1.20_-2.00.cif ../out/$cid.cif 2>/dev/null)
  rm -rf tmp_$cid
done
echo EQEQ_DONE
