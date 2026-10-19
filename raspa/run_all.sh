COF_ROOT=${COF_ROOT:-$(pwd)}  # run from the repository root or export COF_ROOT
cd $COF_ROOT/raspa
# wait for charges of each structure as the queue reaches it
cat widom/order.txt | xargs -P 2 -I{} bash -c 'while [ ! -f $COF_ROOT/eqeq/out/{}.json ] && false; do sleep 15; done; ./run_one.sh {}'
echo WIDOM_DONE
