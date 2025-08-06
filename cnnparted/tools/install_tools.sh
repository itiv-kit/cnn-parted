make -C cnnparted/tools/cacti
#make -j -C cnnparted/tools/ramulator
#make -j -C cnnparted/tools/VAMPIRE
git -C cnnparted/tools/MNSIM-2.0 apply ../mnsim.patch || true

#source cnnparted/tools/setupTimeloop.sh