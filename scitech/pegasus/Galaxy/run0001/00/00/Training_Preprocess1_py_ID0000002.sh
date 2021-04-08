#!/bin/bash
set -e
pegasus_lite_version_major="5"
pegasus_lite_version_minor="0"
pegasus_lite_version_patch="0dev"
pegasus_lite_enforce_strict_wp_check="true"
pegasus_lite_version_allow_wp_auto_download="true"


. pegasus-lite-common.sh

pegasus_lite_init

# cleanup in case of failures
trap pegasus_lite_signal_int INT
trap pegasus_lite_signal_term TERM
trap pegasus_lite_unexpected_exit EXIT

printf "\n########################[Pegasus Lite] Setting up workdir ########################\n"  1>&2
# work dir
export pegasus_lite_work_dir=$PWD
pegasus_lite_setup_work_dir

printf "\n##############[Pegasus Lite] Figuring out the worker package to use ##############\n"  1>&2
# figure out the worker package to use
pegasus_lite_worker_package

pegasus_lite_section_start stage_in
printf "\n##################### Setting the xbit for executables staged #####################\n"  1>&2
# set the xbit for any executables staged
if [ ! -x Training_Preprocess1_py ]; then
    /bin/chmod +x Training_Preprocess1_py
fi

printf "\n##################### Checking file integrity for input files #####################\n"  1>&2
# do file integrity checks
pegasus-integrity --print-timings --verify=stdin 1>&2 << 'eof'
Class3_2.jpg:Class1_3.jpg:Training_Preprocess1_py:Class1_2.jpg:Class0_0.jpg:Class2_3.jpg:Class4_0.jpg:Class0_2.jpg:Class1_1.jpg:Class3_0.jpg:Class2_4.jpg:Class4_3.jpg:Class4_1.jpg:Class0_1.jpg:Class2_0.jpg
eof

pegasus_lite_section_end stage_in
set +e
job_ec=0
pegasus_lite_section_start task_execute
printf "\n######################[Pegasus Lite] Executing the user task ######################\n"  1>&2
pegasus-kickstart  -n Training_Preprocess1.py -N ID0000002 -R condorpool  -s Training_labels_Preprocess1.npy=Training_labels_Preprocess1.npy -s Training_images_Preprocess1.npy=Training_images_Preprocess1.npy -L Galaxy -T 2021-04-08T19:14:50+00:00 ./Training_Preprocess1_py 
job_ec=$?
pegasus_lite_section_end task_execute
set -e
pegasus_lite_section_start stage_out
pegasus_lite_section_end stage_out

set -e


# clear the trap, and exit cleanly
trap - EXIT
pegasus_lite_final_exit

