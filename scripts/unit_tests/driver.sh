#!/bin/bash

# Function to display help information
function display_help {
    echo "Usage:"
    echo "  $0 binary_name all <log_dir>                     # Runs all standard tests"
    echo "  $0 binary_name custom <log_dir> <ranks> <filter> # Runs custom test configuration"
    echo
    echo "Arguments:"
    echo "  binary_name: Name of the binary to run."
    echo "  all: Executes predefined test configurations."
    echo "  custom: Executes a test with custom MPI ranks and GTest filter."
    echo "  ranks: Number of MPI ranks (required for custom mode)."
    echo "  filter: GTest filter string (required for custom mode)."
    echo "  log_dir: The directory where the logs will be placed (Optional)"
    echo
}

# Validate number of arguments for each mode
if { [[ "$2" != "all" ]]    && [[ "$mode" != "custom" ]]; } ||
   { [[ "$2" == "all" ]]    && [[ "$#" -lt 2 ]]; } ||
   { [[ "$2" == "custom" ]] && [[ "$#" -lt 4 ]]; }
then
  display_help
  exit 1
fi

driver_return_status=0
binary_name=$1
mode=$2
log_dir=$3
ranks=$4
filter=$5
timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
mpi_timeout=$((20 * 60)) # 20 minutes in seconds

# Optional Log Directory is not set
if { [[ "$2" == "all" ]]    && [[ "$#" -eq 2 ]]; } ||
   { [[ "$2" == "custom" ]] && [[ "$#" -eq 4 ]]; }
then
  log_dir=$(pwd)
  ranks=$3
  filter=$4
fi

log_file="$log_dir/unit_tests_${timestamp}.log"

# Function to execute mpirun command
function run_mpirun {
    local np=$1
    local gtest_filter=$2
    cmd_str="mpirun -np $np --timeout $mpi_timeout $binary_name --gtest_filter=$gtest_filter >> $log_file 2>&1"
    echo $cmd_str
    eval $cmd_str

    # Test if mpirun failed
    if [ $? -ne 0 ]
    then
        echo "FAILED: $cmd_str" >&2
        cat $log_file
        driver_return_status=1
    fi
}

# Processing modes
case $mode in
    all)
        test_with_two_pes="IPCImplSimpleCoarseTestFixture/*:IPCImplSimpleFineTestFixture/*:IPCImplTiledFineTestFixture/*:DegenerateTiledFine.*"
        run_mpirun 4 "-$test_with_two_pes"
        run_mpirun 2 "$test_with_two_pes"
        ;;
    custom)
        # Check if ranks is a positive integer
        if [[ "$ranks" -le 1 ]]; then
            echo "Error: 'ranks' must be a positive integer."
            display_help
            exit 1
        fi
        run_mpirun $ranks $filter
        ;;
    *)
        echo "Error: Invalid mode '$mode'." | tee -a "$log_file"
        display_help
        exit 1
        ;;
esac

echo "Tests Completed"
echo "log file: '$log_file'"
exit $driver_return_status
