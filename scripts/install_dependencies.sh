#!/bin/bash
set -e
set -o pipefail

# ROCm location
if [[ -z "${_ROCM_DIR}" ]]; then
  export _ROCM_DIR=/opt/rocm
fi

# Location to install dependencies
if [[ -z "${_PREFIX}" ]]; then
  export _PREFIX=$HOME/installDIR
fi

# Location of dependencies source code
export _DEPS_SRC_DIR=$_PREFIX/src
mkdir -p $_DEPS_SRC_DIR

#Adjust branches and installation location as necessary
export _UCX_INSTALL_DIR=$_PREFIX
export _UCX_REPO=https://github.com/openucx/ucx.git
export _UCX_BRANCH=v1.17.x

export _OMPI_INSTALL_DIR=$_PREFIX
export _OMPI_REPO=https://github.com/open-mpi/ompi.git
export _OMPI_BRANCH=v5.0.x

# Step 1: Build UCX with ROCm support
cd $_DEPS_SRC_DIR
rm -rf ucx
git clone $_UCX_REPO -b $_UCX_BRANCH
cd ucx
./autogen.sh
./contrib/configure-release --prefix=$_UCX_INSTALL_DIR \
                            --with-rocm=$_ROCM_DIR     \
                            --enable-mt                \
                            --without-go               \
                            --without-java             \
                            --without-cuda             \
                            --without-verbs            \
                            --without-knem
make -j
make install

# Step 2: Install OpenMPI with UCX support
cd $_DEPS_SRC_DIR
rm -rf ompi
git clone --recursive $_OMPI_REPO -b $_OMPI_BRANCH
cd ompi
./autogen.pl
./configure --prefix=$_OMPI_INSTALL_DIR  \
            --with-rocm=$_ROCM_DIR       \
            --with-ucx=$_UCX_INSTALL_DIR \
            --disable-oshmem             \
            --with-prrte=internal        \
            --with-hwloc=internal        \
            --with-libevent=internal     \
            --without-cuda               \
            --disable-sphinx             \
            --disable-mpi-fortran        \
            --without-ofi
make -j
make install

rm -rf $_DEPS_SRC_DIR

echo "Dependencies for rocSHMEM are now installed"
echo ""
echo "UCX $_UCX_BRANCH Installed to $_UCX_INSTALL_DIR"
echo "OpenMPI $_OMPI_BRANCH Installed to $_OMPI_INSTALL_DIR"
echo ""
echo "Please update your PATH and LD_LIBRARY_PATH"
