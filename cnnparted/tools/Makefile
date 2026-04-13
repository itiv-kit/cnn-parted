# This is an adapted version of the Makefile in 
# accelergy-timeloop-infrastructur to make it work with the container

# 1. Get the directory where THIS Makefile lives
# $(lastword $(MAKEFILE_LIST)) is a special Make variable for the current file
THIS_FILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
TOOLS_DIR := $(patsubst %/,%,$(dir $(THIS_FILE_PATH)))

# 2. Go up to the project root (adjust the number of ../ based on depth)
# Since this Makefile is in cnnparted/tools, we go up two levels
LOCAL_ROOT := $(abspath $(TOOLS_DIR)/../..)

# 3. Use GitLab's variable if it exists, otherwise use our calculated LOCAL_ROOT
PROJECT_ROOT ?= $(CI_PROJECT_DIR)
ifeq ($(PROJECT_ROOT),)
    PROJECT_ROOT := $(LOCAL_ROOT)
endif

EXTERNAL_LIBS_DIR := $(PROJECT_ROOT)/.external_libs


# For Timeloop installation
BARVINOK_VER ?= 0.41.6
NTL_VER      ?= 11.5.1

local_install_accelergy:
	uv pip install setuptools pyyaml
	cd  accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in && make
	cd  accelergy-timeloop-infrastructure/src/accelergy-neurosim-plug-in && make
	cd  accelergy-timeloop-infrastructure/src/ && uv pip install --no-build-isolation ./accelergy*

local_install_timeloop:
	mkdir -p /tmp/build-timeloop
	uv pip install setuptools scons

	cd /tmp/build-timeloop \
		&& wget https://libntl.org/ntl-${NTL_VER}.tar.gz \
		&& tar -xvzf ntl-${NTL_VER}.tar.gz --no-same-owner \
		&& cd ntl-${NTL_VER}/src \
		&& ./configure NTL_GMP_LIP=on SHARED=on \
		&& make -j8 \
		&& echo "Installing to: $(EXTERNAL_LIBS_DIR)" \
		&& make install DESTDIR=$(EXTERNAL_LIBS_DIR)

	cd /tmp/build-timeloop \
		&& wget https://barvinok.sourceforge.io/barvinok-${BARVINOK_VER}.tar.gz \
		&& tar -xvzf barvinok-${BARVINOK_VER}.tar.gz \
		&& cd barvinok-${BARVINOK_VER} \
		&& ./configure --enable-shared-barvinok \
		&& make -j8 \
		&& echo "Installing to: $(EXTERNAL_LIBS_DIR)" \
		&& make install DESTDIR=$(EXTERNAL_LIBS_DIR)

	cd  accelergy-timeloop-infrastructure/src/timeloop \
		&& cp -r pat-public/src/pat src/pat  \
		&& scons -j8 --with-isl --accelergy


local_install_timeloop_python:
	cd  accelergy-timeloop-infrastructure/src/timeloop-python \
		&& uv pip install --no-build-isolation .  

	cd  accelergy-timeloop-infrastructure/src/timeloopfe \
		&& uv pip install --no-build-isolation .  

#cp ${ROOT_DIR}/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-mapper  ~/.local/bin/timeloop-mapper
#cp ${ROOT_DIR}/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-metrics ~/.local/bin/timeloop-metrics
#cp ${ROOT_DIR}/cnnparted/tools/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-model ~/.local/bin/timeloop-model