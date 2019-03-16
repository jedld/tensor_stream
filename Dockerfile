FROM circleci/ruby:2.6.1-node-browsers
RUN sudo apt-get update -q && sudo apt-get install --no-install-recommends -yq alien wget unzip clinfo \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN export DEVEL_URL="https://software.intel.com/file/531197/download" \
    && sudo wget ${DEVEL_URL} -q -O download.zip --no-check-certificate \
    && sudo unzip download.zip \
    && sudo rm -f download.zip *.tar.xz* \
    && sudo alien --to-deb *dev*.rpm \
    && sudo dpkg -i *dev*.deb \
    && sudo rm *.rpm *.deb
RUN export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz" \
    && export TAR=$(basename ${RUNTIME_URL}) \
    && export DIR=$(basename ${RUNTIME_URL} .tgz) \
    && sudo wget -q ${RUNTIME_URL} \
    && sudo tar -xf ${TAR} \
    && for i in ${DIR}/rpm/*.rpm; do sudo alien --to-deb $i; done \
    && sudo rm -rf ${DIR} ${TAR} \
    && sudo dpkg -i *.deb \
    && sudo rm *.deb
RUN sudo mkdir -p /etc/OpenCL/vendors/ \
    &&  echo "/opt/intel/opencl-1.2-6.4.0.25/lib64/libintelocl.so" | sudo tee --append /etc/OpenCL/vendors/intel.icd > /dev/null
ENV OCL_INC /opt/intel/opencl/include
ENV OCL_LIB /opt/intel/opencl-1.2-6.4.0.25/lib64
ENV LIBOPENCL_SO /opt/intel/opencl-1.2-6.4.0.25/lib64/libOpenCL.so
ENV LD_LIBRARY_PATH $OCL_LIB:$LD_LIBRARY_PATH
