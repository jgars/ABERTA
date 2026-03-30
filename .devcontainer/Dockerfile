# [Choice] PyTorch version
ARG VARIANT="1.12.0-cuda11.3-cudnn8-runtime"
FROM pytorch/pytorch:${VARIANT}

# [Optional] Install Git
RUN apt-get update && apt-get install -y git

# Enable safe.dictory in git
RUN git config --global --add safe.directory /workspace

# Install dependencies
RUN apt-get install -y apt-utils gcc libpq-dev libsndfile-dev
RUN apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
RUN apt-get install -y libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev

# Install requirements with pip.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# Enable bash autocompletion for root
RUN apt-get install -y bash-completion
RUN echo 'if [ -f /etc/bash_completion ]; then' >> /root/.bashrc
RUN echo '  . /etc/bash_completion' >> /root/.bashrc
RUN echo 'fi' >> /root/.bashrc

# Create a user to avoid saving all the files in the container as root
ARG USERNAME=ai
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash -d /home/$USERNAME/ --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support 
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Add project to Python path (non-root)
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspaces/Wav2BERT/"' >> ~/.bashrc
RUN echo 'export WavBERT_PATH="/workspaces/Wav2BERT/"' >> ~/.bashrc
RUN echo 'export KENLM_ROOT_DIR="/workspaces/Wav2BERT/external_lib/kenlm/"' >> ~/.bashrc
RUN echo 'export USE_CUDA=0' >> ~/.bashrc
