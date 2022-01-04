# from https://hub.docker.com/r/huggingface/transformers-pytorch-gpu
FROM huggingface/transformers-pytorch-gpu:4.9.1 

# install datasets from huggingface
RUN pip install datasets seqeval black

# https://github.com/microsoft/vscode-remote-release/issues/22#issuecomment-488843424
ARG USERNAME=cgebbe
RUN useradd -m $USERNAME
ENV HOME /home/$USERNAME
USER $USERNAME