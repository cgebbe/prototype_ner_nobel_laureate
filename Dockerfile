# from https://hub.docker.com/r/huggingface/transformers-pytorch-gpu
FROM huggingface/transformers-pytorch-gpu:4.9.1 

RUN pip install black
RUN pip install datasets
RUN pip install seqeval
RUN pip install tensorboard

# https://github.com/microsoft/vscode-remote-release/issues/22#issuecomment-488843424
ARG USERNAME=cgebbe
RUN useradd -m $USERNAME
ENV HOME /home/$USERNAME
USER $USERNAME