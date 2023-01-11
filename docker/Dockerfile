# Fermat3 Dev Docker/Podman image
FROM ubuntu:20.04

WORKDIR /rl

RUN apt update

RUN apt install -y git make perl gcc build-essential libc6-dev gcc-multilib cpanminus less vim

RUN cpanm Bit::Vector && cpanm Set::IntRange

RUN git clone https://gitlab.com/clatu/fermat3.git --recursive

WORKDIR /rl/fermat3

RUN git checkout dev

RUN make test

ENV PATH="/rl/fermat3/bin:${PATH}"
ENV FermaT="/rl/fermat3"


# map this to working directory with -v
# Example: podman run -v `pwd`:/rl/wd fermat wsl bin/features-generator.wsl bin/cognitive.wsl
WORKDIR /rl/wd
