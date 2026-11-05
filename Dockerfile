# SynthKit, for a hospital's own AWS account.
#
# THE IMAGE CARRIES NO DATA AND NO CREDENTIALS. It is the tool;
# the extract stays in the account that already holds it and is
# mounted or pulled at run time. Nothing is baked in, which is
# why this file has no COPY of anything but the source tree.
#
# PINNED TO THE INTERPRETER THE DATA MACHINE INSTALLS. This
# project has twice been bitten by testing against a different
# Python than the one that runs: pandas 3 turned a FutureWarning
# into a TypeError, and 3.14 turned a latent help-string bug
# into a crash on every invocation. The image names its version
# so the container and the laptop cannot drift.
FROM python:3.14-slim

# Build tools only while installing, then gone - a smaller
# image is a smaller thing to review.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/synthkit
COPY pyproject.toml README.md ./
COPY synthkit ./synthkit
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e ".[explain]" boto3 \
    && apt-get purge -y build-essential || true

# RUN AS NOBODY. A process that cannot write outside its work
# directory cannot leave anything behind in a shared account.
RUN useradd -m -u 10001 synth && mkdir -p /work && chown synth /work
USER synth
WORKDIR /work

# THE NET IS PART OF THE IMAGE, so a deployment can prove the
# tool works before it is pointed at an extract:
#   docker run --rm IMAGE python /opt/synthkit/scripts/run_all_smokes.py
#
# AND THE IMAGE DOES NOT PIN THE THREAD COUNT.
#
# It used to set OMP_NUM_THREADS=1 here. That is the TEST-RUN
# setting: fixture-sized frames spend more time starting threads
# than computing, which is worth 35x to the net. A real extract
# is 55,428 rows, where the threads genuinely pay - so pinning
# them in the image would make every production fit in this
# container single-threaded, silently.
#
# The net sets the variable on the subprocesses it spawns, by
# `setdefault`, so an outside value still wins. An ENV here IS
# that outside value. `smoke_packaging` already guards the
# module against this exact generalisation; the image needed
# the same guard, and this one was found by reading the
# deployment surface rather than by running it.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["synthkit"]
CMD ["batch", "--help"]
