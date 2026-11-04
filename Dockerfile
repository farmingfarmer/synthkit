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
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

ENTRYPOINT ["synthkit"]
CMD ["batch", "--help"]
