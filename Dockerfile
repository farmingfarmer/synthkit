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
# so the container and the laptop cannot drift ON THE
# INTERPRETER.
#
# IT DOES NOT PIN THE LIBRARIES, AND THAT IS MEASURED RATHER THAN
# ASSUMED. `pip install -e .` resolves them fresh at build time,
# so an image built today carries pandas 3.1.1 where the data
# machine and the development machine both run 3.0.5 - and the
# library layer is the one this project has actually been bitten
# by. Pinning them here would freeze a stack that stops getting
# security updates, which is a worse trade in a hospital account
# than a recorded difference. So the versions are REPORTED
# instead: every run writes them into `provenance.json` under
# `stack`, and two runs of the same commit can be told apart by
# the stack they used.
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
# AND THE DOCUMENTS, because the net reads them.
#
# This was omitted, and the omission falsified the one claim this
# file makes about itself: ten of the suites failed inside the
# image, and most of them failed because a check that reads
# `docs/` found nothing there - "the scorecard exists where the
# suite looks for it", said by a check written to catch exactly
# that. A deployment running the net as instructed would have
# seen ten red suites and reasonably concluded the tool was
# broken. The documents carry no records; `.dockerignore` already
# refuses anything that could.
COPY docs ./docs
# AND THE FILES THE NET ITSELF READS. `smoke_cloud` asserts the
# deployment definition exists and `smoke_packaging` asserts the
# hand-pasted install line still works - both crashed inside the
# image because neither file was here. A net that cannot read the
# thing it is checking is not a check.
COPY requirements.txt Dockerfile .dockerignore .gitignore ./
# `readmit_demo.json` is a generator SPEC - columns, rules and
# outcomes, no records - and `smoke_hybrid` reads it by name.
COPY readmit_demo.json ./

# THE LAB NOTEBOOK DOES NOT TRAVEL, AND A STUB KEEPS THE NET
# HONEST. `.dockerignore` refuses CONVENTIONS.md for the same
# reason the team bundle does - it is the internal record, not a
# deliverable. But ten code comments and several suites reference
# it, so its ABSENCE is a different failure from its exclusion.
# The bundle solved this with a one-line stub detected from its
# own sentence, and the suites skip with a printed NOTE rather
# than failing. Same stub, same sentence, same reason.
RUN printf '%s\n' \
    '# Lab notebook' '' \
    'Code comments reference CONVENTIONS.md, the project lab' \
    'notebook. It lives in the internal development repo. The' \
    'practices it records - fail-first checks, measured' \
    'decisions, predictions on record - are summarized in' \
    'docs/CODE_TOUR.md.' > /opt/synthkit/CONVENTIONS.md

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
