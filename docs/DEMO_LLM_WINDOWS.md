# Live LLM text generation on the Windows machine (open source)

The demo act: messy clinical prose generated LIVE by an open-
source model on the Windows laptop itself — no cloud, no cost, no
installation, no admin rights. Runtime: **llamafile** (a single
user-directory executable that serves an OpenAI-compatible API);
model: a 3B-class instruct model (fits 16 GB CPU-only with room
to spare, fast enough to watch).

## 1. One-time setup (no installer, no admin)

**Runtime: llama.cpp's official Windows CPU build** (a plain
PE executable - PROVEN on the locked-down machine; note that
llamafile's polyglot APE format was silently refused by the
enterprise security stack, exiting without error, so llama.cpp
is the primary path here, not the fallback). Model: a 3B-class
instruct gguf.

```bat
%USERPROFILE%\dev\verbatim\.venv\Scripts\activate.bat
cd %USERPROFILE%\dev
curl -L -o llama3.2-3b.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest')); u=[a['browser_download_url'] for a in r['assets'] if 'win-cpu-x64' in a['name'] and a['name'].endswith('.zip')][0]; print(u); urllib.request.urlretrieve(u,'llamacpp.zip')"
mkdir llamacpp
tar -xf llamacpp.zip -C llamacpp
dir /s /b llamacpp\llama-server.exe
```

(~2.1 GB model; download before demo day. The python one-liner
resolves the versioned release asset by API - hardcoded "latest"
asset URLs on GitHub can 404 into tiny text files that curl
happily saves; check `dir` sizes after every download.)

## 2. Start the server (demo day, terminal one)

```bat
cd %USERPROFILE%\dev
llamacpp\llama-server.exe -m llama3.2-3b.gguf --port 8080
```

llama-server narrates: tensor loading, then an explicit
"listening" line. Leave it running. Sanity from terminal two:

```bat
curl http://127.0.0.1:8080/v1/models
```

Live result on the target machine (Llama-3.2-3B, CPU, 8 docs):
verified first try 7, after retry 1, fallbacks 0.

## 3. Point synthkit at it

synthkit's `openai` backend defaults to
`http://127.0.0.1:8080/v1`, so no configuration is needed for
the default llamafile port. (A different port/server: set
`SYNTHKIT_OPENAI_BASE`, e.g.
`set SYNTHKIT_OPENAI_BASE=http://127.0.0.1:9090/v1`.)

## 4. The live act

A demo-sized corpus (8 docs keeps it brisk on CPU — roughly a
minute or two):

```bat
cd %USERPROFILE%\dev\synthkit
python -c "import json; s=json.load(open('xray.json',encoding='utf-8')); s['corpus']['size']=8; json.dump(s,open('xray_demo.json','w',encoding='utf-8'),indent=2)"
synthkit render xray_demo.json -o xray_live --backend openai
type xray_live\docs\doc_00000.txt
```

What the audience sees: the RENDER REPORT counting verified /
retried / fallback in real time — nothing from the model is
trusted; every note is checked against its blueprint — then an
actual clinical report written seconds ago by a model running on
the laptop in front of them. Then close the loop:

```bat
synthkit evaluate xray_live --extractor xray_extractor:extract_careful_v2
```

...and the planted traps get scored against ground truth that
existed before the prose did.

Or run the same act from the dashboard: `synthkit gui` ->
paste the spec at station 02 -> station 03 -> backend `openai`
-> Render data, with the ticker counting and Cancel standing by.

## 5. Why a 3B model is the RIGHT choice here (say this aloud)

Small models are fast, free, and imperfect — and synthkit is
built for imperfect: misses are caught by blueprint
verification, retried, and deterministically backfilled, so the
demo cannot fail; it can only show its safety net working. A
few retries in the report are a feature of the show. The same
spec renders through mistral-24B on a development machine, through Bedrock's
open-weight Llama/Mistral catalog for sanctioned work, or
through a shared vLLM cluster at enterprise scale — one
instrument, one backend dialect, every price point from $0 to
governed cloud.

## Cost framing for the enterprise conversation

- Development / demos: local open models — $0.
- Sanctioned pilots: Bedrock open-weight models (Llama,
  Mistral) — HIPAA-eligible, fractions of a cent per 1K tokens;
  the entire 1,662-call vendor study would cost single-digit
  dollars.
- At scale: a shared vLLM/TGI endpoint serving open weights —
  marginal cost near zero, and synthkit's `openai` backend
  points at it with one environment variable.
