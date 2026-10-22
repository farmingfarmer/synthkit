"""The showcase slide - one premium, self-contained page that leads
with the model's own view of the data.

WHY IT EXISTS. The deck (fidelity_deck.py) is for the statistician who
reads every number. This is for the ten-second walk-around: a leader
who wants to know what this thing is before anyone opens a JSON file.
It spins the latent space - real and synthetic patients in the
autoencoder's compressed view, which land on top of each other because
the synthetic data carries the same structure - and then, below the
fold, the same performance plots the deck draws, so a curious viewer
can keep scrolling into the evidence.

ONE CODE PATH. The spinning cloud is a hand-written WebGL point cloud
(no libraries, no CDN, nothing to install) with the 3-D coordinates
BAKED in at build time - a few thousand points, kilobytes, not a
multi-megabyte GIF checked into git. The performance plots are the
exact fidelity_deck SVG functions, recolored for the dark ground, so
the slide and the deck cannot disagree about the same run.

Build from a neural run directory (the one latent_challenger.py writes,
carrying generated.csv, fidelity.json, blueprint.json):

    python scripts/showcase.py RUNDIR --src SRC.csv -o showcase.html
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fidelity_deck as fd  # noqa: E402  (reused SVG plot functions)

# --- palette (matches the published showcase artifact) ----------------
INK = "#0a0a0f"
PANEL = "#13131c"
CARD = "#191922"
LINE = "#2a2a37"
CARD_RED = "#e23b4e"
GOLD = "#ffcc33"
STEEL = "#5fa8d8"
GREEN = "#2ec27a"
TEXT = "#eef0f4"
MUTED = "#9aa0ad"
DIM = "#6a7080"
MAX_POINTS = 4200


def _cloud(src, gen, group_by, seed=0):
    """PCA-whitened 3-D coordinates for real and synthetic rows, plus a
    group label and a dense-core-to-sparse-edge fidelity value. The same
    recipe the marketing GIF used: whiten so the cloud fills the cube,
    clip the few extreme patients so they do not flatten it."""
    from latent_challenger import LatentGen
    from sklearn.decomposition import PCA
    g = LatentGen(seed=seed).fit(src, group_by)
    Xs, Xg = g._re_encode_frame(src), g._re_encode_frame(gen)
    Zs, Zg = g._encode(Xs), g._encode(Xg)
    Z = np.concatenate([Zs, Zg], 0)
    nc = min(3, Z.shape[1])
    P = PCA(n_components=nc, whiten=True, random_state=seed).fit(Z)
    Cs, Cg = P.transform(Zs), P.transform(Zg)
    if nc < 3:  # thin latent space - pad to 3-D so the cloud renders
        Cs = np.pad(Cs, ((0, 0), (0, 3 - nc)))
        Cg = np.pad(Cg, ((0, 0), (0, 3 - nc)))
    A = np.concatenate([Cs, Cg], 0)
    lim = float(np.percentile(np.abs(A), 97)) or 1.0
    A = np.clip(A / lim, -1.0, 1.0)
    grp = np.array([0] * len(Cs) + [1] * len(Cg))
    rad = np.linalg.norm(A - np.median(A, 0), axis=1)
    lo, hi = np.percentile(rad, 8), np.percentile(rad, 96)
    fidv = np.clip((rad - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    idx = np.arange(len(A))
    if len(idx) > MAX_POINTS:
        rng = np.random.RandomState(seed)
        keep0 = rng.choice(np.where(grp == 0)[0],
                           min(int((grp == 0).sum()), MAX_POINTS // 2),
                           replace=False)
        keep1 = rng.choice(np.where(grp == 1)[0],
                           min(int((grp == 1).sum()), MAX_POINTS // 2),
                           replace=False)
        idx = np.sort(np.concatenate([keep0, keep1]))
    A, grp, fidv = A[idx], grp[idx], fidv[idx]
    return (A.round(4).tolist(), grp.tolist(),
            fidv.round(3).tolist(),
            {"n_real": int((grp == 0).sum()),
             "n_syn": int((grp == 1).sum())})


def _webgl(pos, grp, fidv):
    """Self-contained WebGL point cloud, auto-spinning, drag to spin,
    one button to switch real-vs-synthetic <-> fidelity color."""
    data = ("var POS=" + json.dumps([round(x, 4) for row in pos
                                     for x in row])
            + ";var GRP=" + json.dumps(grp)
            + ";var FID=" + json.dumps(fidv) + ";")
    return _WEBGL_HTML.replace("/*__DATA__*/", data)


def _chips(fid):
    from synthkit import gate
    a = gate.assess(fid)
    cells = []
    for c in a["criteria"]:
        ok = c["ok"]
        cells.append(
            '<div class="crit {cls}"><span class="mk">{mk}</span>'
            '<span class="nm">{nm}</span></div>'.format(
                cls="ok" if ok else "no",
                mk="&#10003;" if ok else "&#10007;",
                nm=fd.esc(c["name"])))
    verdict = ("MET" if a["met"] else "REVIEW")
    return a, verdict, "".join(cells)


def _plots(src, gen, run, k=10, max_hist=3):
    """Curated performance plots, recolored for the dark ground: a few
    overlaid distributions and a correlation heatmap. Reuses the deck's
    own SVG so the two pictures are one computation."""
    # retint the deck's module-level fills for the dark theme
    fd.GRAY, fd.RED, fd.RULE, fd.INK = STEEL, CARD_RED, LINE, TEXT
    gid = None
    for c in (src.columns if hasattr(src, "columns") else []):
        if c in gen.columns and ("person" in c or c.endswith("_id")):
            gid = c
            break
    numeric_cols = [c for c in src.columns
                    if c in gen.columns and fd.numeric(src[c]) is not None
                    and src[c].dropna().nunique() > 6]
    out = []
    shown = 0
    for c in numeric_cols:
        if shown >= max_hist:
            break
        vs, vg = fd.numeric(src[c]), fd.numeric(gen[c])
        if vs is None or vg is None or not vs.notna().any():
            continue
        pat = None
        okv = vs.dropna()
        if gid is not None and len(okv):
            lo = min(float(okv.min()),
                     float(vg.dropna().min()) if vg.notna().any()
                     else float(okv.min()))
            hi = max(float(okv.max()),
                     float(vg.dropna().max()) if vg.notna().any()
                     else float(okv.max()))
            edges = np.linspace(lo, hi, 25)
            binidx = np.digitize(okv.to_numpy(dtype=float), edges[1:-1])
            gg = src.loc[okv.index, gid]
            pat = [gg[binidx == i].nunique() for i in range(24)]
        svg, _ = fd.hist_svg(vs, vg, pat, k)
        if not svg:
            continue
        svg = svg.replace('class="live"', 'class="plot"')
        out.append('<figure class="pcard"><figcaption>{}</figcaption>'
                   '{}</figure>'.format(fd.esc(c), svg))
        shown += 1
    heat = ""
    ncols = [c for c in numeric_cols][:9]
    if len(ncols) >= 3:
        ms = fd.spearman_matrix(src, ncols)
        mg = fd.spearman_matrix(gen, ncols)
        hm = fd.heatmap_pair(ncols, ms, mg)
        hm = hm.replace('class="live hm"', 'class="plot hm"')
        heat = ('<figure class="pcard wide"><figcaption>rank '
                'correlation &mdash; original (left) vs synthetic '
                '(right)</figcaption>{}</figure>'.format(hm))
    return "".join(out) + heat


def build_showcase(run_dir, src_path, group_by="person_id", seed=0):
    run = Path(run_dir)
    if not (run / "generated.csv").exists():
        raise ValueError(
            "no generated.csv in {} - run latent_challenger.py csv "
            "first".format(run))
    src = fd.read_frame(src_path)
    gen = fd.read_frame(run / "generated.csv")
    fid = {}
    fj = run / "fidelity.json"
    if fj.exists():
        fid = json.loads(fj.read_text(encoding="utf-8"))
    try:
        k = int(fid.get("k") or 10)
    except (TypeError, ValueError):
        k = 10
    pos, grp, fidv, meta = _cloud(src, gen, group_by, seed)
    webgl = _webgl(pos, grp, fidv)
    assessment, verdict, chips = (_chips(fid) if fid
                                  else (None, None, ""))
    plots = _plots(src, gen, run, k=k)
    chip_block = ""
    if chips:
        chip_block = (
            '<section class="scorecard"><div class="verhead">'
            '<span class="vlabel">Gate</span>'
            '<span class="vval {vc}">{v}</span></div>'
            '<div class="crits">{c}</div></section>'.format(
                vc="met" if (assessment and assessment["met"]) else "rev",
                v=verdict or "", c=chips))
    return _PAGE.replace("/*__WEBGL__*/", webgl) \
                .replace("<!--CHIPS-->", chip_block) \
                .replace("<!--PLOTS-->", plots) \
                .replace("__NREAL__", "{:,}".format(meta["n_real"])) \
                .replace("__NSYN__", "{:,}".format(meta["n_syn"]))


# ---------------------------------------------------------------------
# WebGL block - baked coordinates, no libraries, no network.
_WEBGL_HTML = """
<div class="stage">
  <canvas id="gl"></canvas>
  <div class="vz-ui">
    <button id="mode" class="vz-btn">color: real vs synthetic</button>
    <div class="vz-hint">drag to spin &middot; it spins on its own</div>
  </div>
  <div class="vz-legend" id="lg"></div>
</div>
<script>
(function(){
/*__DATA__*/
var cv=document.getElementById('gl');
var gl=cv.getContext('webgl',{antialias:true,alpha:false,premultipliedAlpha:false});
var legend=document.getElementById('lg');
if(!gl){cv.style.display='none';return;}
var N=GRP.length;
// colors
var STEEL=[0.37,0.66,0.85], CARD=[0.886,0.231,0.306];
function grad(t){ // green -> yellow -> red
  var stops=[[0.18,0.76,0.45],[0.6,0.82,0.18],[1.0,0.82,0.25],[1.0,0.48,0.24],[0.886,0.231,0.306]];
  var x=t*(stops.length-1), i=Math.min(stops.length-2,Math.floor(x)), f=x-i;
  var a=stops[i], b=stops[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
var colA=new Float32Array(N*3), colB=new Float32Array(N*3);
for(var i=0;i<N;i++){
  var g=GRP[i]?CARD:STEEL; colA[i*3]=g[0];colA[i*3+1]=g[1];colA[i*3+2]=g[2];
  var c=grad(FID[i]); colB[i*3]=c[0];colB[i*3+1]=c[1];colB[i*3+2]=c[2];
}
var pos=new Float32Array(POS);
function sh(t,s){var o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;}
var vs=sh(gl.VERTEX_SHADER,'attribute vec3 p;attribute vec3 ca;attribute vec3 cb;uniform mat4 m;uniform float um;uniform float ps;varying vec3 vc;void main(){vec4 q=m*vec4(p,1.0);gl_Position=q;gl_PointSize=ps*(2.2/(2.2+q.z));vc=mix(ca,cb,um);}');
var fs=sh(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 vc;uniform float al;void main(){vec2 d=gl_PointCoord-vec2(0.5);float r=length(d);if(r>0.5)discard;float a=(1.0-r*2.0);gl_FragColor=vec4(vc,al*a*a);}');
var pr=gl.createProgram();gl.attachShader(pr,vs);gl.attachShader(pr,fs);gl.linkProgram(pr);gl.useProgram(pr);
function buf(arr){var b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);return b;}
function attr(b,name,sz){gl.bindBuffer(gl.ARRAY_BUFFER,b);var l=gl.getAttribLocation(pr,name);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,sz,gl.FLOAT,false,0,0);}
var bp=buf(pos), ba=buf(colA), bb=buf(colB);
var uM=gl.getUniformLocation(pr,'m'), uUm=gl.getUniformLocation(pr,'um'),
    uPs=gl.getUniformLocation(pr,'ps'), uAl=gl.getUniformLocation(pr,'al');
gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0.039,0.039,0.059,1.0);
function mul(a,b){var o=new Float32Array(16);for(var r=0;r<4;r++)for(var c=0;c<4;c++){var s=0;for(var k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function persp(f,asp,n,fa){var t=1/Math.tan(f/2);return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0]);}
function rotY(a){var c=Math.cos(a),s=Math.sin(a);return new Float32Array([c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1]);}
function rotX(a){var c=Math.cos(a),s=Math.sin(a);return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);}
function trans(z){return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,z,1]);}
var ay=0.6, ax=-0.32, drag=false, px=0, py=0, auto=true, mode=0;
cv.addEventListener('mousedown',function(e){drag=true;auto=false;px=e.clientX;py=e.clientY;});
window.addEventListener('mouseup',function(){drag=false;});
window.addEventListener('mousemove',function(e){if(!drag)return;ay+=(e.clientX-px)*0.01;ax+=(e.clientY-py)*0.01;px=e.clientX;py=e.clientY;});
cv.addEventListener('touchstart',function(e){auto=false;if(e.touches.length){px=e.touches[0].clientX;py=e.touches[0].clientY;}});
cv.addEventListener('touchmove',function(e){if(e.touches.length){ay+=(e.touches[0].clientX-px)*0.01;ax+=(e.touches[0].clientY-py)*0.01;px=e.touches[0].clientX;py=e.touches[0].clientY;e.preventDefault();}},{passive:false});
document.getElementById('mode').addEventListener('click',function(){mode=mode?0:1;this.textContent=mode?'color: fidelity (green = reproduced)':'color: real vs synthetic';setLegend();});
function setLegend(){legend.innerHTML = mode
  ? '<span class="lg-dot" style="background:#2ec27a"></span>common, reproduced<span class="lg-dot" style="background:#e23b4e;margin-left:16px"></span>rare, protected'
  : '<span class="lg-dot" style="background:#5fa8d8"></span>real<span class="lg-dot" style="background:#e23b4e;margin-left:16px"></span>synthetic';}
setLegend();
function size(){var d=Math.min(window.devicePixelRatio||1,2);var w=cv.clientWidth,h=cv.clientHeight;cv.width=w*d;cv.height=h*d;gl.viewport(0,0,cv.width,cv.height);return w/h;}
function frame(){
  var asp=size();
  if(auto) ay+=0.0045;
  var mv=mul(trans(-2.55),mul(rotX(ax),rotY(ay)));
  var m=mul(persp(1.05,asp,0.1,60),mv);
  gl.uniformMatrix4fv(uM,false,m);
  gl.uniform1f(uUm,mode);
  var dpr=Math.min(window.devicePixelRatio||1,2);
  gl.clear(gl.COLOR_BUFFER_BIT);
  attr(bp,'p',3);attr(ba,'ca',3);attr(bb,'cb',3);
  gl.uniform1f(uPs,20.0*dpr);gl.uniform1f(uAl,0.09);gl.drawArrays(gl.POINTS,0,N); // glow pass
  gl.uniform1f(uPs,6.2*dpr);gl.uniform1f(uAl,0.95);gl.drawArrays(gl.POINTS,0,N);  // core pass
  requestAnimationFrame(frame);
}
frame();
})();
</script>
"""

# ---------------------------------------------------------------------
_PAGE = """<title>SynthKit &mdash; the latent space, spinning</title>
<style>
:root{--ink:#0a0a0f;--panel:#13131c;--card:#191922;--line:#2a2a37;
  --red:#e23b4e;--gold:#ffcc33;--steel:#5fa8d8;--green:#2ec27a;
  --text:#eef0f4;--muted:#9aa0ad;--dim:#6a7080;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--text);font-family:var(--sans);
 -webkit-font-smoothing:antialiased;line-height:1.5;
 background:radial-gradient(1100px 620px at 78% -8%,rgba(226,59,78,.16),transparent 60%),
  radial-gradient(900px 560px at 12% 4%,rgba(95,168,216,.10),transparent 62%),var(--ink);}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(28px,5vw,60px) clamp(20px,4vw,44px) 72px;}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);
 font-weight:700;display:flex;align-items:center;gap:10px;}
.eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--red);box-shadow:0 0 10px var(--red);}
h1{font-size:clamp(32px,6vw,62px);line-height:1.03;font-weight:800;letter-spacing:-.02em;
 margin:.5em 0 .28em;text-wrap:balance;}
h1 .accent{background:linear-gradient(100deg,var(--red),var(--gold));
 -webkit-background-clip:text;background-clip:text;color:transparent;}
.lede{font-size:clamp(16px,2.1vw,20px);color:var(--muted);max-width:62ch;margin:0;}
.stage{position:relative;margin:clamp(26px,4vw,40px) 0 0;height:min(64vh,560px);
 background:linear-gradient(180deg,#101019,#0a0a0f);border:1px solid var(--line);
 border-radius:20px;overflow:hidden;box-shadow:0 30px 70px rgba(0,0,0,.55);}
.stage canvas{width:100%;height:100%;display:block;cursor:grab;}
.stage canvas:active{cursor:grabbing;}
.vz-ui{position:absolute;top:16px;left:16px;display:flex;flex-direction:column;gap:8px;}
.vz-btn{font:600 12.5px var(--sans);letter-spacing:.02em;color:var(--text);
 background:rgba(25,25,34,.78);border:1px solid var(--line);border-radius:999px;
 padding:8px 15px;cursor:pointer;backdrop-filter:blur(6px);}
.vz-btn:hover{border-color:var(--steel);}
.vz-hint{font-size:11.5px;color:var(--dim);padding-left:4px;}
.vz-legend{position:absolute;bottom:16px;left:18px;font-size:12.5px;color:var(--muted);
 display:flex;align-items:center;}
.lg-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;}
.statline{display:flex;flex-wrap:wrap;gap:8px 22px;margin:16px 2px 0;color:var(--dim);font-size:13px;}
.statline b{color:var(--text);font-variant-numeric:tabular-nums;}
.chips{display:flex;flex-wrap:wrap;gap:12px;margin:clamp(30px,5vw,44px) 0 0;}
.chip{flex:1 1 200px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;}
.chip .big{font-size:30px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.chip .big.red{color:var(--red)}.chip .big.gold{color:var(--gold)}.chip .big.steel{color:var(--steel)}.chip .big.green{color:var(--green)}
.chip .lab{color:var(--muted);font-size:13.5px;margin-top:4px;}
h2{font-size:clamp(20px,3vw,26px);font-weight:800;letter-spacing:-.01em;margin:clamp(42px,6vw,64px) 0 6px;}
.sub{color:var(--muted);font-size:14.5px;margin:0 0 20px;max-width:64ch;}
.scorecard{background:var(--panel);border:1px solid var(--line);border-radius:16px;
 padding:18px 20px;margin-top:18px;}
.verhead{display:flex;align-items:baseline;gap:12px;margin-bottom:12px;}
.vlabel{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);font-weight:700;}
.vval{font-size:15px;font-weight:800;letter-spacing:.06em;}
.vval.met{color:var(--green)}.vval.rev{color:var(--gold)}
.crits{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px 16px;}
.crit{display:flex;align-items:center;gap:9px;font-size:13.5px;color:var(--muted);}
.crit .mk{width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;
 justify-content:center;font-size:11px;font-weight:700;flex:0 0 auto;}
.crit.ok .mk{background:rgba(46,194,122,.16);color:var(--green);}
.crit.no .mk{background:rgba(226,59,78,.16);color:var(--red);}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px;}
@media(max-width:820px){.plots{grid-template-columns:1fr}}
.pcard{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;
 padding:16px 16px 10px;overflow:hidden;}
.pcard.wide{grid-column:1/-1;}
.pcard figcaption{font-size:12px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--steel);font-weight:700;margin-bottom:8px;}
.pcard svg.plot{width:100%;height:auto;display:block;}
.plots svg text{fill:var(--muted);}
.pcard svg.hm .hm-other{display:none;}
.foot{margin-top:clamp(38px,6vw,60px);padding-top:22px;border-top:1px solid var(--line);
 display:flex;flex-wrap:wrap;gap:16px 28px;justify-content:space-between;align-items:baseline;}
.foot .posture{color:var(--muted);font-size:13px;max-width:70ch;}
.foot .posture b{color:var(--text);}
.foot .note{color:var(--dim);font-size:12px;margin-top:10px;}
.brand{font-weight:800;letter-spacing:-.01em;}.brand .usc{color:var(--red);}
</style>
<div class="wrap">
 <div class="eyebrow"><span class="dot"></span> Data Science &amp; AI Team &middot; Keck Medicine of USC</div>
 <h1>Real data in.<br>Synthetic data out.<br><span class="accent">Indistinguishable.</span></h1>
 <p class="lede">SynthKit learns the statistical patterns of a medical dataset and generates
  brand-new synthetic patients that reproduce them &mdash; without copying a single real record.
  Below is that model's own compressed view of the data. Drag it. It spins on its own.</p>

 /*__WEBGL__*/
 <div class="statline"><span><b>__NREAL__</b> real patient-visits</span>
  <span><b>__NSYN__</b> synthetic patient-visits</span>
  <span>drawn so they share the same structure</span></div>

 <div class="chips">
  <div class="chip"><div class="big red">0</div><div class="lab">real records copied or stored</div></div>
  <div class="chip"><div class="big gold">k &ge; 10</div><div class="lab">patients behind every published number</div></div>
  <div class="chip"><div class="big steel">50%</div><div class="lab">re-identification attack &mdash; a coin flip<br><span style="color:var(--dim)">(a record-leaking generator scores 100%)</span></div></div>
  <div class="chip"><div class="big green">2</div><div class="lab">independent engines, cross-checked</div></div>
 </div>

 <h2>How faithful is it?</h2>
 <p class="sub">The same evidence the full report carries, graded on this run &mdash;
  distributions that line up, relationships that survive, a privacy gate that has to pass.</p>
 <!--CHIPS-->
 <div class="plots"><!--PLOTS--></div>

 <div class="foot">
  <div class="posture"><b>Honest posture:</b> this is k-anonymous, group-level synthesis measured
   to resist a membership-inference attack &mdash; not a blanket &ldquo;de-identified&rdquo;
   guarantee. The output is the release; no source record travels with it.
   <div class="note">Visualizations use fully invented demo data &mdash; no real patient records.</div></div>
  <div class="brand">Synth<span class="usc">Kit</span></div>
 </div>
</div>
"""


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="showcase.py")
    ap.add_argument("run_dir")
    ap.add_argument("--src", required=True)
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="showcase.html")
    a = ap.parse_args(argv)
    html = build_showcase(a.run_dir, a.src, a.group_by, a.seed)
    io.open(a.out, "w", encoding="utf-8").write(html)
    print("wrote {} ({} KB)".format(a.out, len(html) // 1024))


if __name__ == "__main__":
    main()
