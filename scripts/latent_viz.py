"""THE LATENT SPACE, MADE VISIBLE - a self-contained interactive 3D
comparison of the autoencoder's condensed feature space, original
records beside synthetic ones, so a reader can SEE whether the
synthetic points sit among the real ones or apart.

FOUR measured quantities are computed per point and embedded; the
viewer maps any three to the X/Y/Z axes and a fourth (or the same)
to colour, live, with no recompute:
  centroid distance   to the nearest k-means centroid of the cloud
  neighbour distance  to the nearest neighbour in latent space
  reconstruction err  how well the autoencoder rebuilds the point
  original<->synthetic distance to the nearest point of the OTHER
                      group - the privacy-and-fidelity metric: close
                      means the synthetic point shadows a real one,
                      far means it does not resemble any.

Colour runs green->yellow->red by the chosen metric's percentile.
Labelled 3D axis arrows name which metric is which. Toggle original
and synthetic on and off to see them together or apart. Click a
point to read all four of its numbers.

Self-contained: a hand-written WebGL point cloud, no libraries, no
network. One code path - the CLI writes the file and the bench
serves the same HTML, as fidelity_deck does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

METRICS = [
    ("c", "centroid distance"),
    ("n", "neighbour distance"),
    ("r", "reconstruction error"),
    ("x", "original↔synthetic distance"),
]


def _centroid_dist(Z, seed=0):
    from sklearn.cluster import KMeans
    kc = min(12, max(2, len(Z) // 50))
    km = KMeans(n_clusters=kc, n_init=4, random_state=seed).fit(Z)
    return np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)


def _nn_dist(Z):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    d, _ = nn.kneighbors(Z)
    return d[:, 1]


def _cross_dist(Za, Zb):
    """For every row of Za, the distance to its nearest row of Zb."""
    from sklearn.neighbors import NearestNeighbors
    if not len(Zb):
        return np.full(len(Za), np.nan)
    nn = NearestNeighbors(n_neighbors=1).fit(Zb)
    d, _ = nn.kneighbors(Za)
    return d[:, 0]


def compute_points(src, gen, group_by, seed=0, max_pts=5000):
    """All four latent metrics for source and generated rows."""
    from latent_challenger import LatentGen
    g = LatentGen(seed=seed).fit(src, group_by)
    Xs, Xg = g._re_encode_frame(src), g._re_encode_frame(gen)
    Zs, Zg = g._encode(Xs), g._encode(Xg)
    Z = np.concatenate([Zs, Zg], axis=0)
    labels = np.array([0] * len(Zs) + [1] * len(Zg))
    recon = np.concatenate([
        np.sqrt(((Xs - g._decode(Zs)) ** 2).sum(axis=1)),
        np.sqrt(((Xg - g._decode(Zg)) ** 2).sum(axis=1))])
    cen = _centroid_dist(Z, seed)
    nnd = _nn_dist(Z)
    cross = np.concatenate([_cross_dist(Zs, Zg), _cross_dist(Zg, Zs)])
    idx = np.arange(len(Z))
    if len(idx) > max_pts:
        rng = np.random.RandomState(seed)
        idx = np.sort(np.concatenate([
            rng.choice(np.where(labels == 0)[0],
                       min(int((labels == 0).sum()), max_pts // 2),
                       replace=False),
            rng.choice(np.where(labels == 1)[0],
                       min(int((labels == 1).sum()), max_pts // 2),
                       replace=False)]))
    pts = [{"c": round(float(cen[i]), 4), "n": round(float(nnd[i]), 4),
            "r": round(float(recon[i]), 4),
            "x": round(float(cross[i]), 4) if np.isfinite(cross[i])
            else 0.0, "g": int(labels[i])} for i in idx]
    return pts, {"n_source": int((labels == 0).sum()),
                 "n_generated": int((labels == 1).sum()),
                 "latent_dims": int(Z.shape[1]), "shown": len(pts)}


def build_html(pts, meta):
    return (_TEMPLATE
            .replace("/*DATA*/", json.dumps(pts, separators=(",", ":")))
            .replace("/*META*/", json.dumps(meta))
            .replace("/*METRICS*/", json.dumps(METRICS)))


def build_from_run(src, gen, group_by, seed=0):
    pts, meta = compute_points(src, gen, group_by, seed)
    return build_html(pts, meta)


_TEMPLATE = r"""<style>
html,body{margin:0;height:100%;background:#0a0e13;color:#e2e8f0;
font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh;cursor:grab}#c:active{cursor:grabbing}
.pnl{position:fixed;background:rgba(13,19,27,.86);border:1px solid #1e2b3a;
border-radius:12px;padding:12px 14px;font-size:13px;line-height:1.5;
backdrop-filter:blur(6px)}
#top{top:14px;left:14px;max-width:270px}
#ctl{top:14px;right:14px;width:230px}
#pick{bottom:14px;right:14px;width:230px;display:none}
#leg{bottom:14px;left:14px}
h1{font-size:14px;margin:0 0 7px;letter-spacing:.01em}
.k{color:#7e93a8}.v{color:#e2e8f0;font-weight:600;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
label.row{display:flex;justify-content:space-between;align-items:center;margin:5px 0;gap:8px}
select{background:#0f1720;color:#e2e8f0;border:1px solid #27374a;border-radius:7px;
padding:3px 6px;font:inherit;font-size:12px;flex:1}
.chk{display:flex;align-items:center;gap:6px;margin:4px 0;cursor:pointer;font-size:12px}
.chk input{accent-color:#5dade2}
.bar{height:9px;width:170px;border-radius:5px;margin:5px 0;
background:linear-gradient(90deg,#2ecc71,#f1c40f,#e74c3c)}
.hint{color:#64788c;font-size:11px;margin-top:8px}
.axlab{position:fixed;font-size:11px;font-weight:700;pointer-events:none;
text-shadow:0 0 4px #000,0 0 4px #000;white-space:nowrap}
</style>
<canvas id="c"></canvas>
<div class="pnl" id="top"><h1>Autoencoder latent space</h1>
<div><span class="sw" style="background:#5dade2"></span><span class="v">original</span>
&nbsp;&nbsp;<span class="sw" style="background:#e67e22"></span><span class="v">synthetic</span></div>
<div id="stats"></div>
<div class="hint">drag to spin &middot; wheel to zoom &middot; click a point</div></div>
<div class="pnl" id="ctl"><h1>Axes &amp; colour</h1>
<label class="row"><span class="k">X</span><select id="ax"></select></label>
<label class="row"><span class="k">Y</span><select id="ay"></select></label>
<label class="row"><span class="k">Z</span><select id="az"></select></label>
<label class="row"><span class="k">colour</span><select id="acol"></select></label>
<div style="margin-top:7px">
<div class="chk"><input type="checkbox" id="so" checked><label for="so">show original</label></div>
<div class="chk"><input type="checkbox" id="sg" checked><label for="sg">show synthetic</label></div>
</div></div>
<div class="pnl" id="leg"><span class="k">colour</span> = <span id="colname" class="v"></span> (percentile)
<div class="bar"></div><span class="k">low</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">high</span></div>
<div class="pnl" id="pick"></div>
<div class="axlab" id="lx"></div><div class="axlab" id="ly"></div><div class="axlab" id="lz"></div>
<script>
var PTS=/*DATA*/;var META=/*META*/;var METRICS=/*METRICS*/;
var AX={x:'c',y:'n',z:'r',col:'r'};
function label(key){for(var i=0;i<METRICS.length;i++)if(METRICS[i][0]==key)return METRICS[i][1];return key}
document.getElementById('stats').innerHTML=
 '<div><span class="k">latent dimensions</span> <span class="v">'+META.latent_dims+'</span></div>'+
 '<div><span class="k">original rows</span> <span class="v">'+META.n_source+'</span></div>'+
 '<div><span class="k">synthetic rows</span> <span class="v">'+META.n_generated+'</span></div>'+
 '<div><span class="k">points shown</span> <span class="v">'+META.shown+'</span></div>';
// dropdowns
['ax','ay','az','acol'].forEach(function(id,ix){var s=document.getElementById(id);
 METRICS.forEach(function(m){var o=document.createElement('option');o.value=m[0];o.text=m[1];s.appendChild(o)});
 s.value=[AX.x,AX.y,AX.z,AX.col][ix];
 s.onchange=function(){AX[['x','y','z','col'][ix]]=s.value;rebuild()}});
document.getElementById('so').onchange=document.getElementById('sg').onchange=function(){rebuild()};
var N=PTS.length;var pos=new Float32Array(N*3);var col=new Float32Array(N*3);var grp=new Float32Array(N);var vis=new Float32Array(N);
function norm(key){var mn=1e30,mx=-1e30;for(var i=0;i<N;i++){var v=PTS[i][key];if(v<mn)mn=v;if(v>mx)mx=v}return {mn:mn,r:(mx-mn)||1}}
function ramp(t){var r,g,b;if(t<0.5){var u=t/0.5;r=0.18+u*0.77;g=0.80-u*0.03;b=0.44-u*0.44}else{var u=(t-0.5)/0.5;r=0.95-u*0.05;g=0.77-u*0.47;b=0.06+u*0.09}return [r,g,b]}
function rebuild(){
 var nx=norm(AX.x),ny=norm(AX.y),nz=norm(AX.z);
 var cv=PTS.map(function(p){return p[AX.col]}).slice().sort(function(a,b){return a-b});
 function pct(v){var lo=0,hi=cv.length-1;while(lo<hi){var m=(lo+hi)>>1;if(cv[m]<v)lo=m+1;else hi=m}return lo/(cv.length-1||1)}
 var so=document.getElementById('so').checked,sg=document.getElementById('sg').checked;
 for(var i=0;i<N;i++){var p=PTS[i];
  pos[i*3]=((p[AX.x]-nx.mn)/nx.r)*2-1;pos[i*3+1]=((p[AX.y]-ny.mn)/ny.r)*2-1;pos[i*3+2]=((p[AX.z]-nz.mn)/nz.r)*2-1;
  var c=ramp(pct(p[AX.col]));col[i*3]=c[0];col[i*3+1]=c[1];col[i*3+2]=c[2];grp[i]=p.g;
  vis[i]=((p.g<0.5&&so)||(p.g>0.5&&sg))?1:0}
 gl.bindBuffer(gl.ARRAY_BUFFER,bpos);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.DYNAMIC_DRAW);
 gl.bindBuffer(gl.ARRAY_BUFFER,bcol);gl.bufferData(gl.ARRAY_BUFFER,col,gl.DYNAMIC_DRAW);
 gl.bindBuffer(gl.ARRAY_BUFFER,bvis);gl.bufferData(gl.ARRAY_BUFFER,vis,gl.DYNAMIC_DRAW);
 _pct=pct;document.getElementById('colname').textContent=label(AX.col);}
var _pct=function(){return 0};
// WebGL
var cv=document.getElementById('c'),gl=cv.getContext('webgl');
function sz(){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height)}
window.addEventListener('resize',sz);sz();
function sh(t,s){var o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
var vs=sh(gl.VERTEX_SHADER,'attribute vec3 p;attribute vec3 c;attribute float g;attribute float v;uniform mat4 m;varying vec3 vc;varying float vg;varying float vv;void main(){gl_Position=m*vec4(p,1.0);gl_PointSize=(g>0.5?5.5:4.5);vc=c;vg=g;vv=v;}');
var fs=sh(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 vc;varying float vg;varying float vv;void main(){if(vv<0.5)discard;vec2 d=gl_PointCoord-vec2(0.5);float r=length(d);if(r>0.5)discard;vec3 col=vc;if(vg<0.5)col=mix(vc,vec3(0.36,0.68,0.89),0.3);gl_FragColor=vec4(col,(vg>0.5?1.0:0.82)*(1.0-r));}');
var pr=gl.createProgram();gl.attachShader(pr,vs);gl.attachShader(pr,fs);gl.linkProgram(pr);gl.useProgram(pr);
function mkbuf(){return gl.createBuffer()}
var bpos=mkbuf(),bcol=mkbuf(),bgrp=mkbuf(),bvis=mkbuf();
function bind(b,loc,n){gl.bindBuffer(gl.ARRAY_BUFFER,b);var l=gl.getAttribLocation(pr,loc);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,n,gl.FLOAT,false,0,0)}
gl.bindBuffer(gl.ARRAY_BUFFER,bgrp);gl.bufferData(gl.ARRAY_BUFFER,grp,gl.STATIC_DRAW);
var um=gl.getUniformLocation(pr,'m');
// axis gizmo program (flat colour lines)
var avs=sh(gl.VERTEX_SHADER,'attribute vec3 p;uniform mat4 m;void main(){gl_Position=m*vec4(p,1.0);}');
var afs=sh(gl.FRAGMENT_SHADER,'precision mediump float;uniform vec3 col;void main(){gl_FragColor=vec4(col,0.9);}');
var apr=gl.createProgram();gl.attachShader(apr,avs);gl.attachShader(apr,afs);gl.linkProgram(apr);
var abuf=mkbuf();var acol=gl.getUniformLocation(apr,'col');var aum=gl.getUniformLocation(apr,'m');
function axisVerts(d){ // d is 0,1,2 ; arrow from -1.15 corner along axis to +1.3
 var o=-1.15;var tip=1.3;var a=[o,o,o],b=[o,o,o];b[d]=tip;
 var h1=b.slice(),h2=b.slice();h1[d]=tip-0.12;h2[d]=tip-0.12;var pd=(d+1)%3;h1[pd]+=0.06;h2[pd]-=0.06;
 return new Float32Array([a[0],a[1],a[2], b[0],b[1],b[2], b[0],b[1],b[2], h1[0],h1[1],h1[2], b[0],b[1],b[2], h2[0],h2[1],h2[2]]);
}
var AXC=[[0.9,0.4,0.42],[0.45,0.85,0.5],[0.42,0.6,0.95]];
gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
var az_=0.7,el=0.5,dist=3.6;
function mat(){var ce=Math.cos(el),se=Math.sin(el),ca=Math.cos(az_),sa=Math.sin(az_);
 var ex=dist*ce*sa,ey=dist*se,ez=dist*ce*ca;var zl=Math.sqrt(ex*ex+ey*ey+ez*ez);var zx=ex/zl,zy=ey/zl,zz=ez/zl;
 var xx=zz,xy=0,xz=-zx;var xl=Math.sqrt(xx*xx+xz*xz)||1;xx/=xl;xz/=xl;
 var yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
 var V=[xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*ex+xy*ey+xz*ez),-(yx*ex+yy*ey+yz*ez),-(zx*ex+zy*ey+zz*ez),1];
 var f=1/Math.tan(0.8/2),ar=cv.width/cv.height,nr=0.1,fr=100;
 var P=[f/ar,0,0,0, 0,f,0,0, 0,0,(fr+nr)/(nr-fr),-1, 0,0,(2*fr*nr)/(nr-fr),0];
 var M=new Float32Array(16);for(var r=0;r<4;r++)for(var c=0;c<4;c++){var s=0;for(var k=0;k<4;k++)s+=P[k*4+c]*V[r*4+k];M[r*4+c]=s}return M}
function project(x,y,z,M){var cx=M[0]*x+M[4]*y+M[8]*z+M[12],cy=M[1]*x+M[5]*y+M[9]*z+M[13],cw=M[3]*x+M[7]*y+M[11]*z+M[15];return {x:cx/cw,y:cy/cw,w:cw}}
function draw(){gl.clearColor(0.039,0.055,0.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);var M=mat();
 gl.useProgram(apr);gl.uniformMatrix4fv(aum,false,M);
 for(var d=0;d<3;d++){gl.bindBuffer(gl.ARRAY_BUFFER,abuf);gl.bufferData(gl.ARRAY_BUFFER,axisVerts(d),gl.DYNAMIC_DRAW);var l=gl.getAttribLocation(apr,'p');gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,0,0);gl.uniform3fv(acol,AXC[d]);gl.drawArrays(gl.LINES,0,6)}
 gl.useProgram(pr);gl.uniformMatrix4fv(um,false,M);bind(bpos,'p',3);bind(bcol,'c',3);bind(bgrp,'g',1);bind(bvis,'v',1);gl.drawArrays(gl.POINTS,0,N);
 // axis labels
 var tipd=1.42;var labs=[['lx',0,AX.x],['ly',1,AX.y],['lz',2,AX.z]];
 labs.forEach(function(L){var p=[-1.15,-1.15,-1.15];p[L[1]]=tipd;var s=project(p[0],p[1],p[2],M);var el2=document.getElementById(L[0]);
  if(s.w<=0){el2.style.display='none';return}el2.style.display='block';
  el2.textContent=label(L[2]);el2.style.color='rgb('+Math.round(AXC[L[1]][0]*255)+','+Math.round(AXC[L[1]][1]*255)+','+Math.round(AXC[L[1]][2]*255)+')';
  el2.style.left=((s.x*0.5+0.5)*cv.clientWidth)+'px';el2.style.top=((-s.y*0.5+0.5)*cv.clientHeight)+'px'})}
function loop(){draw();requestAnimationFrame(loop)}
rebuild();loop();
var dn=false,lx=0,ly=0,moved=0;
cv.addEventListener('mousedown',function(e){dn=true;lx=e.clientX;ly=e.clientY;moved=0});
window.addEventListener('mouseup',function(){dn=false});
window.addEventListener('mousemove',function(e){if(!dn)return;var dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;moved+=Math.abs(dx)+Math.abs(dy);az_-=dx*0.006;el=Math.max(-1.5,Math.min(1.5,el+dy*0.006))});
cv.addEventListener('wheel',function(e){e.preventDefault();dist=Math.max(1.4,Math.min(9,dist*(1+(e.deltaY>0?0.1:-0.1))))},{passive:false});
cv.addEventListener('click',function(e){if(moved>5)return;var M=mat();var rect=cv.getBoundingClientRect();var mx=(e.clientX-rect.left)/rect.width*2-1;var my=-(((e.clientY-rect.top)/rect.height)*2-1);var best=-1,bd=0.0025;
 for(var i=0;i<N;i++){if(vis[i]<0.5)continue;var s=project(pos[i*3],pos[i*3+1],pos[i*3+2],M);if(s.w<=0)continue;var d=(s.x-mx)*(s.x-mx)+(s.y-my)*(s.y-my);if(d<bd){bd=d;best=i}}
 var pk=document.getElementById('pick');if(best<0){pk.style.display='none';return}var p=PTS[best];
 pk.style.display='block';pk.innerHTML='<b>'+(p.g>0.5?'synthetic':'original')+' point</b><br>'+
  '<span class="k">centroid distance</span> <span class="v">'+p.c+'</span><br>'+
  '<span class="k">neighbour distance</span> <span class="v">'+p.n+'</span><br>'+
  '<span class="k">reconstruction error</span> <span class="v">'+p.r+'</span><br>'+
  '<span class="k">orig&harr;synth distance</span> <span class="v">'+p.x+'</span><br>'+
  '<span class="k">colour percentile</span> <span class="v">'+(_pct(p[AX.col])*100).toFixed(0)+'%</span>'});
</script>"""



def _read_cohort(path):
    """FOLD "" TO NaN FOR EVERY COLUMN EXCEPT SETS.

    A blanket `.replace("", np.nan)` folds an EMPTY SET into missing,
    and an empty set is a FACT about the visit - it held nothing. That
    is the recorded reader fault which once made the engine learn
    token shares from a source where medication-free visits did not
    exist, every share inflated by 1/(1-empty). Here it would distort
    the very frame the latent space is computed from, which is what
    this picture is ABOUT: a reader is being shown where synthetic
    points sit relative to real ones, so the frame had better be the
    real one. `sets.looks_like_set` is the same predicate the engine
    applies, so the two cannot class a column differently. Per column
    rather than frame-wide, which also avoids the pandas downcast
    warning that is a TypeError on pandas 3 - the version the data
    machine installs.
    """
    from synthkit import sets
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if sets.looks_like_set(df[c]):
            continue
        df[c] = df[c].astype(object).where(
            df[c].astype(str).str.strip() != "", np.nan)
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="latent_viz.py")
    ap.add_argument("src")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--gen", default=None)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = _read_cohort(a.src)
    if a.gen:
        gen = _read_cohort(a.gen)
    else:
        from latent_challenger import LatentGen
        gen = LatentGen(seed=a.seed).fit(src, a.group_by).generate(
            len(src), seed=a.seed + 1)
    Path(a.out).write_text(build_from_run(src, gen, a.group_by, a.seed),
                           encoding="utf-8")
    print("wrote {}".format(a.out))


if __name__ == "__main__":
    main()
