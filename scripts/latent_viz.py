"""THE LATENT SPACE, MADE VISIBLE. A self-contained interactive 3D
view of the autoencoder's condensed feature space - the original
records and the synthetic ones placed together, so a reader can
SEE whether the synthetic points sit among the real ones or apart.

Each point is positioned by three measured quantities of its latent
code:
  X  distance to the nearest k-means centroid of the latent cloud
  Y  distance to its nearest neighbor in latent space
  Z  reconstruction error (how well the autoencoder rebuilds it)
and colored green->yellow->red by its percentile on a chosen metric
(default: reconstruction error). A synthetic point that reconstructs
far worse, or sits far from every real neighbor, is visibly red and
far out - the privacy-and-fidelity story in one picture.

Self-contained: a hand-written WebGL point cloud, no libraries, no
network. Orbit by dragging, zoom with the wheel, click a point to
read it. The same discipline as fidelity_deck - one code path emits
the HTML the bench serves and the file a teammate opens.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _nearest_centroid_dist(Z, n_clusters, seed=0):
    from sklearn.cluster import KMeans
    kc = min(n_clusters, max(2, len(Z) // 50))
    km = KMeans(n_clusters=kc, n_init=4, random_state=seed).fit(Z)
    d = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    return d


def _nn_dist(Z):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    dist, _ = nn.kneighbors(Z)
    return dist[:, 1]          # [:,0] is the point itself


def compute_points(src, gen, group_by, seed=0, max_pts=4000):
    """Latent metrics for source and generated rows, as a list of
    point dicts ready to embed."""
    from latent_challenger import LatentGen
    g = LatentGen(seed=seed).fit(src, group_by)
    Xs = g._re_encode_frame(src)
    Xg = g._re_encode_frame(gen)
    Zs, Zg = g._encode(Xs), g._encode(Xg)
    Z = np.concatenate([Zs, Zg], axis=0)
    labels = np.array([0] * len(Zs) + [1] * len(Zg))
    # reconstruction error in the encoded space
    Rs = g._decode(Zs)
    Rg = g._decode(Zg)
    recon = np.concatenate([
        np.sqrt(((Xs - Rs) ** 2).sum(axis=1)),
        np.sqrt(((Xg - Rg) ** 2).sum(axis=1))])
    cen = _nearest_centroid_dist(Z, 12, seed)
    nnd = _nn_dist(Z)
    # subsample for the browser if huge, keeping both groups
    idx = np.arange(len(Z))
    if len(idx) > max_pts:
        rng = np.random.RandomState(seed)
        keep = np.concatenate([
            rng.choice(np.where(labels == 0)[0],
                       min((labels == 0).sum(), max_pts // 2),
                       replace=False),
            rng.choice(np.where(labels == 1)[0],
                       min((labels == 1).sum(), max_pts // 2),
                       replace=False)])
        idx = np.sort(keep)
    pts = []
    for i in idx:
        pts.append({
            "x": float(cen[i]), "y": float(nnd[i]),
            "z": float(recon[i]),
            "g": int(labels[i])})
    return pts, {"n_source": int((labels == 0).sum()),
                 "n_generated": int((labels == 1).sum()),
                 "latent_dims": int(Z.shape[1]),
                 "shown": len(pts)}


def build_html(pts, meta, color_by="z"):
    data = json.dumps(pts, separators=(",", ":"))
    return _TEMPLATE.replace("/*DATA*/", data).replace(
        "/*META*/", json.dumps(meta)).replace(
        "__COLORBY__", color_by)


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>synthkit - latent space</title><style>
html,body{margin:0;height:100%;background:#0b0f14;color:#dfe6ee;
font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh;cursor:grab}
#c:active{cursor:grabbing}
.panel{position:fixed;top:14px;left:14px;background:rgba(14,20,28,.82);
border:1px solid #223;border-radius:12px;padding:12px 15px;font-size:13px;
max-width:300px;line-height:1.5}
.panel h1{font-size:15px;margin:0 0 6px}
.k{color:#8aa}.v{color:#dfe6ee;font-weight:600}
.leg{position:fixed;bottom:14px;left:14px;background:rgba(14,20,28,.82);
border:1px solid #223;border-radius:10px;padding:9px 12px;font-size:12px}
.bar{height:9px;width:180px;border-radius:5px;margin:5px 0;
background:linear-gradient(90deg,#2ecc71,#f1c40f,#e74c3c)}
#pick{position:fixed;top:14px;right:14px;background:rgba(14,20,28,.9);
border:1px solid #2a3a4a;border-radius:10px;padding:11px 14px;font-size:12px;
max-width:260px;display:none}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.hint{color:#789;font-size:11px;margin-top:7px}
</style></head><body>
<canvas id="c"></canvas>
<div class="panel"><h1>Autoencoder latent space</h1>
<div><span class="sw" style="background:#5dade2"></span><span class="v">original</span>
&nbsp;&nbsp;<span class="sw" style="background:#e67e22"></span><span class="v">synthetic</span></div>
<div id="stats"></div>
<div class="hint">drag to spin &middot; wheel to zoom &middot; click a point to read it</div></div>
<div class="leg">colour = reconstruction error (percentile)
<div class="bar"></div>
<span class="k">low</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="k">high</span></div>
<div id="pick"></div>
<script>
var PTS=/*DATA*/;var META=/*META*/;var COLORBY="__COLORBY__";
document.getElementById('stats').innerHTML=
 '<div><span class="k">latent dimensions</span> <span class="v">'+META.latent_dims+'</span></div>'+
 '<div><span class="k">original rows</span> <span class="v">'+META.n_source+'</span></div>'+
 '<div><span class="k">synthetic rows</span> <span class="v">'+META.n_generated+'</span></div>'+
 '<div><span class="k">points shown</span> <span class="v">'+META.shown+'</span></div>';
// ---- normalise axes to [-1,1] and compute colour percentiles ----
function norm(key){var a=PTS.map(function(p){return p[key]});var mn=Math.min.apply(0,a),mx=Math.max.apply(0,a);var r=(mx-mn)||1;return {mn:mn,mx:mx,r:r}}
var nx=norm('x'),ny=norm('y'),nz=norm('z');
var cvals=PTS.map(function(p){return p[COLORBY]}).slice().sort(function(a,b){return a-b});
function pct(v){var lo=0,hi=cvals.length-1;while(lo<hi){var m=(lo+hi)>>1;if(cvals[m]<v)lo=m+1;else hi=m}return lo/(cvals.length-1||1)}
function ramp(t){ // green-yellow-red
 var r,g,b;if(t<0.5){var u=t/0.5;r=0.18+u*(0.95-0.18);g=0.80+u*(0.77-0.80);b=0.44-u*0.44}
 else{var u=(t-0.5)/0.5;r=0.95-u*0.05;g=0.77-u*(0.77-0.30);b=0.06+u*0.09}return [r,g,b]}
// positions and colours
var N=PTS.length;var pos=new Float32Array(N*3);var col=new Float32Array(N*3);var grp=new Float32Array(N);
for(var i=0;i<N;i++){var p=PTS[i];
 pos[i*3]=((p.x-nx.mn)/nx.r)*2-1;pos[i*3+1]=((p.y-ny.mn)/ny.r)*2-1;pos[i*3+2]=((p.z-nz.mn)/nz.r)*2-1;
 var c=ramp(pct(p[COLORBY]));col[i*3]=c[0];col[i*3+1]=c[1];col[i*3+2]=c[2];grp[i]=p.g}
// ---- WebGL ----
var cv=document.getElementById('c'),gl=cv.getContext('webgl');
function sz(){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height)}
window.addEventListener('resize',sz);sz();
function sh(t,s){var o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
var vs=sh(gl.VERTEX_SHADER,'attribute vec3 p;attribute vec3 c;attribute float g;uniform mat4 m;varying vec3 vc;varying float vg;void main(){gl_Position=m*vec4(p,1.0);gl_PointSize=(g>0.5?5.0:4.0)*(1.0+ (1.0-gl_Position.z/gl_Position.w)*0.5);vc=c;vg=g;}');
var fs=sh(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 vc;varying float vg;void main(){vec2 d=gl_PointCoord-vec2(0.5);float r=length(d);if(r>0.5)discard;float a=vg>0.5?1.0:0.85;vec3 col=vc;if(vg<0.5)col=mix(vc,vec3(0.36,0.68,0.89),0.35);gl_FragColor=vec4(col,a*(1.0-r));}');
var pr=gl.createProgram();gl.attachShader(pr,vs);gl.attachShader(pr,fs);gl.linkProgram(pr);gl.useProgram(pr);
function buf(data,loc,n){var b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);var l=gl.getAttribLocation(pr,loc);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,n,gl.FLOAT,false,0,0)}
buf(pos,'p',3);buf(col,'c',3);buf(grp,'g',1);
var um=gl.getUniformLocation(pr,'m');
gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
// camera
var az=0.7,el=0.5,dist=3.4;
function mat(){
 var ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
 var ex=dist*ce*sa,ey=dist*se,ez=dist*ce*ca;
 // lookAt(eye,0,up)
 var zx=ex,zy=ey,zz=ez;var zl=Math.sqrt(zx*zx+zy*zy+zz*zz);zx/=zl;zy/=zl;zz/=zl;
 var ux=0,uy=1,uz=0;var xx=uy*zz-uz*zy,xy=uz*zx-ux*zz,xz=ux*zy-uy*zx;var xl=Math.sqrt(xx*xx+xy*xy+xz*xz);xx/=xl;xy/=xl;xz/=xl;
 var yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
 var V=[xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*ex+xy*ey+xz*ez),-(yx*ex+yy*ey+yz*ez),-(zx*ex+zy*ey+zz*ez),1];
 var f=1/Math.tan(0.8/2),a=cv.width/cv.height,nr=0.1,fr=100;
 var P=[f/a,0,0,0, 0,f,0,0, 0,0,(fr+nr)/(nr-fr),-1, 0,0,(2*fr*nr)/(nr-fr),0];
 var M=new Float32Array(16);for(var r=0;r<4;r++)for(var c=0;c<4;c++){var s=0;for(var k=0;k<4;k++)s+=P[k*4+c]*V[r*4+k];M[r*4+c]=s}return M;
}
function draw(){gl.clearColor(0.043,0.059,0.078,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniformMatrix4fv(um,false,mat());gl.drawArrays(gl.POINTS,0,N)}
function loop(){draw();requestAnimationFrame(loop)}loop();
// interaction
var dn=false,lx=0,ly=0,moved=0;
cv.addEventListener('mousedown',function(e){dn=true;lx=e.clientX;ly=e.clientY;moved=0});
window.addEventListener('mouseup',function(){dn=false});
window.addEventListener('mousemove',function(e){if(!dn)return;var dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;moved+=Math.abs(dx)+Math.abs(dy);az-=dx*0.006;el=Math.max(-1.5,Math.min(1.5,el+dy*0.006))});
cv.addEventListener('wheel',function(e){e.preventDefault();dist=Math.max(1.3,Math.min(9,dist*(1+(e.deltaY>0?0.1:-0.1))))},{passive:false});
// pick: project all points, nearest to click in screen space
cv.addEventListener('click',function(e){if(moved>5)return;var M=mat();var rect=cv.getBoundingClientRect();var mx=(e.clientX-rect.left)/rect.width*2-1;var my=-(((e.clientY-rect.top)/rect.height)*2-1);var best=-1,bd=0.0025;
 for(var i=0;i<N;i++){var x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];var cx=M[0]*x+M[4]*y+M[8]*z+M[12];var cy=M[1]*x+M[5]*y+M[9]*z+M[13];var cw=M[3]*x+M[7]*y+M[11]*z+M[15];if(cw<=0)continue;var sx=cx/cw,sy=cy/cw;var d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);if(d<bd){bd=d;best=i}}
 var pk=document.getElementById('pick');if(best<0){pk.style.display='none';return}var p=PTS[best];
 pk.style.display='block';pk.innerHTML='<b>'+(p.g>0.5?'synthetic':'original')+' point</b><br>'+
  '<span class="k">centroid dist</span> <span class="v">'+p.x.toFixed(3)+'</span><br>'+
  '<span class="k">nearest-neighbour dist</span> <span class="v">'+p.y.toFixed(3)+'</span><br>'+
  '<span class="k">reconstruction error</span> <span class="v">'+p.z.toFixed(3)+'</span><br>'+
  '<span class="k">colour percentile</span> <span class="v">'+(pct(p[COLORBY])*100).toFixed(0)+'%</span>';
});
</script></body></html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="latent_viz.py")
    ap.add_argument("src", help="source CSV")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--gen", default=None,
                    help="generated CSV; if omitted, one is generated")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = pd.read_csv(a.src, dtype=str,
                      keep_default_na=False).replace("", np.nan)
    if a.gen:
        gen = pd.read_csv(a.gen, dtype=str,
                          keep_default_na=False).replace("", np.nan)
    else:
        from latent_challenger import LatentGen
        gen = LatentGen(seed=a.seed).fit(
            src, a.group_by).generate(len(src), seed=a.seed + 1)
    pts, meta = compute_points(src, gen, a.group_by, a.seed)
    Path(a.out).write_text(build_html(pts, meta), encoding="utf-8")
    print("wrote {} ({} points, {} latent dims)".format(
        a.out, meta["shown"], meta["latent_dims"]))


if __name__ == "__main__":
    main()
