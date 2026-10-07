import cv2, numpy as np, tkinter as tk, json
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# AUTO FACE RESTORE v11.2 — frontal dense recovery + side-safe alignment
# Requires face_detection_yunet_2026may.onnx in the same folder.
# Uses YuNet's face box + 5 landmarks to extract ONLY the source face,
# then aligns it to the target face. Never transfers source background/clothing.

HERE=Path(__file__).resolve().parent
MODEL=HERE/"face_detection_yunet_2026may.onnx"

def read_img(p):
    a=np.fromfile(p,np.uint8); im=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if im is None: raise ValueError("Cannot open image")
    return im
def save_img(p,im):
    ext=Path(p).suffix.lower() or ".png"
    if ext not in [".png",".jpg",".jpeg",".webp"]: p += ".png"; ext=".png"
    ok,b=cv2.imencode(ext,im)
    if not ok: raise ValueError("Save failed")
    b.tofile(p)

_DETECTOR = None
_MEDIAPIPE = None

def _dense_landmarker():
    """Optional dense landmark engine.

    MediaPipe Face Mesh is used only as an alignment refiner; YuNet remains
    the mandatory detector/fallback so the application still runs without
    MediaPipe. The refiner returns 468/478 points in image coordinates.
    """
    global _MEDIAPIPE
    if _MEDIAPIPE is not None:
        return _MEDIAPIPE
    try:
        import mediapipe as mp
        # Prefer the stable legacy FaceMesh API because it is available across
        # more MediaPipe releases. Refine landmarks adds iris points where supported.
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        _MEDIAPIPE = fm
        return fm
    except Exception:
        _MEDIAPIPE = False
        return None

# MediaPipe landmark indices used for stable geometric measurements.
# These are intentionally sparse; the full mesh is used to compute robust
# consensus geometry and contour, while these groups provide pose/scale anchors.
_MP_LEFT_EYE = [33, 133, 159, 145]
_MP_RIGHT_EYE = [362, 263, 386, 374]
_MP_NOSE = [1, 4, 5, 6, 197]
_MP_MOUTH_L = [61, 291, 78, 308]
_MP_MOUTH_R = [291, 61, 308, 78]

def _mp_points(im):
    fm = _dense_landmarker()
    if fm is None:
        return None, 0.0
    try:
        rgb=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        res=fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, 0.0
        lm=res.multi_face_landmarks[0].landmark
        h,w=im.shape[:2]
        pts=np.array([[q.x*w,q.y*h,q.z*w] for q in lm],np.float32)
        if len(pts)<400 or not np.isfinite(pts).all():
            return None, 0.0
        # Reject obviously broken meshes.
        xy=pts[:,:2]
        if np.min(xy[:,0]) < -0.15*w or np.max(xy[:,0]) > 1.15*w or np.min(xy[:,1]) < -0.15*h or np.max(xy[:,1]) > 1.15*h:
            return None, 0.0
        return pts, 1.0
    except Exception:
        return None, 0.0

def _mean_pts(pts, ids):
    ids=[i for i in ids if i < len(pts)]
    return np.mean(pts[ids,:2],axis=0).astype(np.float32) if ids else np.array([np.nan,np.nan],np.float32)

def _dense_anchors(pts):
    """Return five canonical anchors from a dense mesh."""
    if pts is None or len(pts)<400:
        return None
    # MediaPipe's semantic eye/nose/mouth indices are stable in the legacy mesh.
    le=_mean_pts(pts,_MP_LEFT_EYE)
    re=_mean_pts(pts,_MP_RIGHT_EYE)
    no=_mean_pts(pts,_MP_NOSE)
    ml=_mean_pts(pts,[61,78,80,81,82,13])
    mr=_mean_pts(pts,[291,308,310,311,312,14])
    out=np.array([re,le,no,mr,ml],np.float32)
    return out if np.isfinite(out).all() else None

def _dense_face_box(pts):
    if pts is None or len(pts)<400:return None
    xy=pts[:,:2]
    x0,y0=np.percentile(xy,[2,2],axis=0)
    x1,y1=np.percentile(xy,[98,98],axis=0)
    # Mesh box is intentionally slightly contracted to skin/face area.
    bw=max(10.0,float(x1-x0)); bh=max(10.0,float(y1-y0))
    return np.array([x0,y0,bw,bh],np.float32)

def refine_landmarks(im, yf, is_target=False):
    """Refine YuNet's five points with dense landmarks when the mesh agrees.

    Returns (face_record, metadata). For targets with severe corruption, the
    function deliberately keeps YuNet's geometry instead of hallucinating a
    dense mesh.
    """
    if yf is None:return yf,{"method":"none","confidence":0.0,"used":False}
    pts,_=_mp_points(im)
    anchors=_dense_anchors(pts) if pts is not None else None
    if anchors is None:
        return yf,{"method":"yunet","confidence":0.0,"used":False}
    box=_dense_face_box(pts)
    if box is None:
        return yf,{"method":"yunet","confidence":0.0,"used":False}
    # Require dense geometry to agree with detector box. This protects against
    # MediaPipe locking onto a face-like background object.
    ybox=np.asarray(yf[:4],np.float32)
    yc=np.array([ybox[0]+ybox[2]*.5,ybox[1]+ybox[3]*.5])
    dc=np.array([box[0]+box[2]*.5,box[1]+box[3]*.5])
    center_err=np.linalg.norm(dc-yc)/max(1.0,np.mean(ybox[2:4]))
    size_ratio=(box[2]*box[3])/(max(1.0,ybox[2]*ybox[3]))
    if center_err>0.30 or size_ratio<0.45 or size_ratio>2.2:
        return yf,{"method":"yunet","confidence":0.0,"used":False}
    # Build a YuNet-compatible record so the rest of v9.1 can stay unchanged.
    out=np.asarray(yf,dtype=np.float32).copy()
    out[:4]=box
    out[4:14]=anchors.reshape(-1)
    # Dense confidence is a conservative agreement score.
    conf=float(max(0.0,min(1.0,1.0-center_err/.30)))
    if len(out)>=15: out[14]=max(float(out[14]),0.55*conf)
    return out,{"method":"mediapipe_dense","confidence":conf,"used":True,"points":int(len(pts))}


def detector():
    global _DETECTOR
    if _DETECTOR is None:
        if not MODEL.exists():
            raise FileNotFoundError(
                f"Missing model:\n{MODEL}\n\n"
                "Put face_detection_yunet_2026may.onnx beside this script."
            )
        _DETECTOR = cv2.FaceDetectorYN.create(
            str(MODEL), "", (320, 320), 0.35, 0.3, 5000
        )
    return _DETECTOR


def _detect_once(d, im):
    h, w = im.shape[:2]
    d.setInputSize((w, h))
    _, faces = d.detect(im)
    return faces


def _face_geometry_ok(f, shape):
    """Reject obviously broken YuNet landmark geometry."""
    if f is None or len(f) < 15:
        return False
    h, w = shape[:2]
    x, y, bw, bh = map(float, f[:4])
    if bw < max(12, w * 0.025) or bh < max(12, h * 0.025):
        return False
    if x + bw < 0 or y + bh < 0 or x > w or y > h:
        return False

    p = lm5(f)
    if not np.isfinite(p).all():
        return False

    # YuNet order: right eye, left eye, nose, right mouth, left mouth.
    eye_y = (p[0, 1] + p[1, 1]) * .5
    mouth_y = (p[3, 1] + p[4, 1]) * .5
    if not (y - .35 * bh <= eye_y <= y + .75 * bh):
        return False
    if not (y - .10 * bh <= mouth_y <= y + 1.10 * bh):
        return False
    if not (eye_y < mouth_y):
        return False

    # Landmarks should remain near the detector box.
    if np.any(p[:, 0] < x - .45 * bw) or np.any(p[:, 0] > x + 1.45 * bw):
        return False
    if np.any(p[:, 1] < y - .45 * bh) or np.any(p[:, 1] > y + 1.45 * bh):
        return False

    return True


def _face_candidate_score(f, shape, prefer_center=True):
    h, w = shape[:2]
    x, y, bw, bh = map(float, f[:4])
    area = bw * bh
    if area <= 0:
        return -1e18

    cx = x + bw * .5
    cy = y + bh * .5
    central = 1.0 - min(1.0, abs(cx - w * .5) / max(1.0, w * .5))
    size = min(1.0, area / max(1.0, w * h * .08))

    conf = float(f[14]) if len(f) >= 15 else .5
    conf = max(0.0, min(1.0, conf))

    # Mild center preference, but do not overpower detector confidence/size.
    center_term = (.35 + .65 * central) if prefer_center else 1.0
    geom = 1.0 if _face_geometry_ok(f, shape) else .15
    return area * center_term * (.25 + .75 * conf) * geom * (.35 + .65 * size)


def detect_face(im, allow_blur=True, return_meta=False):
    """
    Robust multi-pass YuNet detection.

    Returns the best validated face.  Multiple image representations are
    tested because the target face may be heavily pixelated/occluded.
    """
    d = detector()
    h, w = im.shape[:2]
    max_side = max(h, w)

    # Keep detector input in a practical range, but also upscale small targets.
    base_scale = min(1.0, 1000.0 / max_side)
    if max_side < 700:
        base_scale = min(2.0, 700.0 / max_side)

    work = cv2.resize(
        im,
        (max(1, int(w * base_scale)), max(1, int(h * base_scale))),
        interpolation=cv2.INTER_CUBIC if base_scale > 1 else cv2.INTER_AREA
    )

    variants = [("original", work)]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    variants.append(("gray", gray3))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)))

    if allow_blur:
        variants.append(("enhanced", cv2.detailEnhance(work, sigma_s=10, sigma_r=.15)))
        sharp = cv2.GaussianBlur(work, (0, 0), 1.0)
        variants.append(("sharpen", cv2.addWeighted(work, 1.35, sharp, -.35, 0)))

    candidates = []
    for name, v in variants:
        faces = _detect_once(d, v)
        if faces is None:
            continue
        for f in faces:
            ff = f.copy()
            ff[:14] /= base_scale
            candidates.append((name, ff))

    if not candidates:
        return (None, {
            "confidence": 0.0,
            "method": "none",
            "stable": False,
            "candidates": 0
        }) if return_meta else None

    # Cluster candidates by box center/size. Agreement between independent
    # passes is a strong signal that this is the real face.
    clusters = []
    for method, f in candidates:
        x, y, bw, bh = map(float, f[:4])
        cx, cy = x + bw*.5, y + bh*.5
        assigned = False
        for c in clusters:
            _, bx, by, bbw, bbh, members = c
            if abs(cx-bx) < .18*max(bw, bbw) and abs(cy-by) < .18*max(bh, bbh):
                if abs(np.log(max(1,bw)/max(1,bbw))) < .22 and abs(np.log(max(1,bh)/max(1,bbh))) < .22:
                    members.append((method, f))
                    # Running center/size.
                    n = len(members)
                    c[1] = (c[1]*(n-1)+cx)/n
                    c[2] = (c[2]*(n-1)+cy)/n
                    c[3] = (c[3]*(n-1)+bw)/n
                    c[4] = (c[4]*(n-1)+bh)/n
                    assigned = True
                    break
        if not assigned:
            clusters.append([len(clusters), cx, cy, bw, bh, [(method, f)]])

    ranked = []
    for c in clusters:
        members = c[5]
        best_method, best_f = max(
            members,
            key=lambda mf: _face_candidate_score(mf[1], (h, w))
        )
        scores = [_face_candidate_score(f, (h, w)) for _, f in members]
        mean_score = float(np.mean(scores))
        agreement = min(1.0, len(members) / 3.0)
        geom = 1.0 if _face_geometry_ok(best_f, (h, w)) else 0.0
        conf = float(best_f[14]) if len(best_f) >= 15 else .5
        conf = max(0.0, min(1.0, conf))

        # Confidence combines detector confidence, geometry and cross-pass agreement.
        confidence = .40*conf + .35*agreement + .25*geom
        ranked.append((confidence, mean_score, best_f, best_method, len(members)))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    confidence, _, best, method, count = ranked[0]

    # Do not trust a geometrically broken detection.
    stable = bool(_face_geometry_ok(best, (h, w)) and confidence >= .42)

    meta = {
        "confidence": float(confidence),
        "method": method,
        "stable": stable,
        "candidates": len(candidates),
        "agreement": count,
        "clusters": len(clusters),
    }

    if not stable:
        return (None, meta) if return_meta else None

    return (best, meta) if return_meta else best


def lm5(f):
    return np.asarray(f[4:14],np.float32).reshape(5,2)

def fallback_target_from_source(srcf, target):
    # Only used if pixelation defeats target face detection.
    # Search target upper-center for a similarly sized face region by template edges.
    h,w=target.shape[:2]
    x,y,bw,bh=srcf[:4]
    # conservative centered estimate; user can fine-tune mask but no source rectangle can leak
    size=min(w*.30,h*.28)
    cx=w*.5; cy=h*.19
    return np.array([cx-size/2,cy-size/2,size,size,
                     cx-size*.18,cy-size*.08, cx+size*.18,cy-size*.08,
                     cx,cy+size*.05, cx-size*.13,cy+size*.20, cx+size*.13,cy+size*.20,
                     .1],np.float32)

def target_occlusion_score(target, f):
    """
    Estimate whether the target facial interior is heavily corrupted/covered.
    A large flat/color-dominant region inside the face is treated as unreliable
    evidence for landmark alignment.
    """
    h, w = target.shape[:2]
    x, y, bw, bh = map(float, f[:4])
    x0 = max(0, int(x + .12*bw)); x1 = min(w, int(x + .88*bw))
    y0 = max(0, int(y + .10*bh)); y1 = min(h, int(y + .90*bh))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = target[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Flatness and edge scarcity are useful signals for a covered/pixelated face.
    std = float(np.std(gray))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 90)
    edge_density = float(np.mean(edges > 0))

    flat = max(0.0, min(1.0, (38.0 - std) / 38.0))
    low_edges = max(0.0, min(1.0, (.045 - edge_density) / .045))

    # Saturated solid-color blocks are common in redaction/painted-over targets.
    sat = hsv[...,1].astype(np.float32)
    strong_sat = float(np.mean(sat > 150))
    solid_color = max(0.0, min(1.0, (strong_sat - .35) / .50))

    return float(.45*flat + .25*low_edges + .30*solid_color)



def _project_dense_to_face_box(src_pts, src_face, dst_face):
    """Project source dense landmarks into a recovered target face box.

    This is specifically for targets whose facial pixels are blurred/painted
    over so strongly that YuNet/MediaPipe cannot see individual features.
    It preserves the source's dense frontal topology while changing only the
    global face box. It is never used for a detected target mesh.
    """
    if src_pts is None or len(src_pts) < 400 or src_face is None or dst_face is None:
        return None
    sx,sy,sw,sh=map(float,src_face[:4])
    dx,dy,dw,dh=map(float,dst_face[:4])
    if sw < 10 or sh < 10 or dw < 10 or dh < 10:
        return None
    out=np.asarray(src_pts,dtype=np.float32).copy()
    out[:,0]=dx+(out[:,0]-sx)/sw*dw
    out[:,1]=dy+(out[:,1]-sy)/sh*dh
    # Keep z as a weak source-shape cue; scale it to the recovered face size.
    out[:,2]=out[:,2]*(dw/max(sw,1.0))
    return out

def _recovered_face_confidence(im, f):
    """Estimate whether a geometry fallback is face-like, including blurred faces."""
    try:
        h,w=im.shape[:2]
        x,y,bw,bh=map(float,f[:4])
        if bw<12 or bh<12 or x<0 or y<0 or x+bw>w or y+bh>h:
            return 0.0
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        edge=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),25,90).astype(np.float32)/255.0
        yy,xx=np.ogrid[:h,:w]
        cx=x+bw*.5; cy=y+bh*.50
        rx=bw*.50; ry=bh*.55
        ell=((xx-cx)/max(rx,1))**2+((yy-cy)/max(ry,1))**2
        inner=ell<=.78
        ring=(ell<=1.0)&(ell>=.70)
        outer=(ell>=1.05)&(ell<=1.35)
        if ring.sum()<20:return 0.0
        ring_e=float(edge[ring].mean())
        outer_e=float(edge[outer].mean()) if outer.any() else ring_e
        contrast=max(0.0,min(1.0,(ring_e-outer_e+.015)/.10))
        # A blurred/painted face can have almost no inner detail, so reward
        # a quiet center rather than treating it as a failed detection.
        inner_e=float(edge[inner].mean()) if inner.any() else 0.0
        quiet=max(0.0,min(1.0,1.0-inner_e/.18))
        return float(np.clip(.65*contrast+.35*quiet,0,1))
    except Exception:
        return 0.0


def _detect_hidden_face_box(im, srcf=None):
    """Recover the actual target face box when the face is redacted/blurred."""
    try:
        h,w=im.shape[:2]
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roi=np.zeros((h,w),np.uint8); roi[int(.04*h):int(.58*h),int(.22*w):int(.78*w)]=255
        sat=hsv[...,1].astype(np.float32)/255.0; val=hsv[...,2].astype(np.float32)/255.0
        chroma=((sat>.42)&(val>.12)&(roi>0)).astype(np.uint8)*255
        chroma=cv2.morphologyEx(chroma,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        chroma=cv2.morphologyEx(chroma,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        n,lab,stats,cent=cv2.connectedComponentsWithStats(chroma,8); best=None
        for i in range(1,n):
            x,y,bw,bh,area=stats[i]
            if area<max(80,int(.00008*w*h)) or bw<.12*bh or bh<.35*bw: continue
            if bw<.05*w or bw>.28*w or bh<.045*h or bh>.28*h: continue
            cx,cy=cent[i]; center=max(0.0,1.0-abs(cx-w*.5)/(w*.32)); upper=max(0.0,1.0-abs(cy-h*.22)/(h*.30))
            fill=float(area)/max(1,bw*bh); aspect=1.0-min(1.0,abs((bw/max(bh,1))-.82)/.82)
            score=.38*center+.22*upper+.22*min(1.0,fill*1.6)+.18*aspect
            if best is None or score>best[0]: best=(score,x,y,bw,bh)
        if best is not None and best[0]>=.34:
            _,x,y,bw,bh=best; cx=x+bw*.5; cy=y+bh*.5
            fw=max(bw*1.42,bh*.92); fh=max(bh*1.34,fw*1.05)
            if srcf is not None:
                sw,sh=float(srcf[2]),float(srcf[3]); fw=min(max(fw,sw*.55),sw*1.75); fh=min(max(fh,sh*.55),sh*1.75)
            x=max(0.0,min(w-fw,cx-fw*.5)); y=max(0.0,min(h-fh,cy-fh*.48))
            return np.array([x,y,fw,fh],np.float32),float(best[0]),"redaction_region"
        # Generic actual blur region: quiet center with a stronger surrounding contour.
        edges=cv2.Canny(cv2.GaussianBlur(gray,(9,9),0),20,70).astype(np.float32)/255.0; best=None
        yy,xx=np.ogrid[:h,:w]
        for frac in (.16,.19,.22,.25,.28):
            bw=w*frac; bh=bw*1.10
            for cxf in (.42,.46,.50,.54,.58):
                for cyf in (.16,.21,.26,.31,.36):
                    cx=w*cxf; cy=h*cyf; rx=bw*.5; ry=bh*.52; e=((xx-cx)/rx)**2+((yy-cy)/ry)**2
                    inner=e<=.62; ring=(e<=1.0)&(e>=.72)
                    if ring.sum()<30: continue
                    ie=float(edges[inner].mean()); re=float(edges[ring].mean())
                    score=(1-ie)*.58+min(1,re/.12)*.32+(1-abs(cxf-.5))*.10
                    if best is None or score>best[0]: best=(score,cx-bw*.5,cy-bh*.48,bw,bh)
        if best is not None and best[0]>=.62:
            _,x,y,bw,bh=best; return np.array([x,y,bw,bh],np.float32),float(best[0]),"blur_region"
    except Exception: pass
    return None,0.0,"none"

def fallback_target_geometry(clear, target, srcf):
    """
    Better fallback than the old fixed 50%/19% guess.

    Search a conservative set of upper/middle image positions and sizes.
    The candidates are evaluated from visible edge structure outside the
    corrupted central facial area.  This is intentionally conservative.
    """
    h, w = target.shape[:2]
    sbw, sbh = float(srcf[2]), float(srcf[3])

    # Typical portrait-face proportions. Avoid an enormous search.
    size_fracs = (.22, .26, .30, .34, .38)
    x_fracs = (.35, .42, .50, .58, .65)
    y_fracs = (.18, .24, .30, .36, .42)

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 100).astype(np.float32)
    best = None

    for sf in size_fracs:
        size = min(w*sf, h*.38)
        for xf in x_fracs:
            for yf in y_fracs:
                cx = w*xf
                cy = h*yf
                x = cx-size*.5
                y = cy-size*.52
                if x < 0 or y < 0 or x+size >= w or y+size*1.08 >= h:
                    continue

                # Score a ring around the candidate: head/hair/jaw structure,
                # rather than the potentially corrupted center.
                yy, xx = np.ogrid[:h, :w]
                rx, ry = size*.48, size*.55
                ell = ((xx-cx)/max(1,rx))**2 + ((yy-(y+size*.52))/max(1,ry))**2
                ring = (ell <= 1.0) & (ell >= .72)
                if ring.sum() < 20:
                    continue

                es = float(np.mean(edge[ring]))
                center_penalty = abs(xf-.5)*.15 + abs(yf-.28)*.08
                score = es - center_penalty

                if best is None or score > best[0]:
                    best = (score, x, y, size)

    if best is None:
        size=min(w*.30,h*.28)
        cx=w*.5; cy=h*.25
        x=cx-size/2; y=cy-size*.52
    else:
        _,x,y,size=best
        cx=x+size/2; cy=y+size*.52

    return np.array([
        x,y,size,size,
        cx-size*.18, cy-size*.08,
        cx+size*.18, cy-size*.08,
        cx, cy+size*.05,
        cx-size*.13, cy+size*.20,
        cx+size*.13, cy+size*.20,
        .1
    ], np.float32)


def guarded_similarity_from_faces(sf, tf, occlusion=0.0):
    """
    Landmark alignment when landmarks are trustworthy.
    Under heavy target occlusion, use box geometry and suppress unreliable
    landmark influence.
    """
    if occlusion < .55 and _face_geometry_ok(tf, (10**6,10**6)):
        M = similarity_from_faces(sf, tf)
        return M.astype(np.float32)

    # Box-based transform: robust when target eyes/nose/mouth are unavailable.
    sx = ((tf[2] / max(1, sf[2])) + (tf[3] / max(1, sf[3]))) * .5
    scx = sf[0] + sf[2]/2
    scy = sf[1] + sf[3]*.48
    tcx = tf[0] + tf[2]/2
    tcy = tf[1] + tf[3]*.48
    return np.array([
        [sx, 0, tcx-sx*scx],
        [0, sx, tcy-sx*scy]
    ], np.float32)


def similarity_from_faces(sf,tf):
    S=lm5(sf); T=lm5(tf)
    # Eye + mouth landmarks are most useful for scale/rotation; nose stabilizes translation.
    M,inl=cv2.estimateAffinePartial2D(S,T,method=cv2.LMEDS)
    if M is None:
        sx=(tf[2]/max(1,sf[2])+tf[3]/max(1,sf[3]))*.5
        scx=sf[0]+sf[2]/2; scy=sf[1]+sf[3]/2
        tcx=tf[0]+tf[2]/2; tcy=tf[1]+tf[3]/2
        M=np.array([[sx,0,tcx-sx*scx],[0,sx,tcy-sx*scy]],np.float32)
    return M.astype(np.float32)

def source_face_mask(shape,f,scale=1.0):
    """Tight face-shaped source crop.

    This follows the face/head region rather than using a broad ellipse.
    `scale=1` is the tight crop; values above 1 add a controlled margin.
    """
    h,w=shape[:2]
    x,y,bw,bh=map(float,f[:4])
    cx=x+bw*.5
    cy=y+bh*.50
    s=max(.75,float(scale))

    pts=np.array([
        [cx,        y-.035*bh],
        [x+.14*bw,  y+.045*bh],
        [x+.015*bw, y+.235*bh],
        [x-.025*bw, y+.48*bh],
        [x+.045*bw, y+.72*bh],
        [x+.23*bw,  y+.95*bh],
        [cx,        y+1.14*bh],
        [x+.77*bw,  y+.95*bh],
        [x+.955*bw, y+.72*bh],
        [x+1.025*bw,y+.48*bh],
        [x+.985*bw, y+.235*bh],
        [x+.86*bw,  y+.045*bh],
    ],np.float32)

    pts[:,0]=cx+(pts[:,0]-cx)*s
    pts[:,1]=cy+(pts[:,1]-cy)*s

    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],255)

    # Very small smoothing keeps the contour natural while preserving the
    # tight face crop.
    k=max(3,int(min(bw,bh)*.025))
    if k%2==0:k+=1
    mask=cv2.GaussianBlur(mask,(k,k),0)

    # Hard safety ROI: never let the face crop reach torso/background.
    roi=np.zeros_like(mask)
    x0=max(0,int(x-.12*bw)); x1=min(w,int(x+1.12*bw))
    y0=max(0,int(y-.10*bh)); y1=min(h,int(y+1.18*bh))
    roi[y0:y1,x0:x1]=255
    return cv2.bitwise_and(mask,roi)


def target_face_mask(shape,f,scale=1.0):
    h,w=shape[:2]
    x,y,bw,bh=map(float,f[:4])
    cx=x+bw*.5
    cy=y+bh*.50
    s=max(.75,float(scale))

    pts=np.array([
        [cx,        y-.035*bh],
        [x+.14*bw,  y+.045*bh],
        [x+.015*bw, y+.235*bh],
        [x-.025*bw, y+.48*bh],
        [x+.045*bw, y+.72*bh],
        [x+.23*bw,  y+.95*bh],
        [cx,        y+1.14*bh],
        [x+.77*bw,  y+.95*bh],
        [x+.955*bw, y+.72*bh],
        [x+1.025*bw,y+.48*bh],
        [x+.985*bw, y+.235*bh],
        [x+.86*bw,  y+.045*bh],
    ],np.float32)

    pts[:,0]=cx+(pts[:,0]-cx)*s
    pts[:,1]=cy+(pts[:,1]-cy)*s

    m=np.zeros((h,w),np.uint8)
    cv2.fillPoly(m,[np.round(pts).astype(np.int32)],255)

    k=max(3,int(min(bw,bh)*.025))
    if k%2==0:k+=1
    return cv2.GaussianBlur(m,(k,k),0)


def _frontal_pose_score(pts):
    """Estimate whether a dense mesh is sufficiently frontal for geometry warp.

    Returns 0..1.  This deliberately uses several weak cues rather than one
    brittle yaw estimate, because the target may be blurred.
    """
    if pts is None or len(pts) < 400:
        return 0.0
    try:
        le=_mean_pts(pts,[33,133,159,145]); re=_mean_pts(pts,[362,263,386,374])
        no=_mean_pts(pts,[1,4,5,6,197])
        ml=_mean_pts(pts,[61,78,80,81,82,13]); mr=_mean_pts(pts,[291,308,310,311,312,14])
        if not np.isfinite(np.array([le,re,no,ml,mr])).all(): return 0.0
        eye_mid=(le+re)*.5; mouth_mid=(ml+mr)*.5
        eye_w=float(np.linalg.norm(re-le))+1e-6
        # Nose should be near the midline in a frontal view.
        nose_offset=abs(float(no[0]-eye_mid[0]))/eye_w
        mouth_offset=abs(float(((ml+mr)*.5)[0]-eye_mid[0]))/eye_w
        # Bilateral eye/mouth widths should be reasonably symmetric.
        left_eye=float(np.linalg.norm(le-no)); right_eye=float(np.linalg.norm(re-no))
        left_m=float(np.linalg.norm(ml-no)); right_m=float(np.linalg.norm(mr-no))
        eye_sym=1.0-abs(left_eye-right_eye)/max(left_eye,right_eye,1e-6)
        mouth_sym=1.0-abs(left_m-right_m)/max(left_m,right_m,1e-6)
        z=pts[:,2]
        # MediaPipe z asymmetry is useful as a weak secondary yaw cue.
        zscore=1.0
        if np.isfinite(z).all():
            zi=float(np.median(z[362:477])) if len(z)>477 else 0.0
            zl=float(np.median(z[33:263])) if len(z)>263 else 0.0
            zscore=max(0.0,1.0-min(1.0,abs(zi-zl)/(0.12*eye_w+1e-6)))
        a=1.0-min(1.0,nose_offset/.22)
        b=1.0-min(1.0,mouth_offset/.20)
        return float(np.clip(.34*a+.18*b+.28*eye_sym+.12*mouth_sym+.08*zscore,0,1))
    except Exception:
        return 0.0


def _frontal_landmark_indices(n):
    """Stable facial mesh subset for frontal piecewise-affine warping."""
    groups=[
        # outer face / jaw contour
        list(range(10,18))+list(range(54,69))+list(range(103,118))+list(range(127,150)),
        # brows / eyes
        [33,46,52,55,65,70,105,107,133,144,145,153,154,155,157,158,159,160,161,163,173,
         263,276,282,285,295,300,334,336,362,373,374,380,381,382,384,385,386,387,388,390,398],
        # nose
        [1,2,4,5,6,19,20,94,97,98,99,168,195,197,236,237,238,239,240,241,242,243,244,245,460,461,462,463,464,465,466,467],
        # mouth / lips
        [0,11,12,13,14,15,16,17,37,39,40,61,72,73,74,77,78,80,81,82,84,85,87,88,89,
         267,269,270,291,302,303,304,307,308,310,311,312,314,315,317,318,320,321,323],
    ]
    ids=[]
    for g in groups:
        ids.extend(g)
    ids=sorted(set(i for i in ids if i < n))
    return ids


def _piecewise_dense_warp(clear, target, spts, tpts, mask):
    """Warp only the source face using a target-driven triangular mesh.

    The mesh is built in target coordinates, so frontal faces get local shape
    correction (jaw width, eye spacing, nose/mouth position) instead of one
    global similarity transform. Outside the source face mask nothing is used.
    """
    h,w=target.shape[:2]
    valid=(np.isfinite(spts).all(axis=1)&np.isfinite(tpts).all(axis=1))
    s=spts[valid].astype(np.float32); t=tpts[valid].astype(np.float32)
    if len(s)<30:
        return None,None

    # Keep only points inside a conservative source face ROI and target image.
    smask=mask>8
    keep=[]
    for i,p in enumerate(s):
        x,y=p
        xi=int(round(x)); yi=int(round(y))
        if 0<=xi<clear.shape[1] and 0<=yi<clear.shape[0] and smask[yi,xi]:
            q=t[i]
            if -2<=q[0]<w+2 and -2<=q[1]<h+2:
                keep.append(i)
    if len(keep)<30:
        return None,None
    s=s[keep]; t=t[keep]

    # Remove near-duplicates to keep Subdiv2D stable.
    uniq=[]; seen=set()
    for i,p in enumerate(t):
        key=(int(round(float(p[0]))),int(round(float(p[1]))))
        if key not in seen:
            seen.add(key); uniq.append(i)
    s=s[uniq]; t=t[uniq]
    if len(t)<25:return None,None

    subdiv=cv2.Subdiv2D((0,0,w,h))
    for p in t:
        x=float(np.clip(p[0],0,w-1)); y=float(np.clip(p[1],0,h-1))
        try: subdiv.insert((x,y))
        except Exception: pass
    tri=subdiv.getTriangleList()
    if tri is None or len(tri)==0:return None,None

    out=np.zeros((h,w,3),np.uint8)
    outmask=np.zeros((h,w),np.uint8)
    target_xy=t
    for tr in tri:
        pts=tr.reshape(3,2)
        ids=[]
        ok=True
        for p in pts:
            d=np.sum((target_xy-p[None,:])**2,axis=1)
            j=int(np.argmin(d))
            if d[j]>4.0: ok=False; break
            ids.append(j)
        if not ok or len(set(ids))<3:continue
        ti=target_xy[ids].astype(np.float32); si=s[ids].astype(np.float32)
        xmin=max(0,int(np.floor(ti[:,0].min()))); xmax=min(w-1,int(np.ceil(ti[:,0].max())))
        ymin=max(0,int(np.floor(ti[:,1].min()))); ymax=min(h-1,int(np.ceil(ti[:,1].max())))
        if xmax<=xmin or ymax<=ymin:continue
        patch=np.zeros((ymax-ymin+1,xmax-xmin+1),np.uint8)
        local=np.round(ti-[xmin,ymin]).astype(np.int32)
        cv2.fillConvexPoly(patch,local,255)
        A=cv2.getAffineTransform(si,ti)
        warped=cv2.warpAffine(clear,A,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        # Restrict each triangle to the original source face mask after inverse mapping.
        srcm=cv2.warpAffine(mask,A,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        tri_mask=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(tri_mask,np.round(ti).astype(np.int32),255)
        use=cv2.bitwise_and(tri_mask,srcm)
        out[use>0]=warped[use>0]
        outmask[use>0]=255
    return out,outmask


def build(clear,target):
    sf, smeta = detect_face(clear, False, return_meta=True)
    if sf is None:
        raise ValueError("No reliable face detected in CLEAR image.")

    tf, tmeta = detect_face(target, True, return_meta=True)
    target_detected = tf is not None
    target_recovered = False
    fallback_conf = 0.0
    recovery_method = "none"
    if tf is None:
        hidden_box, hidden_conf, hidden_method = _detect_hidden_face_box(target, sf)
        if hidden_box is not None:
            hx,hy,hw,hh=map(float,hidden_box)
            tf=np.array([hx,hy,hw,hh,hx+hw*.32,hy+hh*.34,hx+hw*.68,hy+hh*.34,hx+hw*.50,hy+hh*.53,hx+hw*.36,hy+hh*.72,hx+hw*.64,hy+hh*.72,.12],np.float32)
            fallback_conf=max(hidden_conf,_recovered_face_confidence(target,tf)); target_recovered=bool(fallback_conf>=.30); recovery_method=hidden_method
        else:
            tf=fallback_target_geometry(clear,target,sf); fallback_conf=_recovered_face_confidence(target,tf); target_recovered=bool(fallback_conf>=.18); recovery_method="geometry_search"
        target_detected=False

    sf_dense, srmeta = refine_landmarks(clear, sf, is_target=False)
    if target_detected:
        tf_dense, trmeta = refine_landmarks(target, tf, is_target=True)
    else:
        tf_dense, trmeta = tf, {"method":"fallback_geometry", "confidence":fallback_conf, "used":False}
    sf,tf=sf_dense,tf_dense
    occ=target_occlusion_score(target,tf)

    src_pts,_=_mp_points(clear)
    tgt_pts,_=_mp_points(target) if target_detected else (None,0.0)
    if target_recovered and tgt_pts is None and src_pts is not None:
        # Critical blurred-target recovery: synthesize target mesh geometry from
        # the recovered face box rather than forcing the old coarse similarity.
        projected=_project_dense_to_face_box(src_pts,sf,tf)
        if projected is not None:
            tgt_pts=projected
            trmeta={"method":"projected_dense_fallback", "confidence":fallback_conf, "used":True}

    src_pose=_frontal_pose_score(src_pts)
    if target_recovered and tgt_pts is not None:
        # The recovered mesh is topology-preserving from the source, so its
        # pose is intentionally inherited from the source only in this
        # fallback case. A real target mesh always takes precedence.
        tgt_pose=_frontal_pose_score(tgt_pts)
    else:
        tgt_pose=_frontal_pose_score(tgt_pts)
    frontal=bool((target_detected or target_recovered) and src_pose>=.62 and tgt_pose>=.58 and occ<.78)

    # v10 path remains the default for side/oblique poses.
    dense_target_ok=bool(trmeta.get("used",False) and trmeta.get("confidence",0)>=.38 and occ<.78)
    h,w=target.shape[:2]
    sm=source_face_mask(clear.shape,sf,1.18)
    sm=expand_mask(sm,115)
    tm=target_face_mask(target.shape,tf,1.18)
    tm=expand_mask(tm,115)

    if frontal and src_pts is not None and tgt_pts is not None:
        ids=_frontal_landmark_indices(min(len(src_pts),len(tgt_pts)))
        warped,wm=_piecewise_dense_warp(clear,target,src_pts[ids,:2],tgt_pts[ids,:2],sm)
        if warped is not None and wm is not None and np.count_nonzero(wm)>0:
            aligned=warped
            valid=wm
            # Target mask remains a hard boundary: no source background leakage.
            mask=cv2.min(valid,tm)
            align_method="frontal_dense_piecewise"
        else:
            M=similarity_from_faces(sf,tf)
            aligned=cv2.warpAffine(clear,M,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
            valid=cv2.warpAffine(sm,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            mask=cv2.min(valid,tm)
            align_method="frontal_similarity_fallback"
    else:
        if dense_target_ok:
            M=similarity_from_faces(sf,tf); align_method="dense_landmarks"
        else:
            M=guarded_similarity_from_faces(sf,tf,occ)
            align_method="yunet_guarded" if target_detected else "geometry_fallback"
        aligned=cv2.warpAffine(clear,M,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        valid=cv2.warpAffine(sm,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        mask=cv2.min(valid,tm)

    meta={
        "source_confidence":float(smeta.get("confidence",0)),
        "target_confidence":float(tmeta.get("confidence",0)),
        "target_detected":bool(target_detected),
        "target_recovered":bool(target_recovered),
        "fallback_confidence":float(fallback_conf),
        "occlusion":float(occ),
        "target_method":(recovery_method if target_recovered and not target_detected else ("projected_dense_fallback" if target_recovered and trmeta.get("used") else tmeta.get("method","fallback"))),
        "target_recovery_method":(recovery_method if target_recovered else "none"),
        "source_landmarks":srmeta.get("method","yunet"),
        "target_landmarks":trmeta.get("method","yunet"),
        "dense_source_confidence":float(srmeta.get("confidence",0)),
        "dense_target_confidence":float(trmeta.get("confidence",0)),
        "source_frontal_score":float(src_pose),
        "target_frontal_score":float(tgt_pose),
        "frontal_mode":bool(frontal),
        "alignment_method":align_method,
        # Frontal dense warp is already locally fitted; only a tiny final search
        # is permitted. Side/oblique behavior retains v10's Auto Fine policy.
        "safe_auto_fine":bool((target_detected or target_recovered) and occ<.72),
        "dense_alignment":dense_target_ok,
    }
    return aligned,mask,valid,sf,tf,target_detected,meta

def expand_mask(mask, percent, reference=None):
    """Expand a face mask without changing its center/transform.

    Expansion is applied as a morphology operation, so source validity and
    target coverage can grow together.  It is deliberately bounded by the
    existing valid/source mask at the caller.
    """
    p=max(100.0,float(percent))
    if p<=100.0:
        return mask
    ys,xs=np.where(mask>8)
    if len(xs)<20:
        return mask
    fw=max(20,int(xs.max()-xs.min()+1))
    fh=max(20,int(ys.max()-ys.min()+1))
    # Percent describes approximate total width/height increase.
    radius=max(1,int(min(fw,fh)*(p/100.0-1.0)*.5))
    radius=min(radius, max(1,int(min(fw,fh)*.16)))
    k=radius*2+1
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    return cv2.dilate(mask,ker,iterations=1)

def blend(aligned,target,mask,feather):
    soft=cv2.GaussianBlur(mask,(0,0),max(1,float(feather)))
    a=soft.astype(np.float32)[...,None]/255
    out=aligned.astype(np.float32)*a+target.astype(np.float32)*(1-a)
    return np.clip(out,0,255).astype(np.uint8)


PROFILE_DIR=HERE/"face_profiles"
PROFILE_DIR.mkdir(exist_ok=True)

def _profile_path(name):
    safe="".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in name).strip()
    if not safe:
        safe="default"
    return PROFILE_DIR/(safe+".json")

def save_face_profile(name, data):
    p=_profile_path(name)
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)

def load_face_profile(name):
    p=_profile_path(name)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {name}")
    return json.loads(p.read_text(encoding="utf-8"))

def list_face_profiles():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))

class App:
    def __init__(self,r):
        self.r=r;r.title("Auto Face Restore v11.2 — Frontal + Side Dense Alignment");r.geometry("1150x880")
        self.clear=self.target=self.aligned=self.mask=self.base=self.result=self.valid=None
        self.scale=1;self.off=(0,0);self.paint=False;self.last=None;self.undo_mask=None
        self.zoom=1.0; self.pan_x=0; self.pan_y=0; self._last_preview_t=0
        self.shift_x=0; self.shift_y=0
        self.base_aligned=None; self.base_mask_auto=None; self.base_valid=None
        self.align_meta={"source_confidence":0.0,"target_confidence":0.0,"occlusion":0.0,"safe_auto_fine":False}
        self.profile_name=tk.StringVar(value="")
        self.profile_status=tk.StringVar(value="No profile")
        self.profile_data=None
        self.profile_locked=False
        b=tk.Frame(r);b.pack(fill="x",padx=8,pady=6)
        for text,cmd in [("1. CLEAR Original",self.lc),("2. BLURRED Target",self.lt),
                         ("AUTO RESTORE",self.run),("AUTO FINE",self.auto_fine),("Save",self.save)]:
            tk.Button(b,text=text,command=cmd,height=2,width=20).pack(side="left",padx=3)

        pbar=tk.Frame(r);pbar.pack(fill="x",padx=8,pady=(0,4))
        tk.Label(pbar,text="Model Profile").pack(side="left")
        self.profile_combo=tk.OptionMenu(pbar,self.profile_name,"",*list_face_profiles())
        self.profile_combo.config(width=18)
        self.profile_combo.pack(side="left",padx=4)
        tk.Button(pbar,text="Save Current as Profile",command=self.save_profile).pack(side="left",padx=3)
        tk.Button(pbar,text="Apply Profile",command=self.apply_profile).pack(side="left",padx=3)
        tk.Button(pbar,text="Update Profile",command=self.update_profile).pack(side="left",padx=3)
        tk.Button(pbar,text="Lock Adjustment",command=self.lock_adjustment).pack(side="left",padx=3)
        tk.Button(pbar,text="Unlock",command=self.unlock_adjustment).pack(side="left",padx=3)
        tk.Label(pbar,textvariable=self.profile_status,anchor="w").pack(side="left",padx=8)
        t=tk.Frame(r);t.pack(fill="x",padx=8)
        self.mode=tk.StringVar(value="Erase")
        tk.Radiobutton(t,text="Eraser",variable=self.mode,value="Erase").pack(side="left")
        tk.Radiobutton(t,text="Restore",variable=self.mode,value="Restore").pack(side="left")
        tk.Label(t,text="Brush").pack(side="left",padx=(12,2))
        self.bs=tk.Scale(t,from_=5,to=120,orient="horizontal",length=150);self.bs.set(30);self.bs.pack(side="left")
        tk.Label(t,text="Feather").pack(side="left",padx=(12,2))
        self.fe=tk.Scale(t,from_=1,to=30,orient="horizontal",length=140);self.fe.set(8);self.fe.pack(side="left")
        tk.Label(t,text="Face Crop %").pack(side="left",padx=(12,2))
        self.facecrop=tk.Scale(t,from_=90,to=135,orient="horizontal",length=130,
                               resolution=1,command=self.face_crop_resize)
        self.facecrop.set(118);self.facecrop.pack(side="left")
        tk.Label(t,text="Mask Size %").pack(side="left",padx=(12,2))
        self.masksz=tk.Scale(t,from_=100,to=140,orient="horizontal",length=130,
                             resolution=1,command=self.mask_resize)
        self.masksz.set(115);self.masksz.pack(side="left")
        tk.Button(t,text="Apply Feather",command=self.rebuild).pack(side="left",padx=4)
        tk.Button(t,text="Undo Stroke",command=self.undo).pack(side="left",padx=4)
        tk.Button(t,text="Reset Mask",command=self.reset).pack(side="left",padx=4)
        move=tk.Frame(r);move.pack(fill="x",padx=8,pady=(3,0))
        tk.Label(move,text="Face Left / Right").pack(side="left")
        self.xslider=tk.Scale(move,from_=-120,to=120,orient="horizontal",length=240,
                              resolution=1,command=self.move_face)
        self.xslider.set(0);self.xslider.pack(side="left",padx=4)
        tk.Label(move,text="Face Up / Down").pack(side="left",padx=(12,0))
        self.yslider=tk.Scale(move,from_=-120,to=120,orient="horizontal",length=240,
                              resolution=1,command=self.move_face)
        self.yslider.set(0);self.yslider.pack(side="left",padx=4)
        tk.Button(move,text="Reset Transform",command=self.reset_position).pack(side="left",padx=8)

        transform=tk.Frame(r);transform.pack(fill="x",padx=8,pady=(3,0))
        tk.Label(transform,text="Face Size %").pack(side="left")
        self.sizeslider=tk.Scale(transform,from_=70,to=130,orient="horizontal",length=240,
                                 resolution=1,command=self.move_face)
        self.sizeslider.set(100);self.sizeslider.pack(side="left",padx=4)
        tk.Label(transform,text="Rotation °").pack(side="left",padx=(12,0))
        self.rotslider=tk.Scale(transform,from_=-20,to=20,orient="horizontal",length=240,
                                resolution=.5,command=self.move_face)
        self.rotslider.set(0);self.rotslider.pack(side="left",padx=4)
        tk.Label(transform,text="Zoom").pack(side="left",padx=(12,0))
        self.zoomslider=tk.Scale(transform,from_=50,to=400,orient="horizontal",length=180,
                                 resolution=10,command=self.set_zoom)
        self.zoomslider.set(100);self.zoomslider.pack(side="left",padx=4)
        tk.Button(transform,text="Fit",command=self.fit_view).pack(side="left",padx=4)

        self.status=tk.Label(r,text="v11 FRONTAL + SIDE ALIGNMENT: dense frontal geometry + v10 side fallback + guarded masks.")
        self.status.pack(pady=4)
        self.confidence_label=tk.Label(
            r,
            text="Alignment: —",
            anchor="w"
        )
        self.confidence_label.pack(fill="x",padx=8)
        self.c=tk.Canvas(r,bg="#222",highlightthickness=0);self.c.pack(fill="both",expand=True,padx=8,pady=6)
        self.c.bind("<Button-1>",self.down);self.c.bind("<B1-Motion>",self.drag);self.c.bind("<ButtonRelease-1>",self.up)
        self.c.bind("<Motion>",self.motion)
        self.c.bind("<MouseWheel>",self.wheel)
        self.c.bind("<Button-3>",self.pan_start);self.c.bind("<B3-Motion>",self.pan_drag)
        self.c.bind("<Configure>",lambda e:self.refresh())

        # Mouse wheel works on every adjustment slider.
        for _slider in (
            self.bs, self.fe, self.facecrop, self.masksz,
            self.xslider, self.yslider, self.sizeslider,
            self.rotslider, self.zoomslider
        ):
            self._enable_slider_mousewheel(_slider)



    def _enable_slider_mousewheel(self, scale):
        """Allow wheel adjustment when the mouse is over a slider."""
        def on_wheel(event):
            try:
                resolution = float(scale.cget("resolution"))
                if resolution <= 0:
                    resolution = 1.0

                if getattr(event, "num", None) == 4:
                    direction = 1
                elif getattr(event, "num", None) == 5:
                    direction = -1
                else:
                    delta = getattr(event, "delta", 0)
                    if not delta:
                        return "break"
                    direction = 1 if delta > 0 else -1

                value = float(scale.get()) + direction * resolution
                minimum = float(scale.cget("from"))
                maximum = float(scale.cget("to"))
                value = max(minimum, min(maximum, value))
                scale.set(value)

                # Trigger the slider's existing command exactly as a normal
                # slider change would.
                command = scale.cget("command")
                if command:
                    try:
                        scale.tk.call(command, str(value))
                    except Exception:
                        pass
            except Exception:
                pass
            return "break"

        scale.bind("<MouseWheel>", on_wheel, add="+")
        scale.bind("<Button-4>", on_wheel, add="+")
        scale.bind("<Button-5>", on_wheel, add="+")

    def _current_profile_data(self):
        if self.target is None:
            raise ValueError("Load the blurred target first.")
        h,w=self.target.shape[:2]
        # Store geometry normalized to the target dimensions. This is deliberately
        # independent of clothing and absolute pixel coordinates.
        ys,xs=np.where(self.base_mask_auto>8) if self.base_mask_auto is not None else (np.array([]),np.array([]))
        if len(xs):
            cx=float(xs.mean())/w; cy=float(ys.mean())/h
            fw=float(xs.max()-xs.min()+1)/w; fh=float(ys.max()-ys.min()+1)/h
        else:
            cx=.5; cy=.25; fw=.2; fh=.2
        return {
            "version": 1,
            "image_size": [int(w),int(h)],
            "transform": {
                "x_norm": float(self.xslider.get())/w,
                "y_norm": float(self.yslider.get())/h,
                "size": float(self.sizeslider.get())/100.0,
                "rotation": float(self.rotslider.get())
            },
            "face_geometry": {
                "center_x_norm": cx,
                "center_y_norm": cy,
                "width_norm": fw,
                "height_norm": fh
            },
            "mask": {
                "face_crop": float(self.facecrop.get()),
                "mask_size": float(self.masksz.get()),
                "feather": float(self.fe.get())
            }
        }

    def _refresh_profile_menu(self):
        menu=self.profile_combo["menu"]
        menu.delete(0,"end")
        names=list_face_profiles()
        if not names:
            menu.add_command(label="",command=lambda:self.profile_name.set(""))
        else:
            for n in names:
                menu.add_command(label=n,command=lambda v=n:self.profile_name.set(v))
            if self.profile_name.get() not in names:
                self.profile_name.set(names[0])

    def _profile_name_dialog(self, title, initial=""):
        from tkinter import simpledialog
        return simpledialog.askstring(title,"Profile name:",initialvalue=initial,parent=self.r)

    def save_profile(self):
        try:
            if self.base_aligned is None:
                return messagebox.showwarning("No adjustment","Run Auto Restore and make your desired manual/Auto Fine adjustments first.")
            name=self.profile_name.get().strip()
            if not name:
                name=self._profile_name_dialog("Save Model Profile")
            if not name:return
            save_face_profile(name,self._current_profile_data())
            self.profile_name.set(name)
            self.profile_status.set(f"Saved: {name}")
            self._refresh_profile_menu()
        except Exception as e:
            messagebox.showerror("Profile Error",str(e))

    def update_profile(self):
        try:
            name=self.profile_name.get().strip()
            if not name:
                return self.save_profile()
            save_face_profile(name,self._current_profile_data())
            self.profile_status.set(f"Updated: {name}")
            self._refresh_profile_menu()
        except Exception as e:
            messagebox.showerror("Profile Error",str(e))

    def lock_adjustment(self):
        try:
            if self.base_aligned is None:
                return messagebox.showwarning("No adjustment","Run Auto Restore first.")
            self.profile_data=self._current_profile_data()
            self.profile_locked=True
            self.profile_status.set("Adjustment locked for next target")
        except Exception as e:
            messagebox.showerror("Lock Error",str(e))

    def unlock_adjustment(self):
        self.profile_locked=False
        self.profile_data=None
        self.profile_status.set("Adjustment unlocked")

    def apply_profile(self):
        try:
            name=self.profile_name.get().strip()
            if not name:
                return messagebox.showwarning("No profile","Select or save a model profile first.")
            data=load_face_profile(name)
            if self.base_aligned is None:
                return messagebox.showwarning("No target","Load clear + blurred images and run AUTO RESTORE first.")
            pw,ph=data.get("image_size",[0,0])
            h,w=self.target.shape[:2]
            if pw and ph and (pw!=w or ph!=h):
                ans=messagebox.askyesno(
                    "Different image size",
                    f"Profile was saved for {pw}×{ph}, current image is {w}×{h}.\\n\\n"
                    "The profile is normalized, so it can still be applied. Continue?"
                )
                if not ans:return

            tr=data["transform"]
            self.xslider.set(int(round(float(tr["x_norm"])*w)))
            self.yslider.set(int(round(float(tr["y_norm"])*h)))
            self.sizeslider.set(int(round(float(tr["size"])*100)))
            self.rotslider.set(float(tr["rotation"]))
            mk=data.get("mask",{})
            if "face_crop" in mk:self.facecrop.set(float(mk["face_crop"]))
            if "mask_size" in mk:self.masksz.set(float(mk["mask_size"]))
            if "feather" in mk:self.fe.set(float(mk["feather"]))
            self.profile_data=data
            self.profile_locked=True
            self.profile_status.set(f"Applied: {name} — Auto Fine can refine locally")
            self.move_face(); self.rebuild()
            # Refine from the profile rather than resetting to automatic alignment.
            self.auto_fine(silent=True)
            self.status.config(text=f"Profile '{name}' applied and locally refined. Manual changes remain the new starting point.")
        except Exception as e:
            messagebox.showerror("Profile Error",str(e))

    def pick(self): return filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.webp")])
    def lc(self):
        p=self.pick()
        if p:self.clear=read_img(p);self.show(self.clear);self.status.config(text="Clear image loaded.")
    def lt(self):
        p=self.pick()
        if p:
            self.target=read_img(p);self.show(self.target)
            if self.profile_locked and self.profile_data is not None:
                self.profile_status.set("Locked model adjustment ready — run AUTO RESTORE, then Apply Profile")
            self.status.config(text="Blurred target loaded.")
    def run(self):
        if self.clear is None or self.target is None:
            return messagebox.showwarning("Missing","Load both images.")
        try:
            self.status.config(text="Robust detection: multi-pass face analysis...")
            self.r.update()

            built = build(self.clear,self.target)
            self.aligned,self.mask,self.valid,sf,tf,ok,self.align_meta = built
            self._sf=sf; self._tf=tf
            self._base_M=guarded_similarity_from_faces(
                sf,tf,self.align_meta.get("occlusion",0.0)
            )

            self.base_aligned=self.aligned.copy()
            self.base_mask_auto=self.mask.copy()
            self.base_valid=self.valid.copy()
            self.shift_x=0; self.shift_y=0
            self.xslider.set(0); self.yslider.set(0)
            self.sizeslider.set(100); self.rotslider.set(0)
            self.base=self.mask.copy()
            self.rebuild()

            safe = bool(self.align_meta.get("safe_auto_fine",False))
            if safe:
                self.auto_fine(silent=True)
                note = (
                    f"target detected; alignment {self.align_meta.get('alignment_method','unknown')}; "
                    f"confidence {self.align_meta.get('target_confidence',0):.2f}; "
                    f"occlusion {self.align_meta.get('occlusion',0):.2f}; auto-fine applied"
                )
            else:
                note = (
                    f"guarded {self.align_meta.get('alignment_method','alignment')}; target confidence "
                    f"{self.align_meta.get('target_confidence',0):.2f}; "
                    f"occlusion {self.align_meta.get('occlusion',0):.2f}; "
                    f"wide auto-fine disabled"
                )

            self.status.config(text="Done — " + note + (" | Locked profile available" if self.profile_locked else "") +
                                 " | Face crop 118%, mask 115% default")
            self.confidence_label.config(
                text=(
                    f"Alignment: {self.align_meta.get('alignment_method','unknown')} | "
                    f"Target confidence: {self.align_meta.get('target_confidence',0):.2f} | "
                    f"Dense target: {self.align_meta.get('dense_target_confidence',0):.2f} ({self.align_meta.get('target_recovery_method','none')}) | "
                    f"Occlusion: {self.align_meta.get('occlusion',0):.2f} | "
                    f"Target detector: {'YES' if ok else ('RECOVERED BLUR' if self.align_meta.get('target_recovered') else 'NO / GEOMETRY FALLBACK')} | "
                    f"Pose: {'FRONTAL DENSE' if self.align_meta.get('frontal_mode') else 'SIDE/OBLIQUE'}"
                )
            )
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def _edge_score(self,aligned,target,mask):
        # Score stable structure only: gradients/edges around face perimeter.
        # Blur first so pixelation in target center contributes less.
        ys,xs=np.where(mask>20)
        if len(xs)<50:return -1e9
        x0=max(0,int(xs.min()));x1=min(target.shape[1],int(xs.max()+1))
        y0=max(0,int(ys.min()));y1=min(target.shape[0],int(ys.max()+1))
        A=cv2.cvtColor(aligned[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        B=cv2.cvtColor(target[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        M=mask[y0:y1,x0:x1]
        if A.size<100:return -1e9
        A=cv2.GaussianBlur(A,(7,7),1.4);B=cv2.GaussianBlur(B,(7,7),1.4)
        ea=cv2.Canny(A,35,100).astype(np.float32)
        eb=cv2.Canny(B,35,100).astype(np.float32)

        # Ring mask emphasizes outer face/jaw/hair boundary, not blurred center.
        k=max(3,int(min(M.shape)*.055));k=min(k, min(M.shape)//2*2-1 if min(M.shape)>=4 else 3);k+=1-k%2
        ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        outer=cv2.dilate(M,ker,iterations=1)
        inner=cv2.erode(M,ker,iterations=2)
        ring=cv2.subtract(outer,inner)>10
        if ring.sum()<30:ring=M>20

        # Distance-transform edge matching tolerates a few pixels of blur.
        inv=(255-eb.astype(np.uint8))
        dist=cv2.distanceTransform(inv,cv2.DIST_L2,3)
        pts=(ea>0)&ring
        if pts.sum()<20:return -1e9
        edge_score=-float(np.mean(np.minimum(dist[pts],15.0)))

        # Add low-frequency grayscale agreement in the ring.
        af=A.astype(np.float32);bf=B.astype(np.float32)
        av=af[ring];bv=bf[ring]
        if len(av)>20:
            av-=av.mean();bv-=bv.mean()
            corr=float((av*bv).sum()/(np.sqrt((av*av).sum()*(bv*bv).sum())+1e-6))
        else:corr=0
        score=edge_score + 2.0*corr
        # If a model profile is locked/applied, mildly prefer staying near its
        # learned transform. Image evidence still dominates; this is only a
        # stabilizing prior against unrelated background edges.
        if self.profile_locked and self.profile_data is not None:
            tr=self.profile_data.get("transform",{})
            h,w=self.target.shape[:2]
            pdx=float(tr.get("x_norm",0))*w
            pdy=float(tr.get("y_norm",0))*h
            psc=float(tr.get("size",1.0))
            pang=float(tr.get("rotation",0))
            # Current candidate parameters are available through caller, so the
            # prior is applied there rather than here.
        return score

    def _make_fast_context(self):
        """Prepare one small face ROI for all optimization trials."""
        h,w=self.target.shape[:2]
        ys,xs=np.where(self.base_mask_auto>8)
        if len(xs)<20:return None
        cx=float(xs.mean());cy=float(ys.mean())
        fw=max(40,float(xs.max()-xs.min()+1));fh=max(40,float(ys.max()-ys.min()+1))
        pad=max(fw,fh)*0.42
        x0=max(0,int(xs.min()-pad));x1=min(w,int(xs.max()+pad+1))
        y0=max(0,int(ys.min()-pad));y1=min(h,int(ys.max()+pad+1))

        A=self.base_aligned[y0:y1,x0:x1].copy()
        B=self.target[y0:y1,x0:x1].copy()
        V=self.base_valid[y0:y1,x0:x1].copy()
        M=self.base_mask_auto[y0:y1,x0:x1].copy()

        # Search at <=320 px on longest side. Final transform is still full quality.
        rh,rw=B.shape[:2]
        ds=min(1.0,320.0/max(rw,rh))
        if ds<1.0:
            sz=(max(1,int(rw*ds)),max(1,int(rh*ds)))
            A=cv2.resize(A,sz,interpolation=cv2.INTER_AREA)
            B=cv2.resize(B,sz,interpolation=cv2.INTER_AREA)
            V=cv2.resize(V,sz,interpolation=cv2.INTER_NEAREST)
            M=cv2.resize(M,sz,interpolation=cv2.INTER_NEAREST)

        yy,xx=np.where(M>8)
        ccx=float(xx.mean());ccy=float(yy.mean())
        return A,B,V,M,ds,ccx,ccy

    def _candidate_fast(self,ctx,dx,dy,scale,angle):
        A,B,V,M,ds,cx,cy=ctx
        hh,ww=B.shape[:2]
        T=cv2.getRotationMatrix2D((cx,cy),angle,scale).astype(np.float32)
        T[0,2]+=dx*ds;T[1,2]+=dy*ds
        al=cv2.warpAffine(A,T,(ww,hh),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        va=cv2.warpAffine(V,T,(ww,hh),flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        ma=cv2.warpAffine(M,T,(ww,hh),flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        ma=cv2.min(ma,va)
        return self._edge_score(al,B,ma)

    def auto_fine(self,silent=False):
        """
        Iterative/manual-aware Auto Fine.

        IMPORTANT: the current sliders are the starting point. Auto Fine never
        resets them to zero. Every press searches a local neighborhood around
        the user's CURRENT X/Y/size/rotation, applies the best improvement,
        and leaves the resulting values as the new starting point.

        This means the workflow is intentionally iterative:

            Auto Restore -> manual adjustment -> Auto Fine
                         -> manual adjustment -> Auto Fine -> ...

        Manual changes therefore act as a strong spatial prior rather than
        being overwritten by the original automatic alignment.
        """
        if self.base_aligned is None or self.target is None:return

        if not silent:
            self.status.config(text="Auto Fine: refining from current manual alignment...")
            self.r.update()

        ctx=self._make_fast_context()
        if ctx is None:return

        # CURRENT transform is the center of this Auto Fine pass.
        cur_dx=float(self.xslider.get())
        cur_dy=float(self.yslider.get())
        cur_scale=float(self.sizeslider.get())/100.0
        cur_angle=float(self.rotslider.get())

        # Auto Restore may be working with a low-confidence target. In that
        # case use a deliberately local search. For reliable targets, allow a
        # wider first pass, but still remain centered on the user's current
        # transform. This is the key difference from v7.x.
        guarded = not bool(self.align_meta.get("safe_auto_fine",False))
        frontal = bool(self.align_meta.get("frontal_mode",False))

        # Search around the CURRENT transform, not around zero.  Frontal dense
        # warping has already corrected local face geometry, so Auto Fine is
        # intentionally much narrower there; this prevents a global affine
        # correction from undoing the dense landmark fit. Side/oblique keeps
        # the v10 search ranges unchanged.
        if frontal:
            scale_offsets=( -.02,-.01,0,.01,.02 )
            angle_offsets=( -1.0,-.5,0,.5,1.0 )
            xy_offsets=(-4,-2,0,2,4)
        elif guarded:
            scale_offsets=( -.03,-.015,0,.015,.03 )
            angle_offsets=( -2.0,-1.0,0,1.0,2.0 )
            xy_offsets=(-8,-4,0,4,8)
        else:
            scale_offsets=( -.06,-.03,0,.03,.06 )
            angle_offsets=( -4,-2,0,2,4 )
            xy_offsets=(-12,-6,0,6,12)

        # Baseline must always be a candidate. This prevents Auto Fine from
        # making the image worse merely because the optimizer is uncertain.
        base_score=self._candidate_fast(
            ctx,cur_dx,cur_dy,cur_scale,cur_angle
        )
        best=(base_score,cur_dx,cur_dy,cur_scale,cur_angle)

        # Coarse local search.
        for ds in scale_offsets:
            sc=max(.70,min(1.30,cur_scale+ds))
            for da in angle_offsets:
                ang=max(-20,min(20,cur_angle+da))
                for dy in xy_offsets:
                    for dx in xy_offsets:
                        q=self._candidate_fast(ctx,cur_dx+dx,cur_dy+dy,sc,ang)
                        if q>best[0]:
                            best=(q,cur_dx+dx,cur_dy+dy,sc,ang)

        # Medium refinement around the best candidate.
        _,bx,by,bs,ba=best
        medium=best
        med_s=.006 if frontal else (.015 if guarded else .02)
        med_a=.35 if frontal else (.75 if guarded else 1.0)
        med_d=1 if frontal else (2 if guarded else 3)
        for sc in (bs-med_s,bs,bs+med_s):
            for ang in (ba-med_a,ba,ba+med_a):
                for dy in (by-med_d,by,by+med_d):
                    for dx in (bx-med_d,bx,bx+med_d):
                        q=self._candidate_fast(ctx,dx,dy,sc,ang)
                        if q>medium[0]:
                            medium=(q,dx,dy,sc,ang)

        # Fine refinement. Again, this is around the CURRENT winner, not zero.
        _,bx,by,bs,ba=medium
        fine=medium
        fine_s=.003 if frontal else .005
        fine_a=.15 if frontal else .25
        fine_d=1
        for sc in (bs-fine_s,bs,bs+fine_s):
            for ang in (ba-fine_a,ba,ba+fine_a):
                for dy in (by-fine_d,by,by+fine_d):
                    for dx in (bx-fine_d,bx,bx+fine_d):
                        q=self._candidate_fast(ctx,dx,dy,sc,ang)
                        if q>fine[0]:
                            fine=(q,dx,dy,sc,ang)

        # Only accept a meaningful improvement. Otherwise preserve exactly
        # what the user had before pressing Auto Fine.
        improvement=fine[0]-base_score
        min_gain=.012 if frontal else (.015 if guarded else .008)
        if improvement < min_gain:
            dx,dy,sc,ang=cur_dx,cur_dy,cur_scale,cur_angle
            changed=False
        else:
            _,dx,dy,sc,ang=fine
            changed=True

        self.xslider.set(int(round(max(-120,min(120,dx)))))
        self.yslider.set(int(round(max(-120,min(120,dy)))))
        self.sizeslider.set(int(round(max(70,min(130,sc*100)))))
        self.rotslider.set(round(max(-20,min(20,ang))*2)/2)

        # Apply once at full resolution.
        self.move_face()
        self.rebuild()

        if not silent:
            if changed:
                self.status.config(
                    text=(
                        f"Auto Fine refined current alignment: "
                        f"X {int(round(dx))}, Y {int(round(dy))}, "
                        f"Size {sc*100:.0f}%, Rotation {ang:.1f}°. "
                        f"Press again to refine further."
                    )
                )
            else:
                self.status.config(
                    text=(
                        "Auto Fine kept your current alignment "
                        "(no meaningful improvement found)."
                    )
                )

    def move_face(self,_=None):
        if self.base_aligned is None or self.target is None:return
        dx=int(self.xslider.get()); dy=int(self.yslider.get())
        scale=float(self.sizeslider.get())/100.0
        angle=float(self.rotslider.get())
        self.shift_x=dx; self.shift_y=dy
        h,w=self.target.shape[:2]

        # Transform around the actual automatic face-mask center, not image center.
        ys,xs=np.where(self.base_mask_auto>8)
        if len(xs):
            cx=float(xs.mean()); cy=float(ys.mean())
        else:
            cx=w*.5; cy=h*.22

        T=cv2.getRotationMatrix2D((cx,cy),angle,scale).astype(np.float32)
        T[0,2]+=dx; T[1,2]+=dy

        self.aligned=cv2.warpAffine(self.base_aligned,T,(w,h),flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        self.valid=cv2.warpAffine(self.base_valid,T,(w,h),flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        self.mask=cv2.warpAffine(self.base_mask_auto,T,(w,h),flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        self.valid=cv2.warpAffine(self.base_valid,T,(w,h),flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        self.mask=cv2.min(self.mask,self.valid)

        self.base=self.mask.copy()
        self.quick()

    def _rebuild_face_masks(self):
        if self.target is None or not hasattr(self,"_sf") or not hasattr(self,"_tf"):
            return
        crop=float(self.facecrop.get())/100.0
        expansion=float(self.masksz.get())

        # Expand the source and target crops separately, then intersect them.
        # This is critical: expanding the already-intersected mask could reveal
        # pixels outside the source face.
        sm=source_face_mask(self.clear.shape,self._sf,crop)
        sm=expand_mask(sm,expansion)

        self.base_valid=cv2.warpAffine(
            sm,self._base_M,self.target.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        tm=target_face_mask(self.target.shape,self._tf,crop)
        tm=expand_mask(tm,expansion)
        self.base_mask_auto=cv2.min(self.base_valid,tm)

    def mask_resize(self,_=None):
        if self.base_aligned is None or self.target is None:
            return
        self._rebuild_face_masks()
        self.move_face()

    def face_crop_resize(self,_=None):
        if self.base_aligned is None or self.target is None:
            return
        self._rebuild_face_masks()
        self.move_face()

    def reset_position(self):
        if self.base_aligned is None:return
        if self.profile_locked:
            if not messagebox.askyesno("Locked adjustment",
                                       "A model adjustment is locked. Reset it anyway?"):
                return
        self.xslider.set(0);self.yslider.set(0)
        self.sizeslider.set(100);self.rotslider.set(0)
        if hasattr(self,"facecrop"): self.facecrop.set(100)
        if hasattr(self,"masksz"): self.masksz.set(115)
        self._rebuild_face_masks()
        self.move_face();self.rebuild()

    def rebuild(self):
        if self.mask is not None:self.result=blend(self.aligned,self.target,self.mask,self.fe.get());self.refresh()
    def quick(self):
        if self.mask is None:return
        # Fast unfeathered preview during sliders/brush strokes.
        a=self.mask.astype(np.float32)[...,None]/255.0
        self.result=np.clip(self.aligned.astype(np.float32)*a+
                            self.target.astype(np.float32)*(1-a),0,255).astype(np.uint8)
        self.refresh()

    def reset(self):
        if self.base is not None:self.mask=self.base.copy();self.rebuild()
    def undo(self):
        if self.undo_mask is not None:self.mask=self.undo_mask.copy();self.undo_mask=None;self.rebuild()
    def show(self,im):
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);p=Image.fromarray(rgb)
        cw=max(100,self.c.winfo_width());ch=max(100,self.c.winfo_height());iw,ih=p.size
        fit=min(cw/iw,ch/ih,1.0)
        s=fit*self.zoom
        nw=max(1,int(iw*s));nh=max(1,int(ih*s))
        p=p.resize((nw,nh),Image.Resampling.LANCZOS)
        self.scale=s
        self.off=((cw-nw)//2+int(self.pan_x),(ch-nh)//2+int(self.pan_y))
        self.tk=ImageTk.PhotoImage(p)
        self.c.delete("image")
        self.c.create_image(*self.off,anchor="nw",image=self.tk,tags="image")
        self.c.tag_lower("image")

    def refresh(self):
        if self.result is not None:self.show(self.result)
    def xy(self,e):
        if self.target is None:return None
        x=int((e.x-self.off[0])/max(.001,self.scale));y=int((e.y-self.off[1])/max(.001,self.scale))
        h,w=self.target.shape[:2];return (x,y) if 0<=x<w and 0<=y<h else None
    def stroke(self,a,b):
        if a is None or b is None:return
        # Brush size is in SCREEN pixels, so it stays visually consistent when zooming.
        rad=max(1,int((self.bs.get()/2)/max(.01,self.scale)))
        v=0 if self.mode.get()=="Erase" else 255
        cv2.line(self.mask,a,b,v,rad*2,cv2.LINE_AA)
        cv2.circle(self.mask,b,rad,v,-1,cv2.LINE_AA)
        if v==255:self.mask=cv2.min(self.mask,self.valid)

    def down(self,e):
        if self.mask is None:return
        self.paint=True;self.undo_mask=self.mask.copy();self.last=self.xy(e)
        self.stroke(self.last,self.last);self.quick()
    def drag(self,e):
        if self.paint:
            p=self.xy(e)
            if p and self.last:
                self.stroke(self.last,p);self.last=p
                self.quick()
    def up(self,e):
        if not self.paint:return
        self.paint=False;self.last=None;self.rebuild()

    def motion(self,e):
        # Always-visible eraser/restore circle.
        self.c.delete("brushcursor")
        if self.mask is None:return
        r=max(3,self.bs.get()/2)
        outline="white" if self.mode.get()=="Erase" else "#00ff88"
        self.c.create_oval(e.x-r,e.y-r,e.x+r,e.y+r,outline=outline,width=2,tags="brushcursor")
        self.c.tag_raise("brushcursor")

    def set_zoom(self,_=None):
        self.zoom=max(.5,float(self.zoomslider.get())/100.0)
        self.refresh()

    def fit_view(self):
        self.zoom=1.0;self.pan_x=0;self.pan_y=0
        self.zoomslider.set(100);self.refresh()

    def wheel(self,e):
        # Wheel zooms around the pixel under the mouse.
        # Ctrl+wheel = face size, Shift+wheel = face rotation.
        step=1 if e.delta>0 else -1
        state=e.state
        if state & 0x0004:
            self.sizeslider.set(max(70,min(130,self.sizeslider.get()+step)))
            self.move_face()
        elif state & 0x0001:
            self.rotslider.set(max(-20,min(20,self.rotslider.get()+step*.5)))
            self.move_face()
        else:
            if self.target is None:return "break"
            # Image coordinate currently under cursor.
            old_scale=max(.001,self.scale)
            ix=(e.x-self.off[0])/old_scale
            iy=(e.y-self.off[1])/old_scale

            oldz=float(self.zoomslider.get())
            newz=max(50,min(400,oldz+step*10))
            self.zoomslider.set(newz)
            self.zoom=max(.5,newz/100.0)

            # Compute new fitted scale, then choose pan so same image point stays under mouse.
            cw=max(100,self.c.winfo_width());ch=max(100,self.c.winfo_height())
            h,w=self.target.shape[:2]
            fit=min(cw/w,ch/h,1.0)
            ns=fit*self.zoom
            nw=w*ns;nh=h*ns
            base_x=(cw-nw)/2;base_y=(ch-nh)/2
            self.pan_x=e.x-base_x-ix*ns
            self.pan_y=e.y-base_y-iy*ns
            self.refresh()
        return "break"

    def pan_start(self,e):
        self._pan_anchor=(e.x,e.y,self.pan_x,self.pan_y)

    def pan_drag(self,e):
        if not hasattr(self,"_pan_anchor"):return
        x0,y0,px,py=self._pan_anchor
        self.pan_x=px+(e.x-x0);self.pan_y=py+(e.y-y0)
        self.refresh()

    def save(self):
        if self.result is None:return
        p=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if p:save_img(p,self.result)

# v10 note: install MediaPipe for dense alignment:
#     pip install mediapipe
# If MediaPipe is absent or fails on an image, v9.1 YuNet alignment remains the fallback.

if __name__=="__main__":
    root=tk.Tk();App(root);root.mainloop()
