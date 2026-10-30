"""Reproducible E0-E5 experiments for egg segmentation.

SAM masks are treated only as evaluation pseudo-labels. Candidate selection
uses image content and fixed, documented feature weights, never reference masks.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
np.random.seed(SEED)


@dataclass(frozen=True)
class Config:
    mser_delta: int = 5
    mser_min_area: int = 500
    mser_max_area_ratio: float = 0.50
    canny_low: int = 45
    canny_high: int = 130
    close_size: int = 9
    min_area_ratio: float = 0.006
    max_area_ratio: float = 0.50
    min_axis_ratio: float = 0.28
    max_axis_ratio: float = 3.6
    score_threshold: float = 0.34
    max_candidates: int = 5
    grabcut_iterations: int = 5
    grabcut_pad: float = 0.12


DEFAULT = Config()
FEATURE_WEIGHTS = {
    "geometry": 0.20,
    "edge_support": 0.20,
    "lab_contrast": 0.20,
    "interior_smoothness": 0.15,
    "solidity": 0.15,
    "border_clearance": 0.10,
}


def read_manifest(file: Path) -> list[Path]:
    if not file.is_file():
        raise FileNotFoundError(f"Manifest missing: {file}")
    paths = []
    for raw in file.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in {file} is missing: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate image paths in {file}")
    return paths


def references() -> dict[str, Path]:
    folder = ROOT / "outputs/automatic_masks"
    if not folder.is_dir():
        raise FileNotFoundError(f"SAM reference directory missing: {folder}")
    index: dict[str, Path] = {}
    for p in sorted(folder.glob("*_mask.png")):
        if "_object_" in p.name:
            continue
        stem = p.name[:-len("_mask.png")].rsplit("__", 1)[-1]
        if stem in index:
            raise ValueError(f"Ambiguous SAM references for image stem {stem}")
        index[stem] = p
    return index


def binary_mask(mask: np.ndarray, label: str = "mask") -> np.ndarray:
    if mask is None or mask.ndim != 2:
        raise ValueError(f"{label} must be a 2-D mask")
    return (mask > 0).astype(np.uint8) * 255


def mask_metrics(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    if pred.shape != ref.shape:
        raise ValueError(f"Mask dimensions differ: {pred.shape} vs {ref.shape}")
    p, r = pred > 0, ref > 0
    intersection = int(np.logical_and(p, r).sum())
    union = int(np.logical_or(p, r).sum())
    denom = int(p.sum() + r.sum())
    return (intersection / union if union else 1.0,
            2 * intersection / denom if denom else 1.0)


def load_reference(path: Path, shape: tuple[int, int]) -> np.ndarray:
    ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise ValueError(f"Could not read reference mask: {path}")
    if ref.shape != shape:
        raise ValueError(f"Reference/image dimensions differ for {path}: {ref.shape} vs {shape}")
    return binary_mask(ref, str(path))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def save_image(path: Path, image: np.ndarray) -> None:
    """Fail the stage if OpenCV cannot persist an expected artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image artifact: {path}")


def summarize(rows: list[dict], iou_key="iou_vs_sam", dice_key="dice_vs_sam") -> dict:
    if not rows:
        return {"n": 0}
    ious = np.asarray([r[iou_key] for r in rows], dtype=float)
    dices = np.asarray([r[dice_key] for r in rows], dtype=float)
    fg = np.asarray([r["foreground_ratio"] for r in rows], dtype=float)
    return {
        "n": len(rows), "mean_iou_vs_sam": float(ious.mean()),
        "median_iou_vs_sam": float(np.median(ious)),
        "mean_dice_vs_sam": float(dices.mean()),
        "median_dice_vs_sam": float(np.median(dices)),
        "zero_iou_count": int((ious == 0).sum()),
        "mean_foreground_ratio": float(fg.mean()),
    }


def overlay(image: np.ndarray, mask: np.ndarray, color=(0, 220, 0)) -> np.ndarray:
    out = image.copy()
    sel = mask > 0
    tint = np.zeros_like(image)
    tint[:] = color
    blend = cv2.addWeighted(image, 0.55, tint, 0.45, 0)
    out[sel] = blend[sel]
    contours, _ = cv2.findContours(binary_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


def load_trial5_module():
    """Import the untouched baseline module and call only its pure functions."""
    source = ROOT / "lab_illumination_segmentation.py"
    spec = importlib.util.spec_from_file_location("trial5_original", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original Trial 5 source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_e0() -> list[dict]:
    out = ROOT / "outputs/e0_baseline_reproduction"
    masks_dir = out / "masks"
    corrected_dir = out / "corrected"
    overlays_dir = out / "overlays"
    for d in (masks_dir, corrected_dir, overlays_dir): d.mkdir(parents=True, exist_ok=True)
    t5 = load_trial5_module()
    # Do not reset OpenCV's RNG before invoking Trial 5: its saved GrabCut
    # outputs were produced by the original process RNG state, and changing it
    # changes fallback masks on some OpenCV builds. This is part of E0 fidelity.
    refs = references()
    existing_csv = ROOT / "outputs/trial5_grabcut_fallback/metrics.csv"
    with existing_csv.open(newline="", encoding="utf-8") as f:
        prior = {r["image"]: r for r in csv.DictReader(f)}
    rows = []
    paths = read_manifest(ROOT / "splits/dev.txt")
    for image_path in paths:
        rel = image_path.relative_to(ROOT).as_posix()
        image = cv2.imread(str(image_path))
        if image is None: raise ValueError(f"Unreadable image: {image_path}")
        ref_path = refs.get(image_path.stem)
        if ref_path is None: raise FileNotFoundError(f"SAM mask missing for {rel}")
        pred, corrected, fallback = t5.segment(image)
        pred = binary_mask(pred, "Trial 5 prediction")
        ref = load_reference(ref_path, image.shape[:2])
        iou, dice = mask_metrics(pred, ref)
        stem = image_path.stem
        mp, cp, op = masks_dir/f"{stem}_mask.png", corrected_dir/f"{stem}_corrected.png", overlays_dir/f"{stem}_overlay.jpg"
        save_image(mp, pred); save_image(cp, corrected); save_image(op, overlay(image, pred))
        old = prior.get(rel)
        if old is None: raise ValueError(f"Trial 5 CSV missing image identity: {rel}")
        old_mask_path = ROOT / old["prediction_mask"]
        old_mask = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        if old_mask is None: raise FileNotFoundError(f"Existing Trial 5 mask missing: {old_mask_path}")
        if old_mask.shape != pred.shape: raise ValueError(f"Existing Trial 5 mask dimensions differ: {rel}")
        rows.append({"image": rel, "iou_vs_sam": iou, "dice_vs_sam": dice,
                     "foreground_ratio": float((pred > 0).mean()), "used_fallback": bool(fallback),
                     "existing_iou": float(old["iou_vs_sam"]), "existing_dice": float(old["dice_vs_sam"]),
                     "existing_foreground_ratio": float(old["predicted_foreground_ratio"]),
                     "existing_fallback": old["used_grabcut_fallback"].lower() == "true",
                     "mask_pixel_mismatches": int(np.count_nonzero((old_mask > 0) != (pred > 0))),
                     "iou_abs_delta": abs(iou-float(old["iou_vs_sam"])),
                     "dice_abs_delta": abs(dice-float(old["dice_vs_sam"])),
                     "foreground_ratio_abs_delta": abs(float((pred>0).mean())-float(old["predicted_foreground_ratio"])),
                     "fallback_matches": bool(fallback) == (old["used_grabcut_fallback"].lower()=="true"),
                     "prediction_mask": mp.relative_to(ROOT).as_posix(), "overlay": op.relative_to(ROOT).as_posix()})
    if len(rows) != len(prior):
        raise ValueError(f"Trial 5 identities differ: current {len(rows)} vs saved {len(prior)}")
    write_csv(out/"metrics.csv", rows)
    (out/"summary.json").write_text(json.dumps({"metrics": summarize(rows),
       "exact_mask_matches": sum(r["mask_pixel_mismatches"] == 0 for r in rows),
       "fallback_matches": sum(r["fallback_matches"] for r in rows),
       "max_iou_abs_delta": max(r["iou_abs_delta"] for r in rows)}, indent=2)+"\n")
    return rows


@dataclass
class Candidate:
    source: str
    ellipse: tuple
    contour: np.ndarray
    mask: np.ndarray
    features: dict
    score: float = 0.0


def ellipse_candidate(contour: np.ndarray, shape: tuple[int,int], source: str, config: Config) -> Candidate | None:
    h,w = shape
    if len(contour) < 5: return None
    area = abs(cv2.contourArea(contour))
    if area < config.min_area_ratio*h*w or area > config.max_area_ratio*h*w: return None
    try: e = cv2.fitEllipse(contour)
    except cv2.error: return None
    (cx,cy),(a,b),angle = e
    if a < 2 or b < 2: return None
    axis_ratio = min(a,b)/max(a,b)
    if not config.min_axis_ratio <= axis_ratio <= config.max_axis_ratio: return None
    ellipse_area = math.pi*a*b/4
    if not config.min_area_ratio*h*w <= ellipse_area <= config.max_area_ratio*h*w: return None
    mask = np.zeros((h,w),np.uint8)
    cv2.ellipse(mask,e,255,-1)
    return Candidate(source,e,contour,mask,{"area_ratio":ellipse_area/(h*w),"axis_ratio":axis_ratio})


def proposal_candidates(image: np.ndarray, config: Config=DEFAULT) -> tuple[list[Candidate], np.ndarray]:
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),config.canny_low,config.canny_high)
    found=[]
    mser=cv2.MSER_create(config.mser_delta,config.mser_min_area,max(1,int(config.mser_max_area_ratio*h*w)))
    regions,_=mser.detectRegions(gray)
    for region in regions:
        if len(region)<5: continue
        cnt=cv2.convexHull(region.reshape(-1,1,2).astype(np.int32))
        c=ellipse_candidate(cnt,(h,w),"mser",config)
        if c: found.append(c)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(config.close_size,config.close_size))
    closed=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations=2)
    contours,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        c=ellipse_candidate(cnt,(h,w),"canny",config)
        if c: found.append(c)
    # Keep source-specific proposals separate. Ranking performs NMS after source
    # filtering, so MSER-only and Canny-only remain independently comparable.
    mser_found=sorted((c for c in found if c.source=="mser"),key=lambda c:cv2.countNonZero(c.mask),reverse=True)
    canny_found=sorted((c for c in found if c.source=="canny"),key=lambda c:cv2.countNonZero(c.mask),reverse=True)
    cap=config.max_candidates*6
    kept=mser_found[:cap]+canny_found[:cap]
    return kept,edges


def candidate_features(candidate: Candidate, image: np.ndarray, edges: np.ndarray) -> dict:
    h,w=image.shape[:2]
    mask=candidate.mask>0
    # Edge support on the estimated ellipse perimeter, not its filled interior.
    er=cv2.erode(candidate.mask,np.ones((5,5),np.uint8))>0
    boundary=mask & ~er
    edge_support=float((edges[boundary]>0).mean()) if boundary.any() else 0.0
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    outer=cv2.dilate(candidate.mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))>0
    ring=outer & ~mask
    inside=lab[mask]
    outside=lab[ring]
    contrast=float(np.linalg.norm(inside.mean(axis=0)-outside.mean(axis=0))) if len(inside) and len(outside) else 0.0
    smooth=float(np.mean(np.std(inside,axis=0))) if len(inside) else 100.0
    # Map scale-dependent scores to [0,1] with fixed, documented saturations.
    geometry=float(np.clip(1-abs(math.log(max(candidate.features["axis_ratio"],1e-6)/0.55))/2.2,0,1))
    solidity=float(np.clip(cv2.contourArea(candidate.contour)/max(cv2.contourArea(cv2.convexHull(candidate.contour)),1),0,1))
    x,y=(candidate.ellipse[0])
    rx,ry=candidate.ellipse[1][0]/2,candidate.ellipse[1][1]/2
    border=float(np.clip(min(x-rx,y-ry,w-(x+rx),h-(y+ry))/max(1,min(rx,ry)),0,1))
    return {"geometry":geometry,"edge_support":float(np.clip(edge_support*2,0,1)),
            "lab_contrast":float(np.clip(contrast/45,0,1)),
            "interior_smoothness":float(np.clip(1-smooth/42,0,1)),
            "solidity":solidity,"border_clearance":border}


def score_candidate(candidate: Candidate, image: np.ndarray, edges: np.ndarray,
                    omitted: Iterable[str]=()) -> float:
    candidate.features.update(candidate_features(candidate,image,edges))
    weights={k:v for k,v in FEATURE_WEIGHTS.items() if k not in set(omitted)}
    denom=sum(weights.values())
    candidate.score=sum(weights[k]*candidate.features[k] for k in weights)/denom
    return candidate.score


def rank_candidates(candidates: list[Candidate], image: np.ndarray, edges: np.ndarray,
                    config: Config=DEFAULT, omitted: Iterable[str]=()) -> list[Candidate]:
    for c in candidates: score_candidate(c,image,edges,omitted)
    selected=[]
    for c in sorted(candidates,key=lambda c:c.score,reverse=True):
        if c.score < config.score_threshold: continue
        # High overlap suppression; spatially separated eggs remain separate.
        if any(np.logical_and(c.mask>0,s.mask>0).sum()/max(1,np.logical_or(c.mask>0,s.mask>0).sum())>.65 for s in selected): continue
        selected.append(c)
        if len(selected)>=config.max_candidates: break
    return selected


def local_grabcut(image: np.ndarray, candidate: Candidate, config: Config=DEFAULT) -> tuple[np.ndarray,str]:
    h,w=image.shape[:2]
    x,y=candidate.ellipse[0]; rx,ry=candidate.ellipse[1][0]/2,candidate.ellipse[1][1]/2
    pad=config.grabcut_pad
    x0=max(0,int(x-rx*(1+pad))); x1=min(w,int(x+rx*(1+pad))+1)
    y0=max(0,int(y-ry*(1+pad))); y1=min(h,int(y+ry*(1+pad))+1)
    if x1-x0<8 or y1-y0<8: return candidate.mask.copy(),"rejected_crop_too_small"
    crop=image[y0:y1,x0:x1].copy(); seed=candidate.mask[y0:y1,x0:x1]
    if np.count_nonzero(seed)<25: return candidate.mask.copy(),"rejected_empty_seed"
    gc=np.full(seed.shape,cv2.GC_PR_BGD,np.uint8)
    gc[seed>0]=cv2.GC_PR_FGD
    sure=cv2.erode(seed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    gc[sure>0]=cv2.GC_FGD
    if not np.any(gc==cv2.GC_FGD): return candidate.mask.copy(),"rejected_no_foreground_seed"
    # One-pixel crop border is certain background.
    gc[[0,-1],:]=cv2.GC_BGD; gc[:,[0,-1]]=cv2.GC_BGD
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(crop,gc,None,bg,fg,config.grabcut_iterations,cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        return candidate.mask.copy(),f"failed_grabcut:{str(exc)[:120]}"
    local=np.isin(gc,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    # Keep the connected component that overlaps the ellipse center.
    n,labels,stats,_=cv2.connectedComponentsWithStats((local>0).astype(np.uint8),8)
    center=(int(round(y-y0)),int(round(x-x0)))
    label=int(labels[min(max(center[0],0),labels.shape[0]-1),min(max(center[1],0),labels.shape[1]-1)])
    if label==0 and n>1: label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    if label==0: return candidate.mask.copy(),"rejected_no_foreground"
    local=np.where(labels==label,255,0).astype(np.uint8)
    area=int(cv2.countNonZero(local)); ellipse_area=int(cv2.countNonZero(seed))
    if area<max(20,int(.20*ellipse_area)) or area>int(1.35*ellipse_area):
        return candidate.mask.copy(),"rejected_implausible_area"
    refined=np.zeros((h,w),np.uint8); refined[y0:y1,x0:x1]=local
    return refined,"accepted"


def process_image(image_path: Path, mode: str, out_dir: Path, config: Config=DEFAULT,
                  omitted: Iterable[str]=()) -> tuple[dict,list[dict]]:
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError(f"Unreadable image: {image_path}")
    refs=references(); ref_path=refs.get(image_path.stem)
    if not ref_path: raise FileNotFoundError(f"SAM reference missing for {image_path}")
    ref=load_reference(ref_path,image.shape[:2])
    candidates,edges=proposal_candidates(image,config)
    if mode=="mser": candidates=[c for c in candidates if "mser" in c.source.split("+") and "canny" not in c.source.split("+")]
    elif mode=="canny": candidates=[c for c in candidates if "canny" in c.source.split("+") and "mser" not in c.source.split("+")]
    # Fused candidates belong to combined mode only, ensuring independent variants.
    selected=rank_candidates(candidates,image,edges,config,omitted)
    proposal_union=np.zeros(image.shape[:2],np.uint8)
    for c in selected: proposal_union=cv2.bitwise_or(proposal_union,c.mask)
    raw=proposal_union.copy()
    refined=np.zeros_like(raw); statuses=[]
    if mode=="combined_grabcut":
        for c in selected:
            m,status=local_grabcut(image,c,config); refined=cv2.bitwise_or(refined,m); statuses.append(status)
        pred=refined
    else: pred=raw
    iou,dice=mask_metrics(pred,ref)
    stem=image_path.stem; masks=out_dir/"masks"; overlays=out_dir/"overlays"; proposals=out_dir/"proposals"
    for d in (masks,overlays,proposals): d.mkdir(parents=True,exist_ok=True)
    save_image(masks/f"{stem}_mask.png",pred)
    save_image(masks/f"{stem}_raw_ellipse.png",raw)
    save_image(masks/f"{stem}_refined.png",refined if mode=="combined_grabcut" else raw)
    save_image(overlays/f"{stem}_overlay.jpg",overlay(image,pred))
    vis=image.copy()
    for idx,c in enumerate(selected,1):
        cv2.ellipse(vis,c.ellipse,(0,220,255),2)
        cx,cy=map(int,c.ellipse[0]); cv2.putText(vis,f"{idx}:{c.score:.2f}",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    save_image(proposals/f"{stem}_proposals.jpg",vis)
    candidate_rows=[]
    selected_ids={id(c) for c in selected}
    for i,c in enumerate(candidates):
        candidate_rows.append({"image":image_path.relative_to(ROOT).as_posix(),"mode":mode,"candidate_index":i,
           "source":c.source,"selected":id(c) in selected_ids,"score":c.score,"ellipse_cx":c.ellipse[0][0],"ellipse_cy":c.ellipse[0][1],
           "ellipse_axis_a":c.ellipse[1][0],"ellipse_axis_b":c.ellipse[1][1],"ellipse_angle":c.ellipse[2],**c.features})
    row={"image":image_path.relative_to(ROOT).as_posix(),"mode":mode,"iou_vs_sam":iou,"dice_vs_sam":dice,
         "foreground_ratio":float((pred>0).mean()),"candidate_count":len(candidates),"selected_count":len(selected),
         "grabcut_status":"|".join(statuses),"prediction_mask":(masks/f"{stem}_mask.png").relative_to(ROOT).as_posix(),
         "overlay":(overlays/f"{stem}_overlay.jpg").relative_to(ROOT).as_posix(),
         "proposal_visualization":(proposals/f"{stem}_proposals.jpg").relative_to(ROOT).as_posix(),
         "proposal_foreground_overlap_diagnostic":float(np.logical_and(proposal_union>0,ref>0).sum()/max(1,(ref>0).sum()))}
    return row,candidate_rows


def run_variant(name: str, mode: str, out_dir: Path, omitted=()) -> list[dict]:
    cv2.setRNGSeed(SEED)
    rows=[]; candidates=[]
    for p in read_manifest(ROOT/"splits/dev.txt"):
        row,cr=process_image(p,mode,out_dir,omitted=omitted); rows.append(row); candidates.extend(cr)
    write_csv(out_dir/"metrics.csv",rows); write_csv(out_dir/"candidates.csv",candidates)
    (out_dir/"summary.json").write_text(json.dumps({"method":name,"config":asdict(DEFAULT),"omitted_features":list(omitted),
       "metrics":summarize(rows),"mean_proposal_foreground_overlap_diagnostic":float(np.mean([r["proposal_foreground_overlap_diagnostic"] for r in rows])),
       "metric_interpretation":"Agreement with SAM pseudo-labels; foreground overlap diagnostic is not instance proposal recall."},indent=2)+"\n")
    return rows


def run_e1() -> None:
    # Per-source proposals saved as separate diagnostic directories.
    run_reference_audit()
    for mode in ("mser","canny","combined"):
        run_variant(mode,mode,ROOT/f"outputs/e1_{mode}_proposals")


def run_reference_audit() -> None:
    """Audit reference availability and mask integrity without scoring predictions."""
    out=ROOT/"outputs/e1_reference_audit"; out.mkdir(parents=True,exist_ok=True)
    refs=references(); rows=[]; instance_counts={}
    for p in sorted((ROOT/"outputs/automatic_masks").glob("*_object_*_mask.png")):
        stem=p.name.split("_object_",1)[0].rsplit("__",1)[-1]
        instance_counts[stem]=instance_counts.get(stem,0)+1
    images=sorted((ROOT/"dataset").glob("*/*/*.jpg"))
    for image_path in images:
        image=cv2.imread(str(image_path)); ref_path=refs.get(image_path.stem)
        status="ok"; shape_ok=False; foreground_ratio=float("nan"); components=-1; binary_values=""
        if image is None: status="unreadable_image"
        elif ref_path is None: status="missing_reference"
        else:
            ref=cv2.imread(str(ref_path),cv2.IMREAD_GRAYSCALE)
            if ref is None: status="unreadable_reference"
            else:
                shape_ok=ref.shape==image.shape[:2]
                if not shape_ok: status="dimension_mismatch"
                foreground_ratio=float((ref>0).mean())
                binary_values=";".join(map(str,np.unique(ref).tolist()))
                n,_=cv2.connectedComponents((ref>0).astype(np.uint8),8); components=n-1
                if foreground_ratio==0: status="empty_reference"
        rows.append({"image":image_path.relative_to(ROOT).as_posix(),"status":status,
                     "reference_mask":ref_path.relative_to(ROOT).as_posix() if ref_path else "",
                     "dimensions_match":shape_ok,"reference_foreground_ratio":foreground_ratio,
                     "connected_components_8":components,"raw_grayscale_values":binary_values,
                     "sam_object_mask_files":instance_counts.get(image_path.stem,0),
                     "valid_independent_instance_labels":False})
    write_csv(out/"audit.csv",rows)
    counts={}
    for r in rows: counts[r["status"]]=counts.get(r["status"],0)+1
    (out/"summary.json").write_text(json.dumps({"images_audited":len(rows),"status_counts":counts,
      "sam_reference_masks_indexed":len(refs),"images_with_sam_object_mask_files":sum(n>0 for n in instance_counts.values()),
      "instance_label_note":"SAM object masks are unverified pseudo-labels and are not treated as valid independent instance ground truth; instance proposal recall is therefore not reported."},indent=2)+"\n")


def run_e2() -> dict[str,list[dict]]:
    ablations={"full":(),"without_edge_support":("edge_support",),"without_lab_contrast":("lab_contrast",),"without_smoothness":("interior_smoothness",)}
    result={}
    for name,omit in ablations.items():
        result[name]=run_variant(f"combined_rank_{name}","combined",ROOT/f"outputs/e2_ranking_{name}",omit)
    return result


def paired(base: list[dict], new: list[dict], name: str) -> list[dict]:
    a={r["image"]:r for r in base}; b={r["image"]:r for r in new}
    if a.keys()!=b.keys(): raise ValueError(f"Paired comparison identities differ for {name}")
    rows=[]
    for image in a:
        da=b[image]["iou_vs_sam"]-a[image]["iou_vs_sam"]
        dd=b[image]["dice_vs_sam"]-a[image]["dice_vs_sam"]
        rows.append({"image":image,"comparison":name,"iou_change":da,"dice_change":dd,
                     "iou_outcome":"improved" if da>1e-12 else "worsened" if da<-1e-12 else "unchanged",
                     "dice_outcome":"improved" if dd>1e-12 else "worsened" if dd<-1e-12 else "unchanged"})
    return rows


def run_e4(e0_rows: list[dict], e2_rows: dict[str,list[dict]]) -> tuple[str,list[dict]]:
    configs={"trial5":e0_rows}
    variants=[("mser","mser"),("canny","canny"),("combined","combined"),("combined_grabcut","combined_grabcut")]
    for label,mode in variants:
        cv2.setRNGSeed(SEED)
        rows=[]; crows=[]
        for p in read_manifest(ROOT/"splits/dev.txt"):
            r,c=process_image(p,mode,ROOT/f"outputs/e3_{label}"); rows.append(r); crows.extend(c)
        out=ROOT/f"outputs/e3_{label}"; write_csv(out/"metrics.csv",rows); write_csv(out/"candidates.csv",crows)
        (out/"summary.json").write_text(json.dumps({"method":label,"config":asdict(DEFAULT),"metrics":summarize(rows),
           "metric_interpretation":"Agreement with SAM pseudo-labels."},indent=2)+"\n")
        configs[label]=rows
    summaries={k:summarize(v) for k,v in configs.items()}
    all_pairs=[]
    baseline=configs["trial5"]
    pair_summary={}
    for name,rows in configs.items():
        if name!="trial5":
            comparison=paired(baseline,rows,name); all_pairs.extend(comparison)
            pair_summary[name]={f"iou_{kind}_count":sum(r["iou_outcome"]==kind for r in comparison) for kind in ("improved","unchanged","worsened")}
            pair_summary[name].update({"mean_iou_change_vs_trial5":float(np.mean([r["iou_change"] for r in comparison])),
                                       "mean_dice_change_vs_trial5":float(np.mean([r["dice_change"] for r in comparison]))})
    write_csv(ROOT/"outputs/e4_controlled_comparison/comparison.csv",[
       {"method":k,**v,**pair_summary.get(k,{})} for k,v in summaries.items()])
    write_csv(ROOT/"outputs/e4_controlled_comparison/paired_per_image.csv",all_pairs)
    # Minimal prespecified ranking ablation; no search over thresholds or weights.
    ablation=[]
    full=e2_rows["full"]
    for name,rows in e2_rows.items():
        s=summarize(rows); changes=paired(full,rows,name)
        ablation.append({"ablation":name,**s,"mean_iou_change_vs_full":float(np.mean([r["iou_change"] for r in changes])) if changes else 0.0})
    write_csv(ROOT/"outputs/e4_controlled_comparison/ranking_ablation.csv",ablation)
    (ROOT/"outputs/e4_controlled_comparison/summary.json").write_text(json.dumps({"summaries":summaries,"ablation":ablation,
      "selection_rule":"Highest development mean IoU vs SAM; tie break by higher mean Dice, then fewer zero-IoU images. This chooses a method on development only; it is not ground-truth accuracy.",
      "paired_changes_vs_trial5":pair_summary,
      "evaluation_access_note":"Evaluation artifacts existed before these experiments; E5 is a frozen final comparison, not a pristine blind test."},indent=2)+"\n")
    # Candidate proposal generation/scoring uses image-only features. Final method may use dev SAM agreement for method selection.
    chosen=max(summaries,key=lambda k:(summaries[k]["mean_iou_vs_sam"],summaries[k]["mean_dice_vs_sam"],-summaries[k]["zero_iou_count"]))
    return chosen,configs[chosen]


def run_e5(chosen: str) -> None:
    """One evaluation pass; no per-image evaluation output is read for tuning."""
    out=ROOT/"outputs/e5_final_evaluation"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite existing E5 output directory: {out}")
    out.mkdir(parents=True,exist_ok=False)
    config=DEFAULT
    if chosen != "trial5": cv2.setRNGSeed(SEED)
    if chosen == "trial5":
        t5=load_trial5_module()
        frozen_config={"source":"lab_illumination_segmentation.py","channel":t5.CHANNEL,
          "background_sigma":t5.BACKGROUND_SIGMA,"smoothing":t5.SMOOTHING,"median_kernel":t5.MEDIAN_KERNEL,
          "bilateral_params":list(t5.BILATERAL_PARAMS),"morph_kernel_size":t5.MORPH_KERNEL_SIZE,
          "min_area_ratio":t5.MIN_AREA_RATIO,"max_area_ratio":t5.MAX_AREA_RATIO,
          "max_aspect_ratio":t5.MAX_ASPECT_RATIO,"min_circularity":t5.MIN_CIRCULARITY,
          "penalize_border_touch":t5.PENALIZE_BORDER_TOUCH,"border_penalty":t5.BORDER_PENALTY,
          "border_margin":t5.BORDER_MARGIN,"fallback":"central-ellipse initialized GrabCut, 5 iterations"}
    else: frozen_config=asdict(config)
    rows=[]; candidates=[]
    for p in read_manifest(ROOT/"splits/eval.txt"):
        if chosen=="trial5":
            # Should selection exclude Trial 5, but retain a correct deterministic path.
            module=load_trial5_module(); image=cv2.imread(str(p)); pred,_,fallback=module.segment(image)
            ref=load_reference(references()[p.stem],image.shape[:2]); iou,dice=mask_metrics(pred,ref)
            mdir=out/"masks"; mdir.mkdir(parents=True,exist_ok=True); mp=mdir/f"{p.stem}_mask.png"; save_image(mp,binary_mask(pred))
            rows.append({"image":p.relative_to(ROOT).as_posix(),"mode":"trial5","iou_vs_sam":iou,"dice_vs_sam":dice,"foreground_ratio":float((pred>0).mean()),"candidate_count":0,"selected_count":0,"grabcut_status":str(fallback),"prediction_mask":mp.relative_to(ROOT).as_posix(),"overlay":"","proposal_visualization":"","proposal_foreground_overlap_diagnostic":float("nan")})
        else:
            row,cr=process_image(p,chosen,out,config); rows.append(row); candidates.extend(cr)
    write_csv(out/"metrics.csv",rows); write_csv(out/"candidates.csv",candidates)
    # Representative failure sheet: lowest agreement rows, with existing overlays.
    failures=sorted(rows,key=lambda r:r["iou_vs_sam"])[:12]
    thumbs=[]
    for row in failures:
        p=ROOT/row["image"]; image=cv2.imread(str(p)); mask=cv2.imread(str(ROOT/row["prediction_mask"]),0)
        if image is None or mask is None:
            raise ValueError(f"Cannot build E5 failure sheet for {row['image']}")
        thumb=cv2.resize(overlay(image,mask),(320,320)); cv2.putText(thumb,f"IoU-SAM {row['iou_vs_sam']:.3f}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        thumbs.append(thumb)
    if not thumbs:
        raise RuntimeError("E5 produced no images for the failure-analysis sheet")
    cols=3; rows_n=math.ceil(len(thumbs)/cols); sheet=np.zeros((rows_n*320,cols*320,3),np.uint8)
    for i,t in enumerate(thumbs): sheet[(i//cols)*320:(i//cols+1)*320,(i%cols)*320:(i%cols+1)*320]=t
    save_image(out/"failure_analysis.jpg",sheet)
    # `summary.json` is the completion marker. Write it only after every output,
    # including the failure sheet, has been successfully produced.
    summary={"status":"completed","selected_development_method":chosen,"frozen_config":frozen_config,
       "metrics":summarize(rows),"interpretation":"Agreement with SAM pseudo-labels, not ground-truth accuracy.",
       "evaluation_note":"Existing eval artifacts predated this run; no E5 scores were used for tuning."}
    temporary=out/"summary.json.tmp"
    temporary.write_text(json.dumps(summary,indent=2)+"\n")
    temporary.replace(out/"summary.json")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["audit","e0","e1","e2","e3","e4","e5","all"]); ap.add_argument("--selected",default=None)
    args=ap.parse_args()
    if args.stage=="audit": run_reference_audit()
    if args.stage in ("e0","all"): e0=run_e0()
    else: e0=None
    if args.stage in ("e1","all"): run_e1()
    if args.stage in ("e2","all"): e2=run_e2()
    else: e2=None
    if args.stage in ("e3",):
        for label,mode in [("mser","mser"),("canny","canny"),("combined","combined"),("combined_grabcut","combined_grabcut")]:
            run_variant(label,mode,ROOT/f"outputs/e3_{label}")
    if args.stage in ("e4","all"):
        if e0 is None:
            with (ROOT/"outputs/e0_baseline_reproduction/metrics.csv").open(newline="") as f: e0=list(csv.DictReader(f))
            for r in e0:
                for key in ("iou_vs_sam","dice_vs_sam","foreground_ratio"): r[key]=float(r[key])
        if e2 is None:
            e2={}
            for name in ("full","without_edge_support","without_lab_contrast","without_smoothness"):
                with (ROOT/f"outputs/e2_ranking_{name}/metrics.csv").open(newline="") as f: e2[name]=list(csv.DictReader(f))
                for r in e2[name]:
                    for key in ("iou_vs_sam","dice_vs_sam","foreground_ratio"):r[key]=float(r[key])
        chosen,_=run_e4(e0,e2)
        (ROOT/"outputs/e4_controlled_comparison/selected_method.txt").write_text(chosen+"\n")
        if args.stage=="all": run_e5(chosen)
    if args.stage=="e5":
        chosen=args.selected
        if not chosen:
            chosen=(ROOT/"outputs/e4_controlled_comparison/selected_method.txt").read_text().strip()
        run_e5(chosen)

if __name__=="__main__": main()
