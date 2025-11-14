# geolife_pipeline.py
# GeoLife -> Preprocess -> Time/Space/Spatio-temporal metrics -> JSON for LLM
# Python >=3.10
# pip install pandas numpy geopandas shapely pyproj scikit-learn scipy fastdtw tqdm folium

from __future__ import annotations
import os, json, math, glob, itertools, warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from pyproj import Transformer

warnings.filterwarnings("ignore", category=UserWarning)

# ===== Utils

BEIJING_TZ = timezone(timedelta(hours=8))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlambda/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def to_local(dt_gmt: datetime) -> datetime:
    if dt_gmt.tzinfo is None:
        dt_gmt = dt_gmt.replace(tzinfo=timezone.utc)
    return dt_gmt.astimezone(BEIJING_TZ)

def parse_excel_serial_day(serial: float) -> datetime:
    # Excel day serial (days since 1899-12-30)
    epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
    return epoch + timedelta(days=float(serial))

def time_entropy_hourly(timestamps_local: List[datetime]) -> float:
    if len(timestamps_local) == 0:
        return float('nan')
    hours = [dt.hour for dt in timestamps_local]
    hist, _ = np.histogram(hours, bins=np.arange(25))
    p = hist / hist.sum() if hist.sum() > 0 else hist
    p = p[p>0]
    H = -(p * np.log2(p)).sum()
    return H / np.log2(24)  # normalized [0,1]

def radius_of_gyration(xy: np.ndarray) -> float:
    if len(xy) == 0:
        return float('nan')
    centroid = xy.mean(axis=0, keepdims=True)
    r2 = ((xy - centroid)**2).sum(axis=1)
    return math.sqrt(r2.mean())

def std_ellipse_params(xy: np.ndarray) -> Dict[str, float]:
    # PCA-based standard deviational ellipse (1-sigma axes)
    if len(xy) < 3:
        return {"sx": float('nan'), "sy": float('nan'), "theta_deg": float('nan')}
    xy_c = xy - xy.mean(axis=0)
    pca = PCA(n_components=2).fit(xy_c)
    # std along components:
    var = pca.explained_variance_
    sx, sy = np.sqrt(var[0]), np.sqrt(var[1])
    # angle of first component (in degrees)
    theta = math.degrees(math.atan2(pca.components_[0,1], pca.components_[0,0]))
    return {"sx": float(sx), "sy": float(sy), "theta_deg": float(theta)}

def project_lonlat_to_utm50N(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    # Beijing ~ 116E -> UTM Zone 50N (EPSG:4326 -> 32650)
    transformer = Transformer.from_crs("EPSG:4326","EPSG:32650", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.vstack([x,y]).T

def kde_hotspots(xy: np.ndarray, bandwidth: float=300, grid_step: float=200, topk:int=3):
    if len(xy) < 10:
        return []
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(xy)
    xmin, ymin = xy.min(axis=0) - 3*bandwidth
    xmax, ymax = xy.max(axis=0) + 3*bandwidth
    xs = np.arange(xmin, xmax, grid_step)
    ys = np.arange(ymin, ymax, grid_step)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    z = kde.score_samples(grid)  # log-density
    idx = np.argsort(z)[::-1][:topk]
    centers = grid[idx]
    scores = z[idx]
    return [{"x": float(c[0]), "y": float(c[1]), "log_density": float(s)} for c,s in zip(centers, scores)]

# ===== Data Loading

@dataclass
class TrajPoint:
    lat: float
    lon: float
    alt_ft: float
    t_gmt: datetime   # original in GMT
    t_local: datetime # converted to Asia/Shanghai (Beijing)

def load_plt_file(path: str) -> List[TrajPoint]:
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # first 6 lines useless per spec; from line 7: csv fields
    for line in lines[6:]:
        line=line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 7:
            continue
        lat = float(fields[0]); lon = float(fields[1])
        alt_ft = float(fields[3])
        excel_date = float(fields[4])
        t_gmt = parse_excel_serial_day(excel_date)  # tz=UTC
        t_local = to_local(t_gmt)
        pts.append(TrajPoint(lat, lon, alt_ft, t_gmt, t_local))
    return pts

def load_user_labels(user_dir: str) -> pd.DataFrame:
    # optional label file in user folder
    # try common names: "labels.txt" or "labels.plt" (GeoLife variations)
    label_files = glob.glob(os.path.join(user_dir, "labels*.txt")) + \
                  glob.glob(os.path.join(user_dir, "labels*.plt")) + \
                  glob.glob(os.path.join(user_dir, "labels*"))
    if not label_files:
        return pd.DataFrame(columns=["start_time","end_time","mode"])
    path = label_files[0]
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.lower().startswith("start"):
                continue
            parts = line.split()
            # expected: "YYYY/MM/DD HH:MM:SS YYYY/MM/DD HH:MM:SS mode"
            if len(parts) < 3:
                continue
            start = " ".join(parts[0:2])
            end   = " ".join(parts[2:4]) if len(parts)>=4 else None
            mode  = parts[-1]
            try:
                st = datetime.strptime(start, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
                et = datetime.strptime(end,   "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc) if end else None
                rows.append({"start_time": st, "end_time": et, "mode": mode})
            except:
                continue
    return pd.DataFrame(rows)

# ===== Stay Point Detection (simple & robust)
# Stay point: within dist_th (m) for at least time_th (s)
def detect_stay_points(points: List[TrajPoint], dist_th_m=200, time_th_s=20*60) -> List[Dict]:
    stays = []
    n = len(points)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            d = haversine_m(points[i].lat, points[i].lon, points[j].lat, points[j].lon)
            if d > dist_th_m:
                dt = (points[j-1].t_local - points[i].t_local).total_seconds()
                if dt >= time_th_s:
                    # centroid over i..j-1
                    lats = [p.lat for p in points[i:j]]
                    lons = [p.lon for p in points[i:j]]
                    t0 = points[i].t_local; t1 = points[j-1].t_local
                    stays.append({
                        "lat": float(np.mean(lats)),
                        "lon": float(np.mean(lons)),
                        "t_start": t0.isoformat(),
                        "t_end": t1.isoformat(),
                        "duration_s": float(dt)
                    })
                break
            j += 1
        i = max(j, i+1)
    return stays

# ===== Trip Segmentation by stay points
def segment_trips(points: List[TrajPoint], stays: List[Dict], gap_th_min=30) -> List[Dict]:
    # Segment by long temporal gaps OR entering/exiting stay windows
    if not points:
        return []
    trips = []
    curr = [points[0]]
    for prev, p in zip(points[:-1], points[1:]):
        gap_min = (p.t_local - prev.t_local).total_seconds()/60.0
        if gap_min >= gap_th_min:
            trips.append(curr); curr=[p]; continue
        curr.append(p)
    if curr:
        trips.append(curr)

    # build records
    trip_records=[]
    for seg in trips:
        if len(seg) < 2: 
            continue
        start, end = seg[0], seg[-1]
        # distance sum
        dist = np.sum([haversine_m(seg[k].lat, seg[k].lon, seg[k+1].lat, seg[k+1].lon) for k in range(len(seg)-1)])
        dur_s = (end.t_local - start.t_local).total_seconds()
        trip_records.append({
            "t_start": start.t_local.isoformat(),
            "t_end": end.t_local.isoformat(),
            "duration_s": float(dur_s),
            "distance_m": float(dist),
            "n_points": len(seg),
            "start_lat": start.lat, "start_lon": start.lon,
            "end_lat": end.lat, "end_lon": end.lon
        })
    return trip_records

# ===== Time Dimension Metrics
def time_metrics(points: List[TrajPoint], trips: List[Dict]) -> Dict:
    timestamps = [p.t_local for p in points]
    n_trips = len(trips)
    total_move_time = sum(t["duration_s"] for t in trips)
    start_hours = [datetime.fromisoformat(t["t_start"]).hour for t in trips] if trips else []
    day_night = {"day_count": 0, "night_count": 0}
    for t in timestamps:
        h = t.hour
        # Day: 06:00-17:59 local; Night otherwise
        if 6 <= h < 18:
            day_night["day_count"] += 1
        else:
            day_night["night_count"] += 1
    ent = time_entropy_hourly(timestamps)
    start_hist, _ = np.histogram(start_hours, bins=np.arange(25)) if start_hours else (np.zeros(24), None)
    return {
        "n_trips": int(n_trips),
        "total_move_time_s": float(total_move_time),
        "trip_start_hist_24h": start_hist.tolist(),
        "time_entropy_hourly_norm": float(ent),
        "day_night_ratio": float(day_night["day_count"] / max(1, day_night["night_count"]))
    }

# ===== Space Dimension Metrics
def space_metrics(points: List[TrajPoint]) -> Dict:
    if len(points)==0:
        return {k: float('nan') for k in ["hull_area_m2","radius_of_gyration_m","ellipse_sx_m","ellipse_sy_m","ellipse_theta_deg"]}
    lons = np.array([p.lon for p in points])
    lats = np.array([p.lat for p in points])
    xy = project_lonlat_to_utm50N(lons, lats)
    # convex hull area
    gdf = gpd.GeoSeries([Point(xy[i,0], xy[i,1]) for i in range(len(xy))], crs="EPSG:32650")
    hull = gdf.unary_union.convex_hull
    hull_area = hull.area if isinstance(hull, Polygon) else float('nan')
    # radius of gyration
    rog = radius_of_gyration(xy)
    # std ellipse
    ell = std_ellipse_params(xy)
    # hotspots via KDE
    hotspots = kde_hotspots(xy, bandwidth=300, grid_step=200, topk=10)
    return {
        "hull_area_m2": float(hull_area),
        "radius_of_gyration_m": float(rog),
        "ellipse_sx_m": float(ell["sx"]),
        "ellipse_sy_m": float(ell["sy"]),
        "ellipse_theta_deg": float(ell["theta_deg"]),
        "kde_hotspots_utm32650": hotspots
    }

# ===== Spatio-temporal Metrics
def spatiotemporal_metrics(stays: List[Dict], trips: List[Dict]) -> Dict:
    # day/night stay duration ratio; mean stay duration; top-5 stays by duration
    total_day_s = 0.0; total_night_s = 0.0
    for s in stays:
        t0 = datetime.fromisoformat(s["t_start"])
        t1 = datetime.fromisoformat(s["t_end"])
        dur = (t1 - t0).total_seconds()
        if 6 <= t0.hour < 18:
            total_day_s += dur
        else:
            total_night_s += dur
    top5 = sorted(stays, key=lambda s: s["duration_s"], reverse=True)[:5]

    # Trajectory similarity (DTW) between daily speed series (coarse)
    # Build daily speed time series at 5-min bins
    def daily_speed_series(trips):
        series_by_day={}
        for tr in trips:
            t0 = datetime.fromisoformat(tr["t_start"])
            t1 = datetime.fromisoformat(tr["t_end"])
            day = t0.date()
            speed = tr["distance_m"] / max(1.0, tr["duration_s"])  # m/s
            # fill across bins roughly proportional to duration
            bins = max(1, int(tr["duration_s"]/300))
            if day not in series_by_day: series_by_day[day]=[]
            series_by_day[day].extend([speed]*bins)
        return series_by_day

    sers = daily_speed_series(trips)
    days = list(sers.keys())
    dtw_pairs=[]
    for i in range(len(days)):
        for j in range(i+1, len(days)):
            d1 = np.array(sers[days[i]]) if len(sers[days[i]])>0 else np.array([0.])
            d2 = np.array(sers[days[j]]) if len(sers[days[j]])>0 else np.array([0.])
            dist, _ = fastdtw(d1, d2)
            dtw_pairs.append(((str(days[i]), str(days[j])), float(dist)))
    dtw_pairs = sorted(dtw_pairs, key=lambda x: x[1])[:5]

    return {
        "stay_day_night_ratio": float(total_day_s / max(1.0, total_night_s)),
        "mean_stay_duration_s": float(np.mean([s["duration_s"] for s in stays]) if stays else float('nan')),
        "top5_stays": top5,
        "dtw_daily_speed_top5_similar_pairs": [{"days": pair, "dtw_distance": dist} for (pair, dist) in dtw_pairs]
    }

# ===== User-level Processing

def process_user(user_id: str, user_dir: str) -> Optional[Dict]:
    traj_dir = os.path.join(user_dir, "Trajectory")
    if not os.path.isdir(traj_dir):
        return None
    plt_files = sorted(glob.glob(os.path.join(traj_dir, "*.plt")))
    all_points: List[TrajPoint] = []
    for pf in plt_files:
        pts = load_plt_file(pf)
        all_points.extend(pts)
    all_points.sort(key=lambda p: p.t_local)
    if not all_points:
        return None

    # stays & trips
    stays = detect_stay_points(all_points, dist_th_m=200, time_th_s=20*60)
    trips = segment_trips(all_points, stays, gap_th_min=30)

    # metrics
    tmet = time_metrics(all_points, trips)
    smet = space_metrics(all_points)
    stmet = spatiotemporal_metrics(stays, trips)

    # labels (optional)
    labels_df = load_user_labels(user_dir)
    label_summary = None
    if not labels_df.empty:
        # duration per mode (hours) & count
        labels_df["dur_h"] = (labels_df["end_time"] - labels_df["start_time"]).dt.total_seconds()/3600.0
        label_summary = labels_df.groupby("mode").agg(n=("mode","count"), hours=("dur_h","sum")).reset_index().to_dict(orient="records")

    return {
        "user_id": user_id,
        "n_points": len(all_points),
        "time_metrics": tmet,
        "space_metrics": smet,
        "spatiotemporal_metrics": stmet,
        "stays": stays,          # 为了体积，默认仅保留前50个；需要可调
        "trips": trips,         # 同上
        "label_summary": label_summary
    }

# ===== Runner

def run_geolife(root_dir: str, output_json: str, max_users: Optional[int]=None):
    """
    root_dir: GeoLife root (the folder that contains 'Data/000/Trajectory/*.plt')
    """
    data_dir = os.path.join(root_dir, "Data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expect GeoLife 'Data' under {root_dir}")
    user_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)])
    if max_users:
        user_dirs = user_dirs[:max_users]

    results=[]
    for ud in tqdm(user_dirs, desc="Processing users"):
        user_id = os.path.basename(ud)
        rec = process_user(user_id, ud)
        if rec:
            results.append(rec)

    bundle = {
        "dataset": "GeoLife 1.3 (MSRA, 2007-2012)",
        "timezone_processing": "Converted GMT -> Asia/Shanghai (UTC+8) for temporal metrics",
        "users": results
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return output_json

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to GeoLife root dir (containing Data/)")
    ap.add_argument("--out",  required=True, help="Output JSON path")
    ap.add_argument("--max_users", type=int, default=None, help="Optional limit for quick tests")
    args = ap.parse_args()
    run_geolife(args.root, args.out, args.max_users)
