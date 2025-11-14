# -*- coding: utf-8 -*-
"""
Trajectory Analysis → Web Map (Folium) - Beijing Only with Raw PLT Support
--------------------------------------
Reads a JSON with {users:[{points:[{lat, lon, t}], stays:[...] }]} (best-effort schema),
filters points to Beijing area only,
draws points & trajectory, convex hull, 1σ/2σ/3σ Standard Deviational Ellipses,
and KDE Hotspot regions (80/90/95%) + HeatMap layer.
Also supports visualizing raw PLT trajectory files.

USAGE (local):
    # Without raw PLT files
    python trajectory_analysis_beijing_only.py --input output.json --html trajectory_analysis_map.html
    
    # With raw PLT files
    python trajectory_analysis_beijing_only.py --input output.json --html trajectory_analysis_map.html --plt_folder ./Geolife/Data/000

Front-end style knobs live in the STYLE dict below.
"""

import json, math, argparse
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pyproj import Transformer, CRS

import folium
from folium.plugins import HeatMap, MousePosition, Fullscreen, MiniMap, MeasureControl
from folium import Popup
from folium.plugins import AntPath

# ===========================
# FRONT-END STYLE CONTROLS
# ===========================
STYLE = {
    # Base tiles: "cartodbpositron", "OpenStreetMap", "CartoDB dark_matter", etc.
    "base_tiles": "cartodbpositron",
    "map_zoom_start": 5,

    # Point markers
    "point_radius": 3,
    "point_color": "#1f77b4",
    "point_fill_opacity": 0.7,
    "point_opacity": 0.8,

    # Trajectory polyline
    "traj_weight": 2,
    "traj_opacity": 0.6,
    "traj_color": "#1f77b4",

    # Stay markers
    "stay_icon_color": "#e67e22",

    # Convex hull polygon
    "hull_stroke_color": "#555555",
    "hull_stroke_weight": 2,
    "hull_fill_color": "#cccccc",
    "hull_fill_opacity": 0.10,

    # Standard Deviational Ellipse (1σ/2σ/3σ)
    "sde_styles": {
        1.0: {"color": "#555555", "weight": 3, "opacity": 0.9},
        2.0: {"color": "#555555", "weight": 3, "opacity": 0.8},
        3.0: {"color": "#555555", "weight": 3, "opacity": 0.7},
    },

    # KDE HeatMap
    "heat_radius": 18,
    "heat_blur": 25,
    "heat_gradient": {
        0.0: "#ffffff",
        0.2: "#7fcdbb",
        0.4: "#41b6c4",
        0.6: "#1d91c0",
        0.8: "#225ea8",
        1.0: "#0c2c84",
    },

    # KDE HDR polygons
    "kde_level_colors": {
        0.80: "#fdd49e",
        0.90: "#fdbb84",
        0.95: "#fc8d59",
    },
    "kde_fill_opacity": 0.12,
    "kde_stroke_weight": 1,
    "kde_stroke_color": "#555555",
    
    # Raw PLT trajectories
    "raw_traj_weight": 1.5,
    "raw_traj_opacity": 0.4,
    "raw_traj_color": "#ff6b6b",
}

# ===========================
# HELPERS
# ===========================
def read_plt_file(plt_path):
    """读取单个PLT文件，返回轨迹点列表"""
    points = []
    try:
        with open(plt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[6:]  # 跳过前6行
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    date_str = parts[5]
                    time_str = parts[6]
                    t = f"{date_str} {time_str}"
                    points.append({"lat": lat, "lon": lon, "t": t})
    except Exception as e:
        print(f"Error reading {plt_path}: {e}")
    return points

def load_plt_trajectories(data_folder):
    """加载指定文件夹下所有PLT文件"""
    import os
    import glob
    
    all_trajectories = []
    plt_files = glob.glob(os.path.join(data_folder, "**/*.plt"), recursive=True)
    
    print(f"Found {len(plt_files)} PLT files in {data_folder}")
    
    for plt_file in plt_files:
        traj_points = read_plt_file(plt_file)
        if traj_points:
            all_trajectories.append(traj_points)
    
    return all_trajectories

def normalize_points(records):
    rows = []
    for r in records:
        lat = r.get("lat") or r.get("latitude")
        lon = r.get("lon") or r.get("lng") or r.get("longitude")
        t = r.get("t") or r.get("timestamp") or r.get("time")
        if isinstance(t, (int, float)):
            try:
                t_iso = datetime.utcfromtimestamp(t).isoformat() + "Z"
            except Exception:
                t_iso = None
        elif isinstance(t, str):
            t_iso = t
        else:
            t_iso = None
        if lat is not None and lon is not None:
            rows.append({"lat": float(lat), "lon": float(lon), "t": t_iso})
    return pd.DataFrame(rows)

def to_utm_df(df_ll, center_lon):
    zone = int(math.floor((center_lon + 180)/6) + 1)
    crs_src = CRS.from_epsg(4326)
    epsg = 32600 + zone  # N hemisphere
    crs_dst = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    xs, ys = transformer.transform(df_ll["lon"].values, df_ll["lat"].values)
    out = df_ll.copy()
    out["x"] = xs
    out["y"] = ys
    out.attrs["epsg"] = epsg
    return out

def convex_hull_area(df_ll):
    if df_ll.empty:
        return None, 0.0
    pts = [Point(lon, lat) for lon, lat in zip(df_ll["lon"], df_ll["lat"])]
    hull = unary_union(pts).convex_hull
    # area in meters via projection
    df_proj = to_utm_df(df_ll, center_lon=float(df_ll["lon"].mean()))
    coords = list(zip(df_proj["x"], df_proj["y"]))
    poly = Polygon(coords).convex_hull
    return hull, poly.area

def standard_deviational_ellipse(df_ll, n_std=1.0):
    if df_ll.empty:
        return None
    df_proj = to_utm_df(df_ll, center_lon=float(df_ll["lon"].mean()))
    X = df_proj[["x","y"]].values
    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    sx, sy = np.sqrt(eigvals) * n_std
    angle = math.atan2(eigvecs[1,0], eigvecs[0,0])  # radians

    t = np.linspace(0, 2*np.pi, 256)
    ex = mean[0] + sx * np.cos(t)
    ey = mean[1] + sy * np.sin(t)
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])
    exy = np.vstack((ex - mean[0], ey - mean[1]))
    rot = R @ exy
    exr = rot[0,:] + mean[0]
    eyr = rot[1,:] + mean[1]

    epsg = df_proj.attrs["epsg"]
    transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    lons, lats = transformer.transform(exr, eyr)
    return list(zip(lats, lons)), float(sx), float(sy), math.degrees(angle)

def assign_city(df_ll):
    """Filter points to Beijing area only (within 400km of Beijing center)"""
    if df_ll.empty:
        df_ll["city"] = []
        return df_ll
    
    beijing_center = (39.9042, 116.4074)
    
    def is_beijing(lat, lon):
        clat, clon = beijing_center
        R = 6371000.0
        dlat = math.radians(clat-lat)
        dlon = math.radians(clon-lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat))*math.cos(math.radians(clat))*math.sin(dlon/2)**2
        d = 2*R*math.asin(math.sqrt(a))
        return "Beijing" if d < 400e3 else "Other"
    
    df_ll = df_ll.copy()
    df_ll["city"] = [is_beijing(lat, lon) for lat, lon in zip(df_ll["lat"], df_ll["lon"])]
    return df_ll

def kde_contours(df_ll, bandwidth_m=1500.0, levels=(0.80, 0.90, 0.95)):
    if df_ll.empty:
        return {}
    from sklearn.neighbors import KernelDensity
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    from skimage import measure

    df_proj = to_utm_df(df_ll, center_lon=float(df_ll["lon"].mean()))
    X = df_proj[["x","y"]].values
    if len(X) < 10:
        return {}
    kde = KernelDensity(bandwidth=bandwidth_m, kernel='gaussian')
    kde.fit(X)
    padding = bandwidth_m * 3
    xmin, ymin = X.min(axis=0) - padding
    xmax, ymax = X.max(axis=0) + padding
    nx = ny = 200
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xv, yv = np.meshgrid(xs, ys)
    grid = np.vstack([xv.ravel(), yv.ravel()]).T
    z = np.exp(kde.score_samples(grid)).reshape(ny, nx)

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    prob = z / z.sum()

    flat = prob.ravel()
    order = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[order])

    results = {}
    for lev in levels:
        mask = np.zeros_like(flat, dtype=bool)
        mask[order[:np.searchsorted(cumsum, lev)]] = True
        mask = mask.reshape(ny, nx)

        contours = measure.find_contours(mask.astype(float), 0.5)
        polys = []
        for c in contours:
            ys_idx, xs_idx = c[:,0], c[:,1]
            xs_coords = xmin + xs_idx * dx
            ys_coords = ymin + ys_idx * dy
            coords_proj = list(zip(xs_coords, ys_coords))
            if len(coords_proj) >= 3:
                poly = Polygon(coords_proj).buffer(0)
                if poly.area > 0:
                    polys.append(poly)
        if not polys:
            continue
        merged = unary_union(polys)
        epsg = df_proj.attrs["epsg"]
        transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)

        def reproj_geom(g):
            if g.geom_type == "Polygon":
                lonlat = [transformer.transform(x,y) for x,y in g.exterior.coords]
                return [(lat, lon) for lon,lat in lonlat]
            elif g.geom_type == "MultiPolygon":
                out = []
                for gg in g.geoms:
                    lonlat = [transformer.transform(x,y) for x,y in gg.exterior.coords]
                    out.append([(lat, lon) for lon,lat in lonlat])
                return out
        results[lev] = reproj_geom(merged)
    return results

# ===========================
# MAIN
# ===========================
def run(input_path, html_out, kde_bandwidth_m=1500.0, plt_folder=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    users = data.get("users", [])
    user = users[0] if users else {}
    raw_points = []
    if "points" in user:
        raw_points = user["points"]
    elif "raw" in user:
        raw_points = user["raw"]
    elif "points" in data:
        raw_points = data["points"]

    df_points = normalize_points(raw_points)
    stays = user.get("stays", [])
    df_stays = pd.DataFrame(stays) if stays else pd.DataFrame(columns=["lat","lon","t_start","t_end","duration_s"])

    if df_points.empty and not df_stays.empty:
        rows = [{"lat": s["lat"], "lon": s["lon"], "t": s.get("t_start")} for s in stays]
        df_points = pd.DataFrame(rows)

    df_points = df_points.dropna(subset=["lat","lon"]).reset_index(drop=True)
    df_points = assign_city(df_points)
    
    # Only keep Beijing points
    df_points = df_points[df_points["city"] == "Beijing"].reset_index(drop=True)
    
    # Filter stays to Beijing only
    if not df_stays.empty:
        df_stays = assign_city(df_stays[["lat","lon"]].copy())
        df_stays_full = stays
        beijing_indices = df_stays[df_stays["city"] == "Beijing"].index.tolist()
        stays = [df_stays_full[i] for i in beijing_indices]
        df_stays = pd.DataFrame(stays) if stays else pd.DataFrame(columns=["lat","lon","t_start","t_end","duration_s"])

    # Load raw PLT trajectories if folder is provided
    raw_trajectories = []
    if plt_folder:
        import os
        if os.path.exists(plt_folder):
            raw_trajectories = load_plt_trajectories(plt_folder)
            print(f"Loaded {len(raw_trajectories)} raw trajectories from PLT files")
        else:
            print(f"Warning: PLT folder not found: {plt_folder}")

    center_lat = df_points["lat"].mean() if not df_points.empty else 39.9042
    center_lon = df_points["lon"].mean() if not df_points.empty else 116.4074
    m = folium.Map(location=[center_lat, center_lon], zoom_start=STYLE["map_zoom_start"], tiles=STYLE["base_tiles"])
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl(primary_length_unit='kilometers').add_to(m)
    MousePosition(position='bottomright', separator=' | ', prefix='Lat/Lon:').add_to(m)

    # Points + trajectory for Beijing only
    if not df_points.empty:
        fg = folium.FeatureGroup(name="Trips", show=False)
        if df_points["t"].notna().any():
            dfc2 = df_points.copy()
            def parse_time(ts):
                try:
                    return pd.to_datetime(ts)
                except Exception:
                    return pd.NaT
            dfc2["ts"] = dfc2["t"].apply(parse_time)
            dfc2 = dfc2.sort_values("ts")
            coords = list(zip(dfc2["lat"], dfc2["lon"]))
            if len(coords) >= 2:
                folium.PolyLine(
                    locations=coords, weight=STYLE["traj_weight"],
                    opacity=STYLE["traj_opacity"], color=STYLE["traj_color"]
                ).add_to(fg)
        fg.add_to(m)

    # Stays
    if not df_stays.empty:
        fg_s = folium.FeatureGroup(name="Stay Points", show=True)
        for _, s in df_stays.iterrows():
            dur = s.get("duration_s", None)
            hours = f"{dur/3600:.1f}h" if isinstance(dur, (int,float)) else "N/A"
            popup = Popup(f"Stay<br>Start: {s.get('t_start','')}<br>End: {s.get('t_end','')}<br>Duration: {hours}", max_width=280)
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=STYLE["point_radius"]+2,
                color=STYLE["stay_icon_color"],
                fill=True,
                fill_color=STYLE["stay_icon_color"],
                fill_opacity=0.9,
                opacity=0.9
            ).add_to(fg_s)
        fg_s.add_to(m)

    # 原始轨迹可视化 (Raw PLT Trajectories)
    # if raw_trajectories:
    #     fg_raw = folium.FeatureGroup(name="Beijing · Raw Trajectories", show=False)
        
    #     beijing_traj_count = 0
    #     for traj in raw_trajectories:
    #         df_traj = pd.DataFrame(traj)
    #         if df_traj.empty:
    #             continue
            
    #         # 过滤到北京范围
    #         df_traj = assign_city(df_traj)
    #         df_traj = df_traj[df_traj["city"] == "Beijing"]
            
    #         if len(df_traj) >= 2:
    #             coords = list(zip(df_traj["lat"], df_traj["lon"]))
    #             folium.PolyLine(
    #                 locations=coords,
    #                 weight=STYLE["raw_traj_weight"],
    #                 opacity=STYLE["raw_traj_opacity"],
    #                 color=STYLE["raw_traj_color"]
    #             ).add_to(fg_raw)
    #             beijing_traj_count += 1
        
    #     fg_raw.add_to(m)
    #     print(f"Visualized {beijing_traj_count} trajectories in Beijing area")
    
    # if raw_trajectories:
    #     fg_raw = folium.FeatureGroup(name="Beijing · Raw Trajectories", show=True)
        
    #     # 颜色插值函数
    #     def interpolate_color(color1, color2, ratio):
    #         """在两个颜色之间插值"""
    #         c1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #         c2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #         c = tuple(int(c1[i] + (c2[i] - c1[i]) * ratio) for i in range(3))
    #         return '#{:02x}{:02x}{:02x}'.format(*c)
        
    #     # 生成每条轨迹的基础颜色
    #     import colorsys
    #     def generate_colors(n):
    #         colors = []
    #         for i in range(n):
    #             hue = i / max(n, 1)
    #             rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    #             colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    #         return colors
        
    #     # 收集所有北京轨迹
    #     beijing_trajs = []
    #     for traj in raw_trajectories:
    #         df_traj = pd.DataFrame(traj)
    #         if df_traj.empty:
    #             continue
    #         df_traj = assign_city(df_traj)
    #         df_traj = df_traj[df_traj["city"] == "Beijing"]
    #         if len(df_traj) >= 2:
    #             beijing_trajs.append(df_traj)
        
    #     # 生成基础颜色
    #     base_colors = generate_colors(len(beijing_trajs))
        
    #     # 为每条轨迹绘制渐变效果
    #     for idx, df_traj in enumerate(beijing_trajs):
    #         coords = list(zip(df_traj["lat"], df_traj["lon"]))
    #         base_color = base_colors[idx % len(base_colors)]
            
    #         # 计算渐变终点色（稍微变暗）
    #         end_color = interpolate_color(base_color, '#2c3e50', 0.3)
            
    #         # 分段绘制，每段颜色略有变化
    #         n_segments = len(coords) - 1
    #         for i in range(n_segments):
    #             ratio = i / max(n_segments, 1)
    #             segment_color = interpolate_color(base_color, end_color, ratio)
                
    #             folium.PolyLine(
    #                 locations=[coords[i], coords[i+1]],
    #                 weight=2.5,
    #                 opacity=0.75,
    #                 color=segment_color,
    #                 tooltip=f"Trajectory {idx+1}, Segment {i+1}/{n_segments}"
    #             ).add_to(fg_raw)
        
    #     fg_raw.add_to(m)
    #     print(f"Visualized {len(beijing_trajs)} trajectories with gradient colors in Beijing area")
    # 原始轨迹可视化 (Raw PLT Trajectories)
    if raw_trajectories:
        fg_raw = folium.FeatureGroup(name="Raw Trajectories", show=True)
        
        # 生成颜色列表
        import colorsys
        def generate_colors(n):
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            return colors
        
        # 先收集所有北京轨迹
        beijing_trajs = []
        for traj in raw_trajectories:
            df_traj = pd.DataFrame(traj)
            if df_traj.empty:
                continue
            df_traj = assign_city(df_traj)
            df_traj = df_traj[df_traj["city"] == "Beijing"]
            if len(df_traj) >= 2:
                beijing_trajs.append(df_traj)
        
        # 生成颜色
        colors = generate_colors(len(beijing_trajs))
        
        # 添加动画轨迹
        for idx, df_traj in enumerate(beijing_trajs):
            coords = list(zip(df_traj["lat"], df_traj["lon"]))
            AntPath(
                locations=coords,
                color=colors[idx % len(colors)],
                weight=3,
                opacity=0.7,
                delay=800,
                dash_array=[10, 20]
            ).add_to(fg_raw)
        
        fg_raw.add_to(m)
        print(f"Visualized {len(beijing_trajs)} trajectories in Beijing area")

    # Hull + SDEs + KDE for Beijing only
    if not df_points.empty:
        # Convex hull - in its own FeatureGroup
        try:
            hull, area_m2 = convex_hull_area(df_points[["lat","lon"]])
            if hull is not None and not hull.is_empty:
                fg_hull = folium.FeatureGroup(name="Convex Hull", show=False)
                folium.Polygon(
                    locations=[(lat,lon) for lon,lat in hull.exterior.coords],
                    fill=True, fill_color=STYLE["hull_fill_color"],
                    fill_opacity=STYLE["hull_fill_opacity"],
                    color=STYLE["hull_stroke_color"],
                    weight=STYLE["hull_stroke_weight"],
                    tooltip=f"Beijing · Convex Hull\nArea: {area_m2/1e6:.2f} km²"
                ).add_to(fg_hull)
                fg_hull.add_to(m)
        except Exception:
            pass

        # SDE: 1σ/2σ/3σ - each in its own FeatureGroup
        for sigma in (1.0, 2.0, 3.0):
            try:
                ellipse_coords, sx, sy, theta_deg = standard_deviational_ellipse(df_points[["lat","lon"]], n_std=sigma)
                style = STYLE["sde_styles"].get(sigma, {"color":"#000000", "weight":2, "opacity":0.8})
                if ellipse_coords:
                    fg_sde = folium.FeatureGroup(name=f"{sigma:.0f}σ SDE", show=False)
                    folium.PolyLine(
                        locations=ellipse_coords,
                        color=style["color"],
                        weight=style["weight"],
                        opacity=style["opacity"],
                        tooltip=f"Beijing · {sigma:.0f}σ SDE\nsx={sx:.0f} m, sy={sy:.0f} m, θ={theta_deg:.1f}°"
                    ).add_to(fg_sde)
                    fg_sde.add_to(m)
            except Exception:
                pass

        # KDE heat
        heat = df_points[["lat","lon"]].values.tolist()
        if len(heat) >= 5:
            HeatMap(
                heat, name="Beijing · KDE Heat",
                radius=STYLE["heat_radius"],
                blur=STYLE["heat_blur"],
                gradient=STYLE.get("heat_gradient"),
                control=True, show=True
            ).add_to(m)

        # KDE HDR polygons - in a single FeatureGroup
        try:
            contours = kde_contours(df_points[["lat","lon"]], bandwidth_m=kde_bandwidth_m, levels=(0.80, 0.90, 0.95))
            if contours:
                fg_kde = folium.FeatureGroup(name="KDE Regions", show=False)
                for lev, geoms in contours.items():
                    if isinstance(geoms, list):
                        for poly in geoms:
                            folium.Polygon(
                                locations=poly,
                                fill=True,
                                fill_opacity=STYLE["kde_fill_opacity"],
                                fill_color=STYLE["kde_level_colors"].get(lev, "#cccccc"),
                                color=STYLE["kde_stroke_color"],
                                weight=STYLE["kde_stroke_weight"],
                                tooltip=f"Beijing · KDE {int(lev*100)}% region"
                            ).add_to(fg_kde)
                fg_kde.add_to(m)
        except Exception:
            pass
    
    if not df_points.empty:
        m.fit_bounds(df_points[["lat","lon"]].values.tolist())
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(html_out)
    return html_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=r"E:\NUS_Applied_GIS\3.Spatial_Programming\Final project\output_000.json", help="Input JSON path")
    parser.add_argument("--html", type=str, default=r"E:\NUS_Applied_GIS\3.Spatial_Programming\Final project\trajectory_map5.html", help="Output HTML path")
    parser.add_argument("--kde_bandwidth_m", type=float, default=1500.0, help="KDE bandwidth in meters")
    parser.add_argument("--plt_folder", type=str, default=r"E:\NUS_Applied_GIS\3.Spatial_Programming\Final project\Geolife Trajectories 1.3\Data\000", help="Input PLT path")
    args = parser.parse_args()
    out = run(args.input, args.html, kde_bandwidth_m=args.kde_bandwidth_m, plt_folder=args.plt_folder)
    print("Saved:", out)