import base64
import io
import math
from contextlib import contextmanager
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import rasterio
import requests
import xarray as xr
from PIL import Image
from dash import ALL, ClientsideFunction, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.colors import qualitative
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from xrspatial import viewshed

PROJECTED_CRS = "EPSG:26912"  # NAD83 / UTM zone 12N, suitable for Utah
GEOGRAPHIC_CRS = "EPSG:4326"
WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_VERSION = "v200"
WORLDCOVER_YEAR = "2021"

DEFAULT_BBOX = {
    "min_lon": -113.90,
    "min_lat": 39.8667,
    "max_lon": -111.15,
    "max_lat": 42.65,
}

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 2400
DEFAULT_NODE_HEIGHT_M = 8.0
DEFAULT_FREQ_MHZ = 915.0
DEFAULT_TX_POWER_DBM = 30.0
DEFAULT_TX_GAIN_DBI = 6.0
DEFAULT_RX_GAIN_DBI = 2.0
DEFAULT_OTHER_LOSSES_DB = 3.0
MIN_LINK_RSSI_DBM = -140.0

COLOR_SEQUENCE = qualitative.Dark24 + qualitative.Bold + qualitative.Safe + qualitative.Set3
ELEVATION_COLOR_SCALE_OPTIONS = ["Magma", "Viridis", "Cividis", "Inferno", "Plasma", "Greys"]
RSSI_COLOR_SCALE_OPTIONS = ["Turbo", "Viridis", "Cividis", "Inferno", "Plasma", "RdYlGn"]

MAP_OVERLAY_BASE_STYLE = {
    "position": "absolute",
    "inset": "0",
    "display": "none",
    "alignItems": "center",
    "justifyContent": "center",
    "backgroundColor": "rgba(255,255,255,0.45)",
    "backdropFilter": "blur(1px)",
    "fontWeight": "600",
    "fontSize": "18px",
    "zIndex": 10,
}

OFFSETS = {
    10: 8,
    20: 3,
    30: 1,
    40: 2,
    50: 10,
    60: 0,
    70: 0,
    80: 0,
    90: 1,
    95: 5,
    100: 0,
}

ATTENUATION = {
    0: 0,
    10: 1,
    20: 0.25,
    30: 0.1,
    40: 0.25,
    50: 10,
    60: 0,
    70: 0,
    80: 0.5,
    90: 1,
    95: 2,
    100: 0,
}

V_OFFSET = np.vectorize(lambda value: OFFSETS.get(int(value), 0), otypes=[np.float32])
V_ATTENUATION = np.vectorize(lambda value: ATTENUATION.get(int(value), 0), otypes=[np.float32])

ANALYSIS_CONTEXT = {}
ANALYSIS_KEY = None
RSSI_OVERLAY_CACHE = {}


def node_signature(node):
    normalized = with_node_defaults(node)
    return "|".join(
        [
            str(normalized["id"]),
            f"{float(normalized['longitude']):.6f}",
            f"{float(normalized['latitude']):.6f}",
            f"{float(normalized['height_agl_m']):.3f}",
            f"{float(normalized['antenna_gain_dbi']):.3f}",
            f"{float(normalized['tx_power_dbm']):.3f}",
        ]
    )


def fetch_image_href(meta_url):
    response = requests.get(meta_url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    href = payload.get("href")
    if not href:
        raise ValueError("Image service response did not include an href.")
    return href


def fetch_binary(url):
    response = requests.get(url, timeout=240)
    response.raise_for_status()
    return response.content


@contextmanager
def open_remote_raster(url):
    data = fetch_binary(url)
    with MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            yield dataset


def worldcover_tile_code(lat_start, lon_start):
    lat_prefix = "N" if lat_start >= 0 else "S"
    lon_prefix = "E" if lon_start >= 0 else "W"
    return f"{lat_prefix}{abs(lat_start):02d}{lon_prefix}{abs(lon_start):03d}"


def worldcover_tile_urls(min_lon, min_lat, max_lon, max_lat):
    eps = 1e-9
    lon_starts = range(
        math.floor(min_lon / 3.0) * 3,
        math.floor((max_lon - eps) / 3.0) * 3 + 3,
        3,
    )
    lat_starts = range(
        math.floor(min_lat / 3.0) * 3,
        math.floor((max_lat - eps) / 3.0) * 3 + 3,
        3,
    )

    urls = []
    for lat_start in lat_starts:
        for lon_start in lon_starts:
            tile_code = worldcover_tile_code(lat_start, lon_start)
            urls.append(
                f"{WORLDCOVER_BASE_URL}/{WORLDCOVER_VERSION}/{WORLDCOVER_YEAR}/map/"
                f"ESA_WorldCover_10m_{WORLDCOVER_YEAR}_{WORLDCOVER_VERSION}_{tile_code}_Map.tif"
            )
    return urls


def raster_axes(transform_affine, rows, cols):
    x_axis = np.array(
        rasterio.transform.xy(
            transform_affine,
            np.zeros(cols, dtype=int),
            np.arange(cols),
            offset="center",
        )[0]
    )
    y_axis = np.array(
        rasterio.transform.xy(
            transform_affine,
            np.arange(rows),
            np.zeros(rows, dtype=int),
            offset="center",
        )[1]
    )
    return x_axis, y_axis


def normalize_bbox(min_lon, min_lat, max_lon, max_lat):
    values = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("Bounding box min values must be smaller than max values.")
    return tuple(round(value, 6) for value in values)


def ranges_from_relayout_data(relayout_data, fallback_x_range, fallback_y_range):
    if not relayout_data:
        return tuple(fallback_x_range), tuple(fallback_y_range)

    if relayout_data.get("xaxis.autorange") or relayout_data.get("yaxis.autorange"):
        return tuple(fallback_x_range), tuple(fallback_y_range)

    x_range = relayout_data.get("xaxis.range")
    y_range = relayout_data.get("yaxis.range")

    if x_range is None:
        x0 = relayout_data.get("xaxis.range[0]")
        x1 = relayout_data.get("xaxis.range[1]")
        if x0 is not None and x1 is not None:
            x_range = [x0, x1]

    if y_range is None:
        y0 = relayout_data.get("yaxis.range[0]")
        y1 = relayout_data.get("yaxis.range[1]")
        if y0 is not None and y1 is not None:
            y_range = [y0, y1]

    if not x_range or not y_range:
        return tuple(fallback_x_range), tuple(fallback_y_range)

    return (
        (float(min(x_range)), float(max(x_range))),
        (float(min(y_range)), float(max(y_range))),
    )


@lru_cache(maxsize=4)
def get_map_bundle(min_lon, min_lat, max_lon, max_lat):
    min_x, min_y, max_x, max_y = transform_bounds(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )

    display_terrain_meta_url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        f"?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&bboxSR={GEOGRAPHIC_CRS.split(':')[-1]}"
        f"&imageSR={GEOGRAPHIC_CRS.split(':')[-1]}"
        f"&size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"
        "&format=tiff"
        "&pixelType=F32"
        "&interpolation=RSP_BilinearInterpolation"
        "&f=json"
    )
    display_terrain_href = fetch_image_href(display_terrain_meta_url)

    imagery_meta_url = (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
        f"?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&bboxSR={GEOGRAPHIC_CRS.split(':')[-1]}"
        f"&imageSR={GEOGRAPHIC_CRS.split(':')[-1]}"
        f"&size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"
        "&format=tiff"
        "&pixelType=U8"
        "&interpolation=RSP_BilinearInterpolation"
        "&f=json"
    )
    imagery_href = fetch_image_href(imagery_meta_url)

    projected_terrain_meta_url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        f"?bbox={min_x},{min_y},{max_x},{max_y}"
        f"&bboxSR={PROJECTED_CRS.split(':')[-1]}"
        f"&imageSR={PROJECTED_CRS.split(':')[-1]}"
        f"&size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"
        "&format=tiff"
        "&pixelType=F32"
        "&interpolation=RSP_BilinearInterpolation"
        "&f=json"
    )
    projected_terrain_href = fetch_image_href(projected_terrain_meta_url)

    with open_remote_raster(imagery_href) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        lon_axis, lat_axis = raster_axes(src.transform, src.height, src.width)

    with open_remote_raster(display_terrain_href) as src:
        terrain_display = src.read(1).astype(np.float32)
        terrain_display_lon_axis, terrain_display_lat_axis = raster_axes(src.transform, src.height, src.width)
        terrain_display_transform = src.transform
        terrain_display_shape = (src.height, src.width)

    with open_remote_raster(projected_terrain_href) as src:
        terrain_band = src.read(1).astype(np.float32)
        projected_transform = src.transform
        projected_shape = (src.height, src.width)
        terrain_x_axis, terrain_y_axis = raster_axes(projected_transform, src.height, src.width)

    worldcover_band = np.zeros_like(terrain_band, dtype=np.uint8)
    for worldcover_url in worldcover_tile_urls(min_lon, min_lat, max_lon, max_lat):
        with open_remote_raster(worldcover_url) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=worldcover_band,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=projected_transform,
                dst_crs=PROJECTED_CRS,
                dst_nodata=0,
                resampling=Resampling.nearest,
                init_dest_nodata=False,
            )

    terrain_projected = xr.DataArray(
        terrain_band,
        dims=("y", "x"),
        coords={"y": terrain_y_axis, "x": terrain_x_axis},
        name="elevation",
    ).sortby("y", ascending=False)

    worldcover_projected = xr.DataArray(
        worldcover_band,
        dims=("y", "x"),
        coords={"y": terrain_y_axis, "x": terrain_x_axis},
        name="worldcover",
    ).sortby("y", ascending=False)

    return {
        "cache_key": (min_lon, min_lat, max_lon, max_lat),
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
        "rgb": rgb,
        "lon_axis": lon_axis.astype(np.float64),
        "lat_axis": lat_axis.astype(np.float64),
        "terrain_display": terrain_display,
        "terrain_display_lon_axis": terrain_display_lon_axis.astype(np.float64),
        "terrain_display_lat_axis": terrain_display_lat_axis.astype(np.float64),
        "terrain_display_transform": terrain_display_transform,
        "terrain_display_shape": terrain_display_shape,
        "terrain_projected": terrain_projected,
        "worldcover_projected": worldcover_projected,
        "projected_transform": projected_transform,
        "projected_shape": projected_shape,
    }


def ensure_analysis_context(bundle):
    global ANALYSIS_KEY

    bundle_key = bundle["cache_key"]
    if ANALYSIS_KEY == bundle_key:
        return

    terrain_projected = bundle["terrain_projected"]
    worldcover_projected = bundle["worldcover_projected"]

    ANALYSIS_CONTEXT.clear()
    ANALYSIS_CONTEXT.update(
        {
            "terrain_da": terrain_projected,
            "worldcover_da": worldcover_projected,
            "terrain_values": terrain_projected.values.astype(np.float32),
            "worldcover_values": worldcover_projected.values.astype(np.int16),
            "terrain_x": terrain_projected.x.values.astype(np.float64),
            "terrain_y": terrain_projected.y.values.astype(np.float64),
        }
    )
    snap_point.cache_clear()
    _get_path_profile_by_index_cached.cache_clear()
    ANALYSIS_KEY = bundle_key


def dist3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def dist2d(p1, p2):
    x1, y1, _z1 = p1
    x2, y2, _z2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@lru_cache(maxsize=200000)
def snap_point(x_coord, y_coord):
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    terrain_values = ANALYSIS_CONTEXT["terrain_values"]
    col = int(np.abs(terrain_x - x_coord).argmin())
    row = int(np.abs(terrain_y - y_coord).argmin())
    return row, col, float(terrain_x[col]), float(terrain_y[row]), float(terrain_values[row, col])


@lru_cache(maxsize=200000)
def _get_path_profile_by_index_cached(row1, col1, row2, col2, observer_height=0.0, target_height=0.0, min_samples=2):
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    terrain_values = ANALYSIS_CONTEXT["terrain_values"]
    worldcover_values = ANALYSIS_CONTEXT["worldcover_values"]

    pos1 = (
        float(terrain_x[col1]),
        float(terrain_y[row1]),
        float(terrain_values[row1, col1] + observer_height),
    )
    pos2 = (
        float(terrain_x[col2]),
        float(terrain_y[row2]),
        float(terrain_values[row2, col2] + target_height),
    )

    dist = dist2d(pos1, pos2)
    sample_count = max(min_samples, int(np.ceil(0.01 * dist)))
    row_path = np.rint(np.linspace(row1, row2, sample_count)).astype(np.int32)
    col_path = np.rint(np.linspace(col1, col2, sample_count)).astype(np.int32)

    x_path = terrain_x[col_path]
    y_path = terrain_y[row_path]
    if sample_count > 1:
        step_distance_m = np.sqrt(np.diff(x_path) ** 2 + np.diff(y_path) ** 2)
        distance_along_m = np.concatenate(([0.0], np.cumsum(step_distance_m)))
        total_distance_m = max(distance_along_m[-1], 1e-6)
        path_fraction = distance_along_m / total_distance_m
    else:
        distance_along_m = np.array([0.0], dtype=np.float64)
        path_fraction = np.array([0.0], dtype=np.float64)

    line_z = pos1[2] + (pos2[2] - pos1[2]) * path_fraction
    line = np.column_stack((x_path, y_path, line_z))

    d1 = np.array([dist3d(pos1, point) for point in line], dtype=np.float64)
    d2 = np.array([dist3d(pos2, point) for point in line], dtype=np.float64)
    fresnel = np.sqrt((0.3304 * d1 * d2) / np.maximum(d1 + d2, 1e-6))

    ground_surface = worldcover_values[row_path, col_path]
    ground_surface_offset = V_OFFSET(ground_surface)
    terrain_only_blocked = terrain_values[row_path, col_path] >= line[:, 2]
    terrain_only_blocked[0] = False
    terrain_only_blocked[-1] = False

    obstruction_top = terrain_values[row_path, col_path] + ground_surface_offset
    clearance_height = line[:, 2] - fresnel * 0.6
    fresnel_zone_height = np.maximum(line[:, 2] - clearance_height, 1e-6)
    blockage_fraction = np.clip((obstruction_top - clearance_height) / fresnel_zone_height, 0.0, 1.0)
    blockage_fraction[0] = 0.0
    blockage_fraction[-1] = 0.0

    attenuation_values = V_ATTENUATION(ground_surface) * blockage_fraction
    attenuation_values = attenuation_values + 20.0 * terrain_only_blocked.astype(np.float32)

    return (
        float(np.sum(attenuation_values)),
        tuple(row_path.tolist()),
        tuple(col_path.tolist()),
        tuple(blockage_fraction.tolist()),
        tuple(terrain_only_blocked.tolist()),
    )


def unpack_path_profile(result):
    if len(result) == 5:
        return result
    if len(result) == 4:
        path_loss_db, row_path, col_path, blockage_fraction = result
        terrain_only_blocked = tuple(False for _ in row_path)
        return path_loss_db, row_path, col_path, blockage_fraction, terrain_only_blocked
    raise ValueError(f"Unexpected path profile result length: {len(result)}")


def reverse_path_profile(result):
    path_loss_db, row_path, col_path, blockage_fraction, terrain_only_blocked = unpack_path_profile(result)
    return (
        path_loss_db,
        tuple(reversed(row_path)),
        tuple(reversed(col_path)),
        tuple(reversed(blockage_fraction)),
        tuple(reversed(terrain_only_blocked)),
    )


def get_path_profile_by_index(row1, col1, row2, col2, observer_height=0.0, target_height=0.0, min_samples=2):
    reverse_requested = (row2, col2, row1, col1) < (row1, col1, row2, col2)
    if reverse_requested:
        result = _get_path_profile_by_index_cached(
            row2,
            col2,
            row1,
            col1,
            observer_height=target_height,
            target_height=observer_height,
            min_samples=min_samples,
        )
        return reverse_path_profile(result)
    return _get_path_profile_by_index_cached(
        row1,
        col1,
        row2,
        col2,
        observer_height=observer_height,
        target_height=target_height,
        min_samples=min_samples,
    )


def get_path_losses_by_index(row1, col1, row2, col2, observer_height=0.0, target_height=0.0, min_samples=2):
    loss_db, _row_path, _col_path, _blockage_fraction, _terrain_only_blocked = unpack_path_profile(
        get_path_profile_by_index(
            row1,
            col1,
            row2,
            col2,
            observer_height=observer_height,
            target_height=target_height,
            min_samples=min_samples,
        )
    )
    return loss_db


def compute_ground_loss_for_chunk(args):
    observer_row, observer_col, cells, observer_height, target_height = args
    rows = []
    cols = []
    losses = []
    for row, col in cells:
        path_loss_db = get_path_losses_by_index(
            observer_row,
            observer_col,
            int(row),
            int(col),
            observer_height=observer_height,
            target_height=target_height,
        )
        rows.append(int(row))
        cols.append(int(col))
        losses.append(path_loss_db)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(losses, dtype=np.float32),
    )


def fspl_db(distance_km, freq_mhz):
    distance_km = np.asarray(distance_km, dtype=np.float64)
    return 32.44 + 20 * np.log10(np.maximum(distance_km, 1e-6)) + 20 * np.log10(freq_mhz)


def received_power_dbm(
    distance_km,
    freq_mhz,
    tx_power_dbm,
    tx_gain_dbi=0.0,
    rx_gain_dbi=0.0,
    other_losses_db=0.0,
):
    return (
        tx_power_dbm
        + tx_gain_dbi
        + rx_gain_dbi
        - fspl_db(distance_km, freq_mhz)
        - other_losses_db
    )


def compute_path_loss(
    c1,
    c2,
    tx_height=DEFAULT_NODE_HEIGHT_M,
    rx_height=DEFAULT_NODE_HEIGHT_M,
    freq_mhz=DEFAULT_FREQ_MHZ,
    tx_power_dbm=DEFAULT_TX_POWER_DBM,
    tx_gain_dbi=DEFAULT_TX_GAIN_DBI,
    rx_gain_dbi=DEFAULT_RX_GAIN_DBI,
    other_losses_db=DEFAULT_OTHER_LOSSES_DB,
    min_samples=2,
):
    lon1, lat1 = c1
    lon2, lat2 = c2
    projected_x, projected_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        [lon1, lon2],
        [lat1, lat2],
    )

    row1, col1, x1, y1, _z1_ground = snap_point(projected_x[0], projected_y[0])
    row2, col2, x2, y2, _z2_ground = snap_point(projected_x[1], projected_y[1])

    path_loss_db, row_path, col_path, blockage_fraction, terrain_only_blocked = unpack_path_profile(
        get_path_profile_by_index(
            row1,
            col1,
            row2,
            col2,
            observer_height=tx_height,
            target_height=rx_height,
            min_samples=min_samples,
        )
    )

    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    terrain_values = ANALYSIS_CONTEXT["terrain_values"]
    worldcover_values = ANALYSIS_CONTEXT["worldcover_values"]

    row_path = np.asarray(row_path, dtype=np.int32)
    col_path = np.asarray(col_path, dtype=np.int32)
    blockage_fraction = np.asarray(blockage_fraction, dtype=np.float32)
    terrain_only_blocked = np.asarray(terrain_only_blocked, dtype=bool)

    if row_path.size and (row_path[0] != row1 or col_path[0] != col1):
        row_path = row_path[::-1]
        col_path = col_path[::-1]
        blockage_fraction = blockage_fraction[::-1]
        terrain_only_blocked = terrain_only_blocked[::-1]

    x_path = terrain_x[col_path]
    y_path = terrain_y[row_path]
    path_longitude, path_latitude = transform(
        PROJECTED_CRS,
        GEOGRAPHIC_CRS,
        x_path.tolist(),
        y_path.tolist(),
    )
    terrain_profile = terrain_values[row_path, col_path]
    worldcover_profile = worldcover_values[row_path, col_path]
    clutter_heights = V_OFFSET(worldcover_profile)
    obstruction_top = terrain_profile + clutter_heights

    tx_z = terrain_profile[0] + tx_height
    rx_z = terrain_profile[-1] + rx_height

    if len(row_path) > 1:
        step_distance_m = np.sqrt(np.diff(x_path) ** 2 + np.diff(y_path) ** 2)
        distance_along_m = np.concatenate(([0.0], np.cumsum(step_distance_m)))
        total_distance_m = max(distance_along_m[-1], 1e-6)
        path_fraction = distance_along_m / total_distance_m
    else:
        distance_along_m = np.array([0.0], dtype=np.float64)
        path_fraction = np.array([0.0], dtype=np.float64)

    los_line = tx_z + (rx_z - tx_z) * path_fraction
    line_points = np.column_stack((x_path, y_path, los_line))
    p1 = (x1, y1, tx_z)
    p2 = (x2, y2, rx_z)

    d1 = np.array([dist3d(p1, point) for point in line_points], dtype=np.float64)
    d2 = np.array([dist3d(p2, point) for point in line_points], dtype=np.float64)
    fresnel_radius = np.sqrt((0.3304 * d1 * d2) / np.maximum(d1 + d2, 1e-6))
    fresnel_60 = 0.6 * fresnel_radius
    fresnel_upper = los_line + fresnel_60
    fresnel_lower = los_line - fresnel_60

    attenuation_per_sample = V_ATTENUATION(worldcover_profile) * blockage_fraction
    attenuation_per_sample = attenuation_per_sample + 20.0 * terrain_only_blocked.astype(np.float32)

    link_distance_km = max(dist3d(p1, p2) / 1000.0, 1e-6)
    rssi_dbm = received_power_dbm(
        link_distance_km,
        freq_mhz,
        tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        other_losses_db=other_losses_db + path_loss_db,
    )

    return {
        "rssi_dbm": float(rssi_dbm),
        "attenuation_event_count": int(np.count_nonzero(blockage_fraction > 0)),
        "terrain_collision_event_count": int(np.count_nonzero(terrain_only_blocked)),
        "direct_los_collision_count": int(np.count_nonzero(obstruction_top >= los_line)),
        "path_loss_db": float(path_loss_db),
        "link_distance_km": float(link_distance_km),
        "distance_along_km": distance_along_m / 1000.0,
        "terrain_profile_m": terrain_profile,
        "obstruction_top_m": obstruction_top,
        "los_line_m": los_line,
        "fresnel_upper_m": fresnel_upper,
        "fresnel_lower_m": fresnel_lower,
        "blockage_fraction": blockage_fraction,
        "attenuation_per_sample_db": attenuation_per_sample,
        "terrain_only_blocked": terrain_only_blocked,
        "worldcover_classes": worldcover_profile,
        "direct_los_hits": obstruction_top >= los_line,
        "path_longitude": np.asarray(path_longitude, dtype=np.float64),
        "path_latitude": np.asarray(path_latitude, dtype=np.float64),
    }


def node_color(index):
    return COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]


def node_color_map(nodes):
    return {
        str(node["id"]): node_color(index)
        for index, node in enumerate(with_node_defaults(node) for node in (nodes or []))
    }


def with_node_defaults(node):
    normalized = dict(node)
    normalized["height_agl_m"] = float(normalized.get("height_agl_m", DEFAULT_NODE_HEIGHT_M))
    normalized["antenna_gain_dbi"] = float(normalized.get("antenna_gain_dbi", DEFAULT_TX_GAIN_DBI))
    normalized["tx_power_dbm"] = float(normalized.get("tx_power_dbm", DEFAULT_TX_POWER_DBM))
    return normalized


def link_key(source_id, target_id):
    ordered = sorted((str(source_id), str(target_id)))
    return f"{ordered[0]}::{ordered[1]}"


def find_node(nodes, node_id):
    for node in nodes:
        if str(node["id"]) == str(node_id):
            return node
    return None


def normalize_selected_node_ids(selected_node_ids, valid_ids=None):
    valid_set = {str(node_id) for node_id in valid_ids} if valid_ids is not None else None
    normalized = []
    for value in selected_node_ids or []:
        node_id = str(value)
        if valid_set is not None and node_id not in valid_set:
            continue
        if node_id in normalized:
            normalized.remove(node_id)
        normalized.append(node_id)
    return normalized[-2:]


def toggle_selected_node(selected_node_ids, node_id, valid_ids=None):
    node_id = str(node_id)
    valid_set = {str(value) for value in valid_ids} if valid_ids is not None else None
    if valid_set is not None and node_id not in valid_set:
        return normalize_selected_node_ids(selected_node_ids, valid_ids)

    selected = normalize_selected_node_ids(selected_node_ids, valid_ids)
    if node_id in selected:
        selected = [value for value in selected if value != node_id]
    else:
        selected.append(node_id)
    return normalize_selected_node_ids(selected, valid_ids)


def clicked_node_id_from_click_data(click_data):
    if not click_data:
        return None
    for point in click_data.get("points", []):
        custom = point.get("customdata")
        if isinstance(custom, np.ndarray):
            if custom.ndim == 0:
                continue
            custom = custom.tolist()
        if isinstance(custom, (list, tuple)) and len(custom) >= 3 and custom[0] == "node":
            return str(custom[2])
    return None


def select_primary_node(selected_node_ids, node_id):
    selected = normalize_selected_node_ids(selected_node_ids)
    previous_primary = selected[-1] if selected else None
    node_id = str(node_id)
    if previous_primary == node_id:
        return [node_id]
    if previous_primary is None:
        return [node_id]
    return [previous_primary, node_id]


def node_marker_outline_arrays(nodes, selected_node_ids):
    selected = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in nodes])
    primary_id = selected[-1] if selected else None
    colors = []
    widths = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id == primary_id:
            colors.append("#1b7f3a")
            widths.append(3.0)
        elif node_id in selected:
            colors.append("#d2b100")
            widths.append(3.0)
        else:
            colors.append("white")
            widths.append(1.5)
    return colors, widths


def build_observer_frame(nodes):
    if not nodes:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "longitude",
                "latitude",
                "observer_elev",
                "antenna_gain_dbi",
                "tx_power_dbm",
                "x",
                "y",
            ]
        )

    observer_frame = pd.DataFrame(with_node_defaults(node) for node in nodes)
    observer_frame["observer_elev"] = observer_frame["height_agl_m"].astype(float)
    projected_x, projected_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        observer_frame["longitude"].tolist(),
        observer_frame["latitude"].tolist(),
    )
    observer_frame["x"] = projected_x
    observer_frame["y"] = projected_y
    return observer_frame


def compute_bidirectional_link_result(source, target):
    forward = compute_path_loss(
        (source["longitude"], source["latitude"]),
        (target["longitude"], target["latitude"]),
        tx_height=source["height_agl_m"],
        rx_height=target["height_agl_m"],
        tx_power_dbm=source["tx_power_dbm"],
        tx_gain_dbi=source["antenna_gain_dbi"],
        rx_gain_dbi=target["antenna_gain_dbi"],
    )
    reverse = compute_path_loss(
        (target["longitude"], target["latitude"]),
        (source["longitude"], source["latitude"]),
        tx_height=target["height_agl_m"],
        rx_height=source["height_agl_m"],
        tx_power_dbm=target["tx_power_dbm"],
        tx_gain_dbi=target["antenna_gain_dbi"],
        rx_gain_dbi=source["antenna_gain_dbi"],
    )
    link_rssi = min(forward["rssi_dbm"], reverse["rssi_dbm"])
    return {
        "source_id": source["id"],
        "target_id": target["id"],
        "source_name": source["name"],
        "target_name": target["name"],
        "rssi_dbm": link_rssi,
        "forward_rssi_dbm": forward["rssi_dbm"],
        "reverse_rssi_dbm": reverse["rssi_dbm"],
        "path_loss_db": forward["path_loss_db"],
        "distance_km": forward["link_distance_km"],
        "good_link": link_rssi > MIN_LINK_RSSI_DBM,
        "forward_result": forward,
        "reverse_result": reverse,
    }


def compute_link_results(nodes, bundle):
    if len(nodes) < 2:
        return []

    ensure_analysis_context(bundle)
    normalized_nodes = [with_node_defaults(node) for node in nodes]
    link_results = []
    for source_index, target_index in combinations(range(len(normalized_nodes)), 2):
        source = normalized_nodes[source_index]
        target = normalized_nodes[target_index]
        pair_result = compute_bidirectional_link_result(source, target)
        link_results.append(
            {
                "id": link_key(source["id"], target["id"]),
                "source_id": pair_result["source_id"],
                "target_id": pair_result["target_id"],
                "source_name": pair_result["source_name"],
                "target_name": pair_result["target_name"],
                "source_index": source_index,
                "target_index": target_index,
                "rssi_dbm": pair_result["rssi_dbm"],
                "forward_rssi_dbm": pair_result["forward_rssi_dbm"],
                "reverse_rssi_dbm": pair_result["reverse_rssi_dbm"],
                "path_loss_db": pair_result["path_loss_db"],
                "distance_km": pair_result["distance_km"],
                "good_link": pair_result["good_link"],
            }
        )
    return link_results


def compute_node_rssi_summaries(nodes, bundle, global_rx_height_agl_m, global_rx_gain_dbi):
    if not nodes:
        return {}

    ensure_analysis_context(bundle)
    observer_frame = build_observer_frame(nodes)
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    x_grid, y_grid = np.meshgrid(terrain_da.x.values, terrain_da.y.values)

    x_step = float(abs(terrain_da.x.values[1] - terrain_da.x.values[0])) if terrain_da.x.size > 1 else 0.0
    y_step = float(abs(terrain_da.y.values[1] - terrain_da.y.values[0])) if terrain_da.y.size > 1 else 0.0
    cell_area_km2 = (x_step * y_step) / 1_000_000.0 if x_step and y_step else 0.0

    summaries = {}
    for observer in observer_frame.itertuples(index=False):
        vs = viewshed(
            terrain_da,
            x=observer.x,
            y=observer.y,
            observer_elev=observer.observer_elev,
            target_elev=float(global_rx_height_agl_m),
        )
        visible_mask = np.asarray(vs > 0)
        observer_row, observer_col, _obs_x, _obs_y, _obs_z = snap_point(observer.x, observer.y)

        distance_km = np.sqrt((x_grid - observer.x) ** 2 + (y_grid - observer.y) ** 2) / 1000.0
        distance_km = np.maximum(distance_km, 1e-6)
        ground_loss_db = np.zeros(terrain_da.shape, dtype=np.float32)

        visible_indices = np.argwhere(visible_mask)
        cells = [(int(row), int(col)) for row, col in visible_indices]
        if cells:
            chunk_rows, chunk_cols, chunk_losses = compute_ground_loss_for_chunk(
                (observer_row, observer_col, cells, float(observer.observer_elev), float(global_rx_height_agl_m))
            )
            ground_loss_db[chunk_rows, chunk_cols] = chunk_losses

        observer_rssi = np.where(
            visible_mask,
            received_power_dbm(
                distance_km,
                DEFAULT_FREQ_MHZ,
                float(observer.tx_power_dbm),
                tx_gain_dbi=float(observer.antenna_gain_dbi),
                rx_gain_dbi=float(global_rx_gain_dbi),
                other_losses_db=DEFAULT_OTHER_LOSSES_DB + ground_loss_db,
            ),
            np.nan,
        )

        observer_rssi = np.where(distance_km > 0.05, observer_rssi, np.nan)
        valid = observer_rssi[np.isfinite(observer_rssi)]
        valid_above_floor = valid[valid >= MIN_LINK_RSSI_DBM]
        summaries[str(observer.id)] = {
            "coverage_area_km2": float(np.count_nonzero(np.isfinite(observer_rssi)) * cell_area_km2),
            "peak_rssi_dbm": float(np.nanmax(valid_above_floor)) if valid_above_floor.size else None,
            "p95_rssi_dbm": float(np.nanpercentile(valid_above_floor, 95)) if valid_above_floor.size else None,
        }

    return summaries


def compute_rssi_overlay(nodes, bundle, include_ground_loss, global_rx_height_agl_m, global_rx_gain_dbi):
    if not nodes:
        return None

    normalized_nodes = [with_node_defaults(node) for node in nodes]
    cache_key = overlay_cache_key(
        bundle,
        normalized_nodes,
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
    )
    if cache_key in RSSI_OVERLAY_CACHE:
        return RSSI_OVERLAY_CACHE[cache_key]

    ensure_analysis_context(bundle)
    observer_frame = build_observer_frame(normalized_nodes)
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    x_grid, y_grid = np.meshgrid(terrain_da.x.values, terrain_da.y.values)
    max_rssi = np.full(terrain_da.shape, np.nan, dtype=np.float32)

    for observer in observer_frame.itertuples(index=False):
        vs = viewshed(
            terrain_da,
            x=observer.x,
            y=observer.y,
            observer_elev=observer.observer_elev,
            target_elev=float(global_rx_height_agl_m),
        )
        visible_mask = np.asarray(vs > 0)
        distance_km = np.sqrt((x_grid - observer.x) ** 2 + (y_grid - observer.y) ** 2) / 1000.0
        distance_km = np.maximum(distance_km, 1e-6)
        ground_loss_db = np.zeros(terrain_da.shape, dtype=np.float32)

        if include_ground_loss:
            observer_row, observer_col, _obs_x, _obs_y, _obs_z = snap_point(observer.x, observer.y)
            visible_indices = np.argwhere(visible_mask)
            cells = [(int(row), int(col)) for row, col in visible_indices]
            if cells:
                chunk_rows, chunk_cols, chunk_losses = compute_ground_loss_for_chunk(
                    (observer_row, observer_col, cells, float(observer.observer_elev), float(global_rx_height_agl_m))
                )
                ground_loss_db[chunk_rows, chunk_cols] = chunk_losses

        observer_rssi = np.where(
            visible_mask,
            received_power_dbm(
                distance_km,
                DEFAULT_FREQ_MHZ,
                float(observer.tx_power_dbm),
                tx_gain_dbi=float(observer.antenna_gain_dbi),
                rx_gain_dbi=float(global_rx_gain_dbi),
                other_losses_db=DEFAULT_OTHER_LOSSES_DB + ground_loss_db,
            ),
            np.nan,
        )
        observer_rssi = np.where(observer_rssi >= MIN_LINK_RSSI_DBM, observer_rssi, np.nan)
        max_rssi = np.fmax(max_rssi, observer_rssi.astype(np.float32))

    display_rssi = np.full(bundle["terrain_display_shape"], np.nan, dtype=np.float32)
    reproject(
        source=max_rssi,
        destination=display_rssi,
        src_transform=bundle["projected_transform"],
        src_crs=PROJECTED_CRS,
        src_nodata=np.nan,
        dst_transform=bundle["terrain_display_transform"],
        dst_crs=GEOGRAPHIC_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    result = {
        "max_rssi": np.flipud(display_rssi),
        "lon_axis": bundle["terrain_display_lon_axis"],
        "lat_axis": bundle["terrain_display_lat_axis"][::-1],
    }
    RSSI_OVERLAY_CACHE[cache_key] = result
    return result


def compute_single_node_rssi_overlay(node, bundle, include_ground_loss, global_rx_height_agl_m, global_rx_gain_dbi):
    normalized = with_node_defaults(node)
    cache_key = overlay_cache_key(
        bundle,
        [normalized],
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
    )
    if cache_key in RSSI_OVERLAY_CACHE:
        return cache_key, RSSI_OVERLAY_CACHE[cache_key]

    ensure_analysis_context(bundle)
    observer_frame = build_observer_frame([normalized])
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    x_grid, y_grid = np.meshgrid(terrain_da.x.values, terrain_da.y.values)
    observer = next(observer_frame.itertuples(index=False))

    vs = viewshed(
        terrain_da,
        x=observer.x,
        y=observer.y,
        observer_elev=observer.observer_elev,
        target_elev=float(global_rx_height_agl_m),
    )
    visible_mask = np.asarray(vs > 0)
    distance_km = np.sqrt((x_grid - observer.x) ** 2 + (y_grid - observer.y) ** 2) / 1000.0
    distance_km = np.maximum(distance_km, 1e-6)
    ground_loss_db = np.zeros(terrain_da.shape, dtype=np.float32)

    if include_ground_loss:
        observer_row, observer_col, _obs_x, _obs_y, _obs_z = snap_point(observer.x, observer.y)
        visible_indices = np.argwhere(visible_mask)
        cells = [(int(row), int(col)) for row, col in visible_indices]
        if cells:
            chunk_rows, chunk_cols, chunk_losses = compute_ground_loss_for_chunk(
                (observer_row, observer_col, cells, float(observer.observer_elev), float(global_rx_height_agl_m))
            )
            ground_loss_db[chunk_rows, chunk_cols] = chunk_losses

    observer_rssi = np.where(
        visible_mask,
        received_power_dbm(
            distance_km,
            DEFAULT_FREQ_MHZ,
            float(observer.tx_power_dbm),
            tx_gain_dbi=float(observer.antenna_gain_dbi),
            rx_gain_dbi=float(global_rx_gain_dbi),
            other_losses_db=DEFAULT_OTHER_LOSSES_DB + ground_loss_db,
        ),
        np.nan,
    )
    observer_rssi = np.where(observer_rssi >= MIN_LINK_RSSI_DBM, observer_rssi, np.nan)

    display_rssi = np.full(bundle["terrain_display_shape"], np.nan, dtype=np.float32)
    reproject(
        source=observer_rssi.astype(np.float32),
        destination=display_rssi,
        src_transform=bundle["projected_transform"],
        src_crs=PROJECTED_CRS,
        src_nodata=np.nan,
        dst_transform=bundle["terrain_display_transform"],
        dst_crs=GEOGRAPHIC_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    result = {
        "max_rssi": np.flipud(display_rssi),
        "lon_axis": bundle["terrain_display_lon_axis"],
        "lat_axis": bundle["terrain_display_lat_axis"][::-1],
    }
    RSSI_OVERLAY_CACHE[cache_key] = result
    return cache_key, result


def compose_cached_rssi_overlay(bundle, calculation_store, overlay_selection_store):
    if not calculation_store:
        return None
    if tuple(calculation_store.get("bundle_key", ())) != bundle["cache_key"]:
        return None

    node_keys = calculation_store.get("node_overlay_keys", {})
    enabled_nodes = {
        str(node_id)
        for node_id, enabled in (overlay_selection_store or {}).items()
        if bool(enabled) and str(node_id) in node_keys
    }
    if not enabled_nodes:
        return None

    combined = None
    lon_axis = None
    lat_axis = None
    for node_id in calculation_store.get("node_order", []):
        node_id = str(node_id)
        if node_id not in enabled_nodes:
            continue
        overlay = RSSI_OVERLAY_CACHE.get(node_keys[node_id])
        if overlay is None:
            continue
        if combined is None:
            combined = np.array(overlay["max_rssi"], copy=True)
            lon_axis = overlay["lon_axis"]
            lat_axis = overlay["lat_axis"]
        else:
            combined = np.fmax(combined, overlay["max_rssi"])

    if combined is None:
        return None

    return {
        "max_rssi": combined,
        "lon_axis": lon_axis,
        "lat_axis": lat_axis,
    }


def overlay_cache_key(bundle, nodes, include_ground_loss, global_rx_height_agl_m, global_rx_gain_dbi):
    normalized_nodes = [with_node_defaults(node) for node in nodes]
    parts = [
        f"{bundle['cache_key']}",
        str(bool(include_ground_loss)),
        f"{float(global_rx_height_agl_m):.3f}",
        f"{float(global_rx_gain_dbi):.3f}",
    ]
    for node in normalized_nodes:
        parts.append(
            "|".join(
                [
                    str(node["id"]),
                    f"{float(node['longitude']):.6f}",
                    f"{float(node['latitude']):.6f}",
                    f"{float(node['height_agl_m']):.3f}",
                    f"{float(node['antenna_gain_dbi']):.3f}",
                    f"{float(node['tx_power_dbm']):.3f}",
                ]
            )
        )
    return "::".join(parts)


def bounds_from_axes(x_axis, y_axis):
    x_step = abs(float(x_axis[1] - x_axis[0])) if len(x_axis) > 1 else 0.0
    y_step = abs(float(y_axis[1] - y_axis[0])) if len(y_axis) > 1 else 0.0
    x_min = float(np.min(x_axis) - x_step / 2)
    x_max = float(np.max(x_axis) + x_step / 2)
    y_min = float(np.min(y_axis) - y_step / 2)
    y_max = float(np.max(y_axis) + y_step / 2)
    return x_min, x_max, y_min, y_max


def build_map_figure(
    bundle,
    nodes,
    terrain_alpha,
    terrain_clip_range=None,
    elevation_colorscale="Magma",
    rssi_overlay=None,
    rssi_colorscale="Turbo",
    shapes=None,
    path_traces=None,
    rssi_opacity=0.55,
    selected_node_ids=None,
):
    lon_min, lon_max, lat_min, lat_max = bounds_from_axes(bundle["lon_axis"], bundle["lat_axis"])
    terrain_lon = bundle["terrain_display_lon_axis"]
    terrain_lat = bundle["terrain_display_lat_axis"][::-1]
    terrain_display = np.flipud(bundle["terrain_display"])
    terrain_valid = terrain_display[np.isfinite(terrain_display)]
    terrain_min = float(np.nanmin(terrain_valid)) if terrain_valid.size else 0.0
    terrain_max = float(np.nanmax(terrain_valid)) if terrain_valid.size else terrain_min + 1.0
    if terrain_max <= terrain_min:
        terrain_max = terrain_min + 1.0

    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])

    clip_min = terrain_min
    clip_max = terrain_max
    if terrain_clip_range and len(terrain_clip_range) == 2:
        clip_min = min(max(float(terrain_clip_range[0]), terrain_min), terrain_max)
        clip_max = min(max(float(terrain_clip_range[1]), clip_min), terrain_max)

    terrain_clipped = np.clip(terrain_display, clip_min, clip_max)

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=Image.fromarray(bundle["rgb"]),
            xref="x",
            yref="y",
            x=lon_min,
            y=lat_max,
            sizex=lon_max - lon_min,
            sizey=lat_max - lat_min,
            sizing="stretch",
            layer="below",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=terrain_clipped,
            x=terrain_lon,
            y=terrain_lat,
            customdata=terrain_display,
            colorscale=elevation_colorscale,
            opacity=max(float(terrain_alpha), 0.001),
            zmin=clip_min,
            zmax=clip_max,
            colorbar={"title": "Elevation (m)", "x": -0.14, "y": 0.5, "len": 0.84},
            showscale=terrain_alpha > 0,
            zsmooth=False,
            hovertemplate=(
                "Longitude=%{x:.4f}<br>"
                "Latitude=%{y:.4f}<br>"
                "Elevation=%{customdata:.1f} m<extra></extra>"
            ),
            name="terrain",
        )
    )

    if rssi_overlay is not None:
        fig.add_trace(
            go.Heatmap(
                z=rssi_overlay["max_rssi"],
                x=rssi_overlay["lon_axis"],
                y=rssi_overlay["lat_axis"],
                colorscale=rssi_colorscale,
                opacity=max(float(rssi_opacity), 0.001),
                zsmooth=False,
                zmin=-140,
                zmax=-60,
                colorbar={"title": "Max RSSI (dBm)", "x": 1.08, "y": 0.5, "len": 0.84},
                hovertemplate=(
                    "Longitude=%{x:.4f}<br>"
                    "Latitude=%{y:.4f}<br>"
                    "Max RSSI=%{z:.1f} dBm<extra></extra>"
                ),
                name="max-rssi",
            )
        )

    for trace in path_traces or []:
        fig.add_trace(trace)

    if normalized_nodes:
        node_colors = [node_color(index) for index, _node in enumerate(normalized_nodes)]
        node_customdata = [["node", node["name"], str(node["id"])] for node in normalized_nodes]
        line_colors, line_widths = node_marker_outline_arrays(normalized_nodes, selected_node_ids)
        fig.add_trace(
            go.Scatter(
                x=[node["longitude"] for node in normalized_nodes],
                y=[node["latitude"] for node in normalized_nodes],
                mode="markers",
                marker={
                    "size": 12,
                    "color": node_colors,
                    "line": {"color": line_colors, "width": line_widths},
                    "opacity": 1,
                },
                selected={"marker": {"opacity": 1}},
                unselected={"marker": {"opacity": 1}},
                customdata=node_customdata,
                hovertemplate="<b>%{customdata[1]}</b><br>Longitude=%{x:.5f}<br>Latitude=%{y:.5f}<extra></extra>",
                showlegend=False,
                name="nodes",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[node["longitude"] for node in normalized_nodes],
                y=[node["latitude"] for node in normalized_nodes],
                mode="markers",
                marker={
                    "size": 26,
                    "color": "rgba(255,255,255,0.001)",
                    "line": {"color": "rgba(255,255,255,0)", "width": 0},
                },
                customdata=node_customdata,
                hoverinfo="skip",
                showlegend=False,
                name="node-hitbox",
            )
        )
        for index, node in enumerate(normalized_nodes):
            fig.add_annotation(
                x=node["longitude"],
                y=node["latitude"],
                text=node["name"],
                showarrow=False,
                yshift=16,
                bgcolor="rgba(17,24,39,0.92)",
                bordercolor=node_color(index),
                borderpad=3,
                font={"size": 11, "color": "#f9fafb", "family": "Open Sans, sans-serif"},
            )

    fig.update_layout(
        height=720,
        shapes=shapes or [],
        margin={
            "l": 120 if terrain_alpha > 0 else 20,
            "r": 120 if rssi_overlay is not None else 70,
            "t": 10,
            "b": 10,
        },
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"family": "Open Sans, sans-serif", "color": "#e5e7eb"},
        clickmode="event",
        dragmode="pan",
        uirevision=f"{bundle['cache_key']}-map-{len(nodes or [])}",
    )
    fig.update_xaxes(
        title="Longitude",
        tickformat=".4f",
        range=[lon_min, lon_max],
        showgrid=False,
        zeroline=False,
        constrain="domain",
        color="#e5e7eb",
    )
    fig.update_yaxes(
        title="Latitude",
        tickformat=".4f",
        range=[lat_min, lat_max],
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        color="#e5e7eb",
    )
    return fig


def build_map_path_event_traces(result):
    path_longitude = np.asarray(result.get("path_longitude", []), dtype=np.float64)
    path_latitude = np.asarray(result.get("path_latitude", []), dtype=np.float64)
    blockage_fraction = np.asarray(result.get("blockage_fraction", []), dtype=np.float64)
    attenuation_per_sample_db = np.asarray(result.get("attenuation_per_sample_db", []), dtype=np.float64)
    terrain_only_blocked = np.asarray(result.get("terrain_only_blocked", []), dtype=bool)

    traces = []
    if path_longitude.size == 0 or path_latitude.size == 0:
        return traces

    attenuation_mask = (blockage_fraction > 0) & ~terrain_only_blocked
    if np.any(attenuation_mask):
        attenuation_customdata = np.column_stack(
            (
                blockage_fraction[attenuation_mask],
                attenuation_per_sample_db[attenuation_mask],
            )
        )
        traces.append(
            go.Scatter(
                x=path_longitude[attenuation_mask],
                y=path_latitude[attenuation_mask],
                mode="markers",
                marker={
                    "size": np.clip(8 + 10 * blockage_fraction[attenuation_mask], 8, 18),
                    "color": "#f97316",
                    "opacity": 0.88,
                    "line": {"color": "#7c2d12", "width": 1.0},
                    "symbol": "circle",
                },
                customdata=attenuation_customdata,
                hovertemplate=(
                    "Attenuation event<br>"
                    "Longitude=%{x:.5f}<br>"
                    "Latitude=%{y:.5f}<br>"
                    "Blocked fraction=%{customdata[0]:.2f}<br>"
                    "Added loss=%{customdata[1]:.2f} dB<extra></extra>"
                ),
                showlegend=False,
                name="attenuation-events",
            )
        )

    if np.any(terrain_only_blocked):
        terrain_customdata = attenuation_per_sample_db[terrain_only_blocked][:, None]
        traces.append(
            go.Scatter(
                x=path_longitude[terrain_only_blocked],
                y=path_latitude[terrain_only_blocked],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "#a855f7",
                    "opacity": 0.95,
                    "line": {"color": "#f5d0fe", "width": 1.2},
                    "symbol": "diamond",
                },
                customdata=terrain_customdata,
                hovertemplate=(
                    "Terrain LOS block<br>"
                    "Longitude=%{x:.5f}<br>"
                    "Latitude=%{y:.5f}<br>"
                    "Added loss=%{customdata[0]:.2f} dB<extra></extra>"
                ),
                showlegend=False,
                name="terrain-los-blocks",
            )
        )

    return traces


def compute_map_path_overlay(nodes, selected_node_ids, point_path_data, bbox_data, global_rx_height_agl, global_rx_gain_dbi):
    nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in nodes])

    def line_shape(x0, y0, x1, y1, good_link):
        return {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "line": {
                "color": "#1b7f3a" if good_link else "#c23b22",
                "width": 4,
                "dash": "solid" if good_link else "dash",
            },
        }

    def overlay_from_result(result, x0, y0, x1, y1, good_link):
        return {
            "shapes": [line_shape(x0, y0, x1, y1, good_link)],
            "traces": build_map_path_event_traces(result),
        }

    if len(selected_node_ids) == 2:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox)
        ensure_analysis_context(bundle)
        source = find_node(nodes, selected_node_ids[0])
        target = find_node(nodes, selected_node_ids[1])
        if source is None or target is None:
            return {"shapes": [], "traces": []}
        link_result = compute_bidirectional_link_result(source, target)
        return overlay_from_result(
            link_result["forward_result"],
            source["longitude"],
            source["latitude"],
            target["longitude"],
            target["latitude"],
            link_result["good_link"],
        )

    if point_path_data:
        source = find_node(nodes, point_path_data.get("source_node_id"))
        target_lon = point_path_data.get("target_longitude")
        target_lat = point_path_data.get("target_latitude")
        if source is None or target_lon is None or target_lat is None:
            return {"shapes": [], "traces": []}
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox)
        ensure_analysis_context(bundle)
        result = compute_path_loss(
            (source["longitude"], source["latitude"]),
            (float(target_lon), float(target_lat)),
            tx_height=source["height_agl_m"],
            rx_height=float(global_rx_height_agl or DEFAULT_NODE_HEIGHT_M),
            tx_power_dbm=source["tx_power_dbm"],
            tx_gain_dbi=source["antenna_gain_dbi"],
            rx_gain_dbi=float(global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI),
        )
        return overlay_from_result(
            result,
            source["longitude"],
            source["latitude"],
            float(target_lon),
            float(target_lat),
            result["rssi_dbm"] > MIN_LINK_RSSI_DBM,
        )

    return {"shapes": [], "traces": []}


def empty_path_profile_figure(message="Select two nodes or use Draw Path To Map Point to inspect a path profile."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#e5e7eb", "family": "Open Sans, sans-serif"},
    )
    fig.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"family": "Open Sans, sans-serif", "color": "#e5e7eb"},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def build_path_profile_figure(source_name, target_name, result, source_color="#22c55e", target_color="#f59e0b"):
    distance = np.asarray(result["distance_along_km"], dtype=np.float64)
    terrain_profile = np.asarray(result["terrain_profile_m"], dtype=np.float64)
    obstruction_top = np.asarray(result["obstruction_top_m"], dtype=np.float64)
    los_line = np.asarray(result["los_line_m"], dtype=np.float64)
    fresnel_upper = np.asarray(result["fresnel_upper_m"], dtype=np.float64)
    fresnel_lower = np.asarray(result["fresnel_lower_m"], dtype=np.float64)
    blockage_fraction = np.asarray(result["blockage_fraction"], dtype=np.float64)
    terrain_only_blocked = np.asarray(result["terrain_only_blocked"], dtype=bool)
    direct_los_hits = np.asarray(result["direct_los_hits"], dtype=bool)
    attenuation_events = (blockage_fraction > 0) & ~terrain_only_blocked
    direct_los_hits = direct_los_hits & ~terrain_only_blocked

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=terrain_profile,
            mode="lines",
            line={"color": "#5b341c", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(122,75,42,0.35)",
            name="Terrain",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=obstruction_top,
            mode="lines",
            line={"color": "#2e8b57", "width": 1.5},
            name="Terrain + clutter",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=fresnel_lower,
            mode="lines",
            line={"color": "#4682b4", "width": 1.0, "dash": "dash"},
            name="60% Fresnel lower",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=fresnel_upper,
            mode="lines",
            line={"color": "#4682b4", "width": 1.0, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(135,206,235,0.15)",
            name="60% Fresnel envelope",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=los_line,
            mode="lines",
            line={"color": "black", "width": 1.5},
            name="Direct LOS",
        )
    )

    if np.any(attenuation_events):
        fig.add_trace(
            go.Scatter(
                x=distance[attenuation_events],
                y=obstruction_top[attenuation_events],
                mode="markers",
                marker={
                    "size": 8 + 12 * blockage_fraction[attenuation_events],
                    "color": blockage_fraction[attenuation_events],
                    "colorscale": "Reds",
                    "cmin": 0,
                    "cmax": 1,
                    "line": {"color": "black", "width": 0.5},
                    "colorbar": {"title": "Blocked fraction"},
                },
                name="Attenuation events",
            )
        )

    if np.any(direct_los_hits):
        fig.add_trace(
            go.Scatter(
                x=distance[direct_los_hits],
                y=obstruction_top[direct_los_hits],
                mode="markers",
                marker={"symbol": "x", "size": 8, "color": "darkred"},
                name="Direct LOS blocked",
            )
        )

    if np.any(terrain_only_blocked):
        fig.add_trace(
            go.Scatter(
                x=distance[terrain_only_blocked],
                y=terrain_profile[terrain_only_blocked],
                mode="markers",
                marker={"symbol": "diamond", "size": 8, "color": "purple"},
                name="Terrain LOS block",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[distance[0]],
            y=[los_line[0]],
            mode="markers+text",
            marker={"size": 12, "color": source_color, "line": {"color": "#f8fafc", "width": 1.5}},
            text=[source_name],
            textposition="top center",
            textfont={"color": source_color, "family": "Open Sans, sans-serif", "size": 12},
            name=source_name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[distance[-1]],
            y=[los_line[-1]],
            mode="markers+text",
            marker={"size": 12, "color": target_color, "line": {"color": "#f8fafc", "width": 1.5}},
            text=[target_name],
            textposition="top center",
            textfont={"color": target_color, "family": "Open Sans, sans-serif", "size": 12},
            name=target_name,
        )
    )

    title_text = (
        f"<b>Path profile: {source_name} -> {target_name}</b>"
        "<br>"
        f"<span style='font-size:12px;'>"
        f"RSSI={result['rssi_dbm']:.1f} dBm | "
        f"FSPL+other={DEFAULT_OTHER_LOSSES_DB + result['path_loss_db']:.1f} dB | "
        f"Link={result['link_distance_km']:.2f} km"
        "</span>"
    )

    fig.update_layout(
        height=420,
        margin={"l": 40, "r": 20, "t": 145, "b": 40},
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.96,
            "yanchor": "top",
            "pad": {"b": 16},
        },
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"family": "Open Sans, sans-serif", "color": "#e5e7eb"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "x": 0},
    )
    fig.update_xaxes(title="Distance Along Path (km)", color="#e5e7eb", gridcolor="#374151")
    fig.update_yaxes(title="Elevation (m)", color="#e5e7eb", gridcolor="#374151")
    return fig


def build_node_summary(nodes, selected_node_ids, calculation_store, overlay_selection_store):
    if not nodes:
        return html.Div("No nodes added yet.", style={"color": "#cbd5e1"})

    del calculation_store
    normalized_nodes = [with_node_defaults(node) for node in nodes]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    selected_set = set(selected_node_ids)
    primary_selected_id = str(selected_node_ids[-1]) if selected_node_ids else None
    header = html.Div(
        [
            html.Div("Node Name", style={"fontSize": "12px", "fontWeight": "600", "flex": "1 1 auto"}),
            html.Div(
                "Include in RSSI",
                style={"fontSize": "12px", "fontWeight": "600", "width": "88px", "textAlign": "center"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "padding": "0 10px",
            "color": "#cbd5e1",
        },
    )
    summary_cards = []
    for index, node in enumerate(normalized_nodes):
        is_selected = str(node["id"]) in selected_set
        is_primary = str(node["id"]) == primary_selected_id
        details = [
            html.Div(f"Lon {node['longitude']:.5f} | Lat {node['latitude']:.5f}", style={"fontSize": "12px"}),
        ]

        selector = html.Button(
            [
                html.Div(
                    style={
                        "width": "12px",
                        "height": "12px",
                        "borderRadius": "999px",
                        "backgroundColor": node_color(index),
                        "marginTop": "4px",
                        "flex": "0 0 auto",
                    }
                ),
                html.Div(
                    [html.Div(node["name"], style={"fontWeight": "600"})] + details,
                    style={"minWidth": "0"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "padding": "10px",
                "backgroundColor": "#111827",
                "borderRadius": "8px",
                "border": (
                    "2px solid #1b7f3a"
                    if is_primary
                    else ("2px solid #d2b100" if is_selected else "1px solid transparent")
                ),
                "cursor": "pointer",
                "textAlign": "left",
                "width": "100%",
                "flex": "1 1 auto",
                "color": "#f9fafb",
            },
            id={"type": "node-select", "node_id": str(node["id"])},
            n_clicks=0,
        )

        overlay_toggle = None
        checkbox_value = ["enabled"] if overlay_selection_store.get(str(node["id"]), True) else []
        overlay_toggle = dcc.Checklist(
            id={"type": "rssi-node-enable", "node_id": str(node["id"])},
            options=[{"label": "", "value": "enabled"}],
            value=checkbox_value,
            style={"padding": "0", "margin": "0"},
            inputStyle={"margin": "0"},
            labelStyle={"margin": "0", "padding": "0", "fontSize": "0"},
        )
        checkbox_cell = html.Div(
            overlay_toggle if overlay_toggle is not None else html.Div(style={"height": "18px"}),
            style={
                "width": "88px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "paddingRight": "2px",
                "flex": "0 0 auto",
            },
        )

        editor = None
        if is_primary:
            editor = html.Div(
                [
                    html.Div("Height AGL (m)", style={"fontSize": "12px", "fontWeight": "600"}),
                    dcc.Input(
                        id={"type": "node-config", "field": "height_agl_m", "node_id": str(node["id"])},
                        type="number",
                        value=node["height_agl_m"],
                        debounce=True,
                        style={"width": "100%"},
                    ),
                    html.Div("Antenna Gain (dBi)", style={"fontSize": "12px", "fontWeight": "600"}),
                    dcc.Input(
                        id={"type": "node-config", "field": "antenna_gain_dbi", "node_id": str(node["id"])},
                        type="number",
                        value=node["antenna_gain_dbi"],
                        debounce=True,
                        style={"width": "100%"},
                    ),
                    html.Div("Transmit Power (dBm)", style={"fontSize": "12px", "fontWeight": "600"}),
                    dcc.Input(
                        id={"type": "node-config", "field": "tx_power_dbm", "node_id": str(node["id"])},
                        type="number",
                        value=node["tx_power_dbm"],
                        debounce=True,
                        style={"width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Update Position From Map Click",
                                id={"type": "move-node-button", "node_id": str(node["id"])},
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "4px"},
                            ),
                            html.Button(
                                "Delete Node",
                                id={"type": "delete-node-button", "node_id": str(node["id"])},
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "marginTop": "4px",
                                    "backgroundColor": "#991b1b",
                                    "borderColor": "#b91c1c",
                                    "color": "#fef2f2",
                                },
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "6px",
                    "padding": "10px 10px 0 10px",
                },
            )

        summary_cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            selector,
                            checkbox_cell,
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    ),
                    editor,
                ],
                style={
                    "backgroundColor": "#1f2937",
                    "borderRadius": "8px",
                    "paddingBottom": "10px" if is_selected else "0",
                    "paddingTop": "2px",
                },
            )
        )

    return html.Div(
        [header, html.Div(summary_cards, style={"display": "flex", "flexDirection": "column", "gap": "8px"})],
        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
    )


def parse_uploaded_nodes(contents):
    content_type, content_string = contents.split(",", 1)
    del content_type
    decoded = base64.b64decode(content_string)
    frame = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    expected = {"name", "longitude", "latitude"}
    if not expected.issubset(frame.columns):
        raise ValueError("CSV must contain name, longitude, latitude columns.")
    if "height_agl_m" not in frame.columns:
        frame["height_agl_m"] = DEFAULT_NODE_HEIGHT_M
    if "antenna_gain_dbi" not in frame.columns:
        frame["antenna_gain_dbi"] = DEFAULT_TX_GAIN_DBI
    if "tx_power_dbm" not in frame.columns:
        frame["tx_power_dbm"] = DEFAULT_TX_POWER_DBM
    frame = frame.loc[
        :,
        [
            "name",
            "longitude",
            "latitude",
            "height_agl_m",
            "antenna_gain_dbi",
            "tx_power_dbm",
        ],
    ].dropna()
    return frame


def build_placeholder_map(message="Press Update Graph to load the map."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#e5e7eb", "family": "Open Sans, sans-serif", "size": 16},
    )
    fig.update_layout(
        height=720,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        font={"family": "Open Sans, sans-serif", "color": "#e5e7eb"},
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def info_banner(children, background_color="#c92a2a"):
    return html.Div(
        children,
        style={
            "margin": "0 20px 16px",
            "padding": "10px 14px",
            "backgroundColor": background_color,
            "color": "white",
            "fontWeight": "600",
            "borderRadius": "8px",
        },
    )


app = Dash()
app.title = "🏔️Non Flatlander Mesh Terrain Planner (NF-MTP)"

app.layout = [
    dcc.Store(id="nodes-store", data=[]),
    dcc.Store(id="node-counter-store", data=0),
    dcc.Store(id="selected-node-ids-store", data=[]),
    dcc.Store(id="map-click-mode-store", data={"mode": "none", "node_id": None}),
    dcc.Store(id="bbox-store", data=None),
    dcc.Store(id="rssi-calculation-store", data=None),
    dcc.Store(id="rssi-overlay-selection-store", data={}),
    dcc.Store(id="new-primary-node-store", data=None),
    dcc.Store(id="map-interaction-loading-store", data=False),
    dcc.Store(id="rssi-run-request-store", data=None),
    dcc.Store(id="rssi-progress-meta-store", data=None),
    dcc.Store(id="rssi-progress-complete-store", data=None),
    dcc.Store(id="point-path-store", data=None),
    dcc.Download(id="node-download"),
    dcc.Interval(id="rssi-progress-interval", interval=1000, n_intervals=0, disabled=True),
    html.H1("Non Flatlander Mesh Terrain Planner (NF-MTP)", style={"textAlign": "center", "color": "#f9fafb"}),
    html.Div(id="top-banner-container"),
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Terrain Overlay",
                                title="Control the terrain layer visibility, displayed elevation range, and how elevation colors are clipped on the map.",
                                style={"fontWeight": "600"},
                            ),
                            html.Div(
                                "Adjust the terrain layer visibility and the elevation band emphasized on the map.",
                                style={"fontSize": "12px", "color": "#cbd5e1"},
                            ),
                            html.Div("Terrain overlay alpha", style={"fontWeight": "600", "marginTop": "4px"}),
                            dcc.Slider(0, 1, step=0.05, value=0.45, id="elevation_alpha"),
                            html.Div("Terrain elevation clipping range (m)", style={"fontWeight": "600", "marginTop": "4px"}),
                            dcc.RangeSlider(
                                id="elevation_clip_range",
                                min=0,
                                max=1,
                                value=[0, 1],
                                allowCross=False,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Button(
                                "Set Clipping To Current View",
                                id="clip-visible-elevation-range",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        "Map Settings",
                                        title="Set the longitude and latitude bounds for the terrain area to load, then choose the elevation colormap used for display.",
                                        style={"fontWeight": "600", "cursor": "pointer"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "Define the geographic bounds to load and choose how the elevation layer is colored.",
                                                style={"fontSize": "12px", "color": "#cbd5e1"},
                                            ),
                                            html.Div("Minimum longitude (west boundary)", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                            dcc.Input(id="min_lon", type="number", value=DEFAULT_BBOX["min_lon"], style={"width": "100%"}),
                                            html.Div("Minimum latitude (south boundary)", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                            dcc.Input(id="min_lat", type="number", value=DEFAULT_BBOX["min_lat"], style={"width": "100%"}),
                                            html.Div("Maximum longitude (east boundary)", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                            dcc.Input(id="max_lon", type="number", value=DEFAULT_BBOX["max_lon"], style={"width": "100%"}),
                                            html.Div("Maximum latitude (north boundary)", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                            dcc.Input(id="max_lat", type="number", value=DEFAULT_BBOX["max_lat"], style={"width": "100%"}),
                                            html.Div("Elevation colormap", style={"fontWeight": "600", "marginTop": "8px"}),
                                            dcc.Dropdown(
                                                id="elevation-colormap",
                                                options=[{"label": value, "value": value} for value in ELEVATION_COLOR_SCALE_OPTIONS],
                                                value="Magma",
                                                clearable=False,
                                            ),
                                            html.Button("Update Graph (Can take a bit)", id="update_graph", n_clicks=0, style={"width": "100%"}),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                                    ),
                                ],
                                id="map-bounds-section",
                                open=True,
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        "RSSI Calculations",
                                        title="Configure the RSSI overlay appearance and receiver assumptions, then run or update the node coverage calculations.",
                                        style={"fontWeight": "600", "cursor": "pointer"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "Configure receiver assumptions, overlay appearance, and when to run coverage calculations.",
                                                style={"fontSize": "12px", "color": "#cbd5e1"},
                                            ),
                                            html.Div("RSSI overlay opacity", style={"fontWeight": "600", "marginTop": "8px"}),
                                            dcc.Slider(0, 1, step=0.05, value=0.55, id="rssi-opacity"),
                                            html.Div("Global RX Height AGL (m)", style={"fontWeight": "600"}),
                                            dcc.Input(id="global-rx-height-agl", type="number", value=DEFAULT_NODE_HEIGHT_M, style={"width": "100%"}),
                                            html.Div("Global RX Antenna Gain (dBi)", style={"fontWeight": "600"}),
                                            dcc.Input(id="global-rx-gain-dbi", type="number", value=DEFAULT_RX_GAIN_DBI, style={"width": "100%"}),
                                            html.Div("Max RSSI colormap", style={"fontWeight": "600", "marginTop": "8px"}),
                                            dcc.Dropdown(
                                                id="rssi-colormap",
                                                options=[{"label": value, "value": value} for value in RSSI_COLOR_SCALE_OPTIONS],
                                                value="Turbo",
                                                clearable=False,
                                            ),
                                            html.Button("Draw RSSI For All Nodes", id="draw-rssi-overlay", n_clicks=0, style={"width": "100%"}),
                                            html.Progress(id="rssi-progress-bar", value="0", max="100", style={"width": "100%", "height": "14px"}),
                                            html.Div(id="rssi-progress-label", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                            dcc.Checklist(
                                                id="include-rssi-ground-loss",
                                                options=[{"label": " Include per-cell path attenuation in RSSI overlay", "value": "enabled"}],
                                                value=[],
                                            ),
                                            html.Div(
                                                "⚠️Very Computationally Intensive (1H+ for 10 Nodes)",
                                                style={"fontSize": "12px", "color": "#cbd5e1"},
                                            ),

                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                                    ),
                                ],
                                open=True,
                            ),
                        ],
                        style={
                            "width": "320px",
                            "backgroundColor": "#111827",
                            "padding": "16px",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "12px",
                            "borderRadius": "8px",
                            "border": "1px solid #374151",
                            "color": "#e5e7eb",
                        },
                    ),
                ],
                style={"flex": "0 0 auto"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Loading(
                                [
                                    dcc.Graph(
                                        id="graph-map",
                                        figure=build_placeholder_map(),
                                        style={"height": "78vh", "width": "100%"},
                                        config={"responsive": True},
                                    ),
                                    html.Div(id="rssi-worker", style={"display": "none"}),
                                ],
                                type="circle",
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        html.Div(className="map-loading-spinner"),
                                        html.Div("Updating map..."),
                                    ],
                                    className="map-loading-content",
                                ),
                                id="map-interaction-overlay",
                                style={
                                    "display": "none",
                                    "position": "absolute",
                                    "inset": "0",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "backgroundColor": "rgba(3,7,18,0.5)",
                                    "backdropFilter": "blur(2px)",
                                    "color": "#f9fafb",
                                    "fontWeight": "700",
                                    "fontSize": "16px",
                                    "zIndex": 5,
                                },
                            ),
                        ],
                        style={"position": "relative"},
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="path-profile-graph",
                            figure=empty_path_profile_figure(),
                            style={"height": "40vh", "width": "100%"},
                            config={"responsive": True},
                        ),
                        type="default",
                    ),
                    html.Div(id="path-profile-stats"),
                ],
                style={"flex": "1 1 auto", "minWidth": "0", "display": "flex", "flexDirection": "column", "gap": "16px"},
            ),
            html.Div(
                [
                    html.Div("Nodes", style={"fontWeight": "600"}),
                    html.Div(
                        "Select up to two nodes. If two are selected, the path between them will be drawn on the map and the profile will be shown below.",
                        style={"fontSize": "12px", "color": "#cbd5e1"},
                    ),
                    html.Button("Draw Path To Map Point", id="draw-point-path", n_clicks=0, style={"width": "100%"}),
                    html.Details(
                        [
                            html.Summary("Click To Add Node", style={"fontWeight": "600", "cursor": "pointer"}),
                            html.Div(
                                [
                                    html.Button("Enable Click-To-Add", id="toggle-click-add", n_clicks=0, style={"width": "100%"}),
                                    html.Div(id="click-add-status", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        open=True,
                    ),
                    html.Details(
                        [
                            html.Summary("Manual Node Entry", style={"fontWeight": "600", "cursor": "pointer"}),
                            html.Div(
                                [
                                    dcc.Input(id="manual-node-name", type="text", placeholder="Node name", style={"width": "100%"}),
                                    dcc.Input(id="manual-node-lon", type="number", placeholder="Longitude", style={"width": "100%"}),
                                    dcc.Input(id="manual-node-lat", type="number", placeholder="Latitude", style={"width": "100%"}),
                                    html.Button("Add Node", id="add-manual-node", n_clicks=0, style={"width": "100%"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        open=False,
                    ),
                    html.Div(
                        dcc.Loading(html.Div(id="node-summary"), type="default"),
                        style={
                            "maxHeight": "480px",
                            "overflowY": "auto",
                            "paddingRight": "4px",
                        },
                    ),
                    html.Details(
                        [
                            html.Summary("CSV Import / Export", style={"fontWeight": "600", "cursor": "pointer"}),
                            html.Div(
                                [
                                    dcc.Upload(
                                        id="node-upload",
                                        children=html.Button("Load Nodes CSV", style={"width": "100%"}),
                                        multiple=False,
                                    ),
                                    html.Button("Save Nodes CSV", id="save-nodes", n_clicks=0, style={"width": "100%"}),
                                    html.Div(id="node-action-message", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        open=False,
                    ),
                ],
                style={
                    "width": "320px",
                    "backgroundColor": "#111827",
                    "padding": "16px",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "12px",
                    "borderRadius": "8px",
                    "border": "1px solid #374151",
                    "color": "#e5e7eb",
                    "flex": "0 0 auto",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "flex-start", "gap": "20px", "padding": "0 20px 20px"},
    ),
]


@app.callback(
    Output("bbox-store", "data"),
    Input("update_graph", "n_clicks"),
    State("min_lon", "value"),
    State("min_lat", "value"),
    State("max_lon", "value"),
    State("max_lat", "value"),
    prevent_initial_call=True,
)
def update_bbox_store(_n_clicks, min_lon, min_lat, max_lon, max_lat):
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(min_lon, min_lat, max_lon, max_lat)
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


@app.callback(
    Output("elevation_clip_range", "min"),
    Output("elevation_clip_range", "max"),
    Output("elevation_clip_range", "value"),
    Output("elevation_clip_range", "marks"),
    Input("bbox-store", "data"),
)
def update_elevation_clip_controls(bbox_data):
    if not bbox_data:
        return 0, 1, [0, 1], {0: "0", 1: "1"}

    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox)
        terrain_display = np.flipud(bundle["terrain_display"])
        finite = terrain_display[np.isfinite(terrain_display)]
        if finite.size == 0:
            return 0, 1, [0, 1], {0: "0", 1: "1"}
        slider_min = int(math.floor(float(np.nanmin(finite))))
        slider_max = int(math.ceil(float(np.nanmax(finite))))
        if slider_max <= slider_min:
            slider_max = slider_min + 1
        midpoint = int(round((slider_min + slider_max) / 2))
        marks = {
            slider_min: str(slider_min),
            midpoint: str(midpoint),
            slider_max: str(slider_max),
        }
        return slider_min, slider_max, [slider_min, slider_max], marks
    except Exception:
        return 0, 1, [0, 1], {0: "0", 1: "1"}


@app.callback(
    Output("elevation_clip_range", "value", allow_duplicate=True),
    Input("clip-visible-elevation-range", "n_clicks"),
    State("graph-map", "relayoutData"),
    State("bbox-store", "data"),
    prevent_initial_call=True,
)
def set_elevation_clip_to_current_view(_n_clicks, relayout_data, bbox_data):
    if not bbox_data:
        raise PreventUpdate

    bbox = normalize_bbox(
        bbox_data["min_lon"],
        bbox_data["min_lat"],
        bbox_data["max_lon"],
        bbox_data["max_lat"],
    )
    bundle = get_map_bundle(*bbox)

    terrain_lon = bundle["terrain_display_lon_axis"]
    terrain_lat = bundle["terrain_display_lat_axis"][::-1]
    terrain_display = np.flipud(bundle["terrain_display"])

    full_x_range = (float(np.min(terrain_lon)), float(np.max(terrain_lon)))
    full_y_range = (float(np.min(terrain_lat)), float(np.max(terrain_lat)))
    x_range, y_range = ranges_from_relayout_data(relayout_data, full_x_range, full_y_range)

    lon_mask = (terrain_lon >= x_range[0]) & (terrain_lon <= x_range[1])
    lat_mask = (terrain_lat >= y_range[0]) & (terrain_lat <= y_range[1])
    if not np.any(lon_mask) or not np.any(lat_mask):
        raise PreventUpdate

    visible = terrain_display[np.ix_(lat_mask, lon_mask)]
    finite = visible[np.isfinite(visible)]
    if finite.size == 0:
        raise PreventUpdate

    visible_min = int(math.floor(float(np.nanmin(finite))))
    visible_max = int(math.ceil(float(np.nanmax(finite))))
    if visible_max <= visible_min:
        visible_max = visible_min + 1
    return [visible_min, visible_max]


@app.callback(
    Output("top-banner-container", "children"),
    Input("bbox-store", "data"),
    Input("nodes-store", "data"),
    Input("rssi-calculation-store", "data"),
)
def update_top_banner(bbox_data, nodes, calculation_store):
    if not bbox_data:
        return info_banner("Enter the coordinates for the map area you want to inspect, then click Update Graph (This can take up to a minute, especially with larger areas)")

    if calculation_store:
        current_signatures = {str(node["id"]): node_signature(node) for node in (nodes or [])}
        calculated_signatures = calculation_store.get("node_signatures", {})
        if current_signatures != calculated_signatures:
            return info_banner(
                "Nodes have been added or changed since the last RSSI calculation. Re-run RSSI calculations to refresh the overlay."
            )
    return None


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("toggle-click-add", "n_clicks"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_click_add_mode(_n_clicks, click_mode):
    mode = (click_mode or {}).get("mode", "none")
    if mode == "add-node":
        return {"mode": "none", "node_id": None}
    return {"mode": "add-node", "node_id": None}


@app.callback(
    Output("toggle-click-add", "children"),
    Output("click-add-status", "children"),
    Input("map-click-mode-store", "data"),
    State("nodes-store", "data"),
)
def update_click_add_text(click_mode, nodes):
    del nodes
    mode = (click_mode or {}).get("mode", "none")
    if mode == "add-node":
        return "Disable Click-To-Add", "Click once on the map to add a node."
    return "Enable Click-To-Add", "Click-to-add is off."


@app.callback(
    Output("draw-point-path", "children"),
    Output("draw-point-path", "disabled"),
    Input("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
)
def update_point_path_button(selected_node_ids, nodes):
    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    primary_id = str(selected_node_ids[-1]) if selected_node_ids else None
    primary = find_node(normalized_nodes, primary_id)
    if primary is None:
        return "Draw Path To Map Point", True
    return f"Draw Path From {primary['name']} To Map Point", False


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Input("draw-point-path", "n_clicks"),
    State("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def enable_point_path_mode(n_clicks, selected_node_ids, nodes):
    if not n_clicks:
        raise PreventUpdate

    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    primary_id = str(selected_node_ids[-1]) if selected_node_ids else None
    primary = find_node(normalized_nodes, primary_id)
    if primary is None:
        return no_update, "Select a primary node before drawing a path to a map point."
    return {"mode": "point-path", "node_id": str(primary["id"])}, f"Click once on the map to trace from {primary['name']}."


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Input({"type": "move-node-button", "node_id": ALL}, "n_clicks"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def enable_move_node_mode(_n_clicks, nodes):
    triggered = ctx.triggered_id
    if not triggered or "node_id" not in triggered:
        raise PreventUpdate
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        raise PreventUpdate

    node = find_node(nodes or [], triggered["node_id"])
    if node is None:
        raise PreventUpdate
    return {"mode": "move-node", "node_id": str(node["id"])}, f"Click once on the map to move {node['name']}."


@app.callback(
    Output("nodes-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input({"type": "delete-node-button", "node_id": ALL}, "n_clicks"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def delete_node(_n_clicks, nodes):
    triggered = ctx.triggered_id
    if not triggered or "node_id" not in triggered:
        raise PreventUpdate
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        raise PreventUpdate

    node_id = str(triggered["node_id"])
    updated_nodes = [node for node in (nodes or []) if str(node["id"]) != node_id]
    if len(updated_nodes) == len(nodes or []):
        raise PreventUpdate
    return updated_nodes, "Node deleted.", {"mode": "none", "node_id": None}


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Input("graph-map", "relayoutData"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def cancel_move_mode_on_map_relayout(relayout_data, click_mode):
    if not relayout_data:
        raise PreventUpdate
    if (click_mode or {}).get("mode") != "move-node":
        raise PreventUpdate
    return {"mode": "none", "node_id": None}, "Move-node mode canceled."


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="handleMapClick"),
    Output("nodes-store", "data", allow_duplicate=True),
    Output("node-counter-store", "data", allow_duplicate=True),
    Output("new-primary-node-store", "data", allow_duplicate=True),
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Output("point-path-store", "data", allow_duplicate=True),
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Input("graph-map", "clickData"),
    State("map-click-mode-store", "data"),
    State("nodes-store", "data"),
    State("node-counter-store", "data"),
    State("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Input({"type": "node-select", "node_id": ALL}, "n_clicks"),
    Input("graph-map", "clickData"),
    State("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def update_selected_nodes(_n_clicks, click_data, selected_node_ids, nodes, click_mode):
    triggered = ctx.triggered_id
    valid_ids = [str(node["id"]) for node in (nodes or [])]
    if isinstance(triggered, dict) and "node_id" in triggered:
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        return toggle_selected_node(selected_node_ids, triggered["node_id"], valid_ids)
    if triggered != "graph-map":
        raise PreventUpdate
    if (click_mode or {}).get("mode", "none") != "none":
        raise PreventUpdate
    node_id = clicked_node_id_from_click_data(click_data)
    if node_id is None:
        raise PreventUpdate
    return toggle_selected_node(selected_node_ids, node_id, valid_ids)


@app.callback(
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Input("new-primary-node-store", "data"),
    State("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)
def apply_new_primary_node(primary_node_id, selected_node_ids):
    if not primary_node_id:
        raise PreventUpdate
    return select_primary_node(selected_node_ids, primary_node_id)


@app.callback(
    Output("point-path-store", "data", allow_duplicate=True),
    Input("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)
def clear_point_path_when_two_nodes_selected(selected_node_ids):
    if len(normalize_selected_node_ids(selected_node_ids)) == 2:
        return None
    raise PreventUpdate


@app.callback(
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Input("nodes-store", "data"),
    State("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)
def sync_selected_nodes(nodes, selected_node_ids):
    valid_ids = {str(node["id"]) for node in (nodes or [])}
    return normalize_selected_node_ids(selected_node_ids, valid_ids)


@app.callback(
    Output("nodes-store", "data", allow_duplicate=True),
    Input({"type": "node-config", "field": ALL, "node_id": ALL}, "value"),
    State({"type": "node-config", "field": ALL, "node_id": ALL}, "id"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def update_node_config(_values, ids, nodes):
    del _values
    triggered = ctx.triggered_id
    if not triggered or "node_id" not in triggered or "field" not in triggered:
        raise PreventUpdate

    node_id = str(triggered["node_id"])
    field = str(triggered["field"])
    value = ctx.triggered[0]["value"]
    if value is None:
        raise PreventUpdate

    updated_nodes = []
    changed = False
    for node in (nodes or []):
        normalized = with_node_defaults(node)
        if str(normalized["id"]) == node_id:
            normalized[field] = float(value)
            changed = True
        updated_nodes.append(normalized)

    if not changed:
        raise PreventUpdate
    return updated_nodes


@app.callback(
    Output("rssi-overlay-selection-store", "data", allow_duplicate=True),
    Input({"type": "rssi-node-enable", "node_id": ALL}, "value"),
    State({"type": "rssi-node-enable", "node_id": ALL}, "id"),
    State("rssi-overlay-selection-store", "data"),
    prevent_initial_call=True,
)
def update_rssi_overlay_selection(_values, ids, selection_store):
    del _values
    triggered = ctx.triggered_id
    if not triggered or "node_id" not in triggered:
        raise PreventUpdate

    node_id = str(triggered["node_id"])
    value = ctx.triggered[0]["value"]
    updated = dict(selection_store or {})
    updated[node_id] = bool(value and "enabled" in value)
    return updated


@app.callback(
    Output("nodes-store", "data"),
    Output("node-counter-store", "data"),
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Output("manual-node-name", "value"),
    Output("manual-node-lon", "value"),
    Output("manual-node-lat", "value"),
    Output("node-action-message", "children"),
    Input("add-manual-node", "n_clicks"),
    Input("node-upload", "contents"),
    State("node-upload", "filename"),
    State("manual-node-name", "value"),
    State("manual-node-lon", "value"),
    State("manual-node-lat", "value"),
    State("nodes-store", "data"),
    State("node-counter-store", "data"),
    State("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)
def manage_nodes(
    add_manual_clicks,
    upload_contents,
    upload_filename,
    manual_name,
    manual_lon,
    manual_lat,
    nodes,
    counter,
    selected_node_ids,
):
    del add_manual_clicks

    nodes = list(nodes or [])
    counter = int(counter or 0)
    trigger = ctx.triggered_id

    if trigger == "node-upload":
        if not upload_contents:
            raise PreventUpdate
        try:
            frame = parse_uploaded_nodes(upload_contents)
        except Exception as exc:
            return no_update, no_update, no_update, no_update, no_update, no_update, f"CSV load failed: {exc}"
        loaded_nodes = []
        last_loaded_id = None
        for row in frame.itertuples(index=False):
            counter += 1
            last_loaded_id = f"node-{counter}"
            loaded_nodes.append(
                {
                    "id": last_loaded_id,
                    "name": str(row.name),
                    "longitude": float(row.longitude),
                    "latitude": float(row.latitude),
                    "height_agl_m": float(row.height_agl_m),
                    "antenna_gain_dbi": float(row.antenna_gain_dbi),
                    "tx_power_dbm": float(row.tx_power_dbm),
                }
            )
        message = f"Loaded {len(loaded_nodes)} nodes from {upload_filename or 'CSV'}."
        next_selected = select_primary_node(selected_node_ids, last_loaded_id) if last_loaded_id is not None else (selected_node_ids or [])
        return loaded_nodes, counter, next_selected, "", None, None, message

    if trigger == "add-manual-node":
        if not manual_name or manual_lon is None or manual_lat is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, "Manual node requires name, longitude, and latitude."
        counter += 1
        node_id = f"node-{counter}"
        nodes.append(
            {
                "id": node_id,
                "name": str(manual_name),
                "longitude": float(manual_lon),
                "latitude": float(manual_lat),
                "height_agl_m": DEFAULT_NODE_HEIGHT_M,
                "antenna_gain_dbi": DEFAULT_TX_GAIN_DBI,
                "tx_power_dbm": DEFAULT_TX_POWER_DBM,
            }
        )
        message = f"Added node {manual_name}."
        return nodes, counter, select_primary_node(selected_node_ids, node_id), "", None, None, message

    raise PreventUpdate


@app.callback(
    Output("node-download", "data"),
    Input("save-nodes", "n_clicks"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def save_nodes_csv(_n_clicks, nodes):
    frame = pd.DataFrame([with_node_defaults(node) for node in (nodes or [])])
    frame = (
        frame.loc[
            :,
            [
                "name",
                "longitude",
                "latitude",
                "height_agl_m",
                "antenna_gain_dbi",
                "tx_power_dbm",
            ],
        ]
        if not frame.empty
        else pd.DataFrame(
            columns=[
                "name",
                "longitude",
                "latitude",
                "height_agl_m",
                "antenna_gain_dbi",
                "tx_power_dbm",
            ]
        )
    )
    return dcc.send_data_frame(frame.to_csv, "mesh_nodes.csv", index=False)


@app.callback(
    Output("graph-map", "figure"),
    Input("bbox-store", "data"),
    Input("nodes-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("elevation_alpha", "value"),
    Input("elevation_clip_range", "value"),
    Input("elevation-colormap", "value"),
    Input("rssi-calculation-store", "data"),
    Input("rssi-overlay-selection-store", "data"),
    Input("rssi-opacity", "value"),
    Input("rssi-colormap", "value"),
    Input("point-path-store", "data"),
    State("global-rx-height-agl", "value"),
    State("global-rx-gain-dbi", "value"),
)
def update_map_figure(
    bbox_data,
    nodes,
    selected_node_ids,
    terrain_alpha,
    terrain_clip_range,
    elevation_colormap,
    rssi_calculation_store,
    rssi_overlay_selection_store,
    rssi_opacity,
    rssi_colormap,
    point_path_data,
    global_rx_height_agl,
    global_rx_gain_dbi,
):
    if not bbox_data:
        return build_placeholder_map("Enter a map area above and click Update Graph.")
    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox)
    except Exception as exc:
        return build_placeholder_map(f"Failed to load map data: {exc}")

    nodes = list(nodes or [])

    rssi_overlay = compose_cached_rssi_overlay(bundle, rssi_calculation_store, rssi_overlay_selection_store or {})

    try:
        path_overlay = compute_map_path_overlay(
            nodes,
            selected_node_ids,
            point_path_data,
            bbox_data,
            global_rx_height_agl,
            global_rx_gain_dbi,
        )
    except Exception:
        path_overlay = {"shapes": [], "traces": []}

    return build_map_figure(
        bundle,
        nodes,
        terrain_alpha or 0.0,
        terrain_clip_range=terrain_clip_range,
        elevation_colorscale=elevation_colormap or "Magma",
        rssi_overlay=rssi_overlay,
        rssi_colorscale=rssi_colormap or "Turbo",
        shapes=path_overlay["shapes"],
        path_traces=path_overlay["traces"],
        rssi_opacity=rssi_opacity or 0.55,
        selected_node_ids=selected_node_ids,
    )


@app.callback(
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Input("update_graph", "n_clicks"),
    prevent_initial_call=True,
)
def start_map_loading(_n_clicks):
    return True


@app.callback(
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Input("graph-map", "figure"),
    State("map-interaction-loading-store", "data"),
    prevent_initial_call=True,
)
def clear_map_interaction_loading(_figure, is_loading):
    if is_loading:
        return False
    raise PreventUpdate


@app.callback(
    Output("map-interaction-overlay", "style"),
    Input("map-interaction-loading-store", "data"),
    State("map-interaction-overlay", "style"),
)
def update_map_interaction_overlay(is_loading, current_style):
    style = dict(current_style or {})
    style["display"] = "flex" if is_loading else "none"
    return style


@app.callback(
    Output("node-summary", "children"),
    Input("nodes-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("rssi-calculation-store", "data"),
    Input("rssi-overlay-selection-store", "data"),
)
def update_node_summary(nodes, selected_node_ids, calculation_store, overlay_selection_store):
    return build_node_summary(
        list(nodes or []),
        [str(value) for value in (selected_node_ids or [])][:2],
        calculation_store,
        overlay_selection_store or {},
    )


@app.callback(
    Output("rssi-calculation-store", "data"),
    Output("rssi-overlay-selection-store", "data"),
    Output("rssi-progress-complete-store", "data"),
    Output("node-action-message", "children", allow_duplicate=True),
    Output("rssi-worker", "children"),
    Input("rssi-run-request-store", "data"),
    State("bbox-store", "data"),
    State("nodes-store", "data"),
    State("rssi-overlay-selection-store", "data"),
    State("global-rx-height-agl", "value"),
    State("global-rx-gain-dbi", "value"),
    State("include-rssi-ground-loss", "value"),
    prevent_initial_call=True,
)
def generate_rssi_overlay(
    run_request,
    bbox_data,
    nodes,
    existing_overlay_selection,
    global_rx_height_agl,
    global_rx_gain_dbi,
    include_rssi_ground_loss,
):
    if not run_request:
        raise PreventUpdate
    request_id = int(run_request.get("request_id", 0))
    if not request_id:
        raise PreventUpdate
    nodes = list(nodes or [])
    if not nodes:
        return no_update, no_update, {"request_id": request_id, "done": True}, "Add at least one node before drawing RSSI.", f"noop-{request_id}"
    if not bbox_data:
        return no_update, no_update, {"request_id": request_id, "done": True}, "Load a map area before drawing RSSI.", f"noop-{request_id}"

    bbox = normalize_bbox(
        bbox_data["min_lon"],
        bbox_data["min_lat"],
        bbox_data["max_lon"],
        bbox_data["max_lat"],
    )
    bundle = get_map_bundle(*bbox)
    include_ground_loss = "enabled" in (include_rssi_ground_loss or [])
    rx_height = global_rx_height_agl or DEFAULT_NODE_HEIGHT_M
    rx_gain = global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI
    node_overlay_keys = {}
    node_signatures = {}
    try:
        for node in nodes:
            cache_key, _result = compute_single_node_rssi_overlay(node, bundle, include_ground_loss, rx_height, rx_gain)
            node_id = str(node["id"])
            node_overlay_keys[node_id] = cache_key
            node_signatures[node_id] = node_signature(node)
    except Exception as exc:
        return no_update, no_update, {"request_id": request_id, "done": True}, f"RSSI overlay failed: {exc}", f"error-{request_id}"

    calc_store = {
        "request_id": request_id,
        "bundle_key": list(bundle["cache_key"]),
        "include_ground_loss": include_ground_loss,
        "global_rx_height_agl": float(rx_height),
        "global_rx_gain_dbi": float(rx_gain),
        "node_overlay_keys": node_overlay_keys,
        "node_signatures": node_signatures,
        "node_order": [str(node["id"]) for node in nodes],
    }
    existing_overlay_selection = dict(existing_overlay_selection or {})
    overlay_selection = {
        str(node["id"]): bool(existing_overlay_selection.get(str(node["id"]), True))
        for node in nodes
    }
    mode = "with path attenuation" if include_ground_loss else "with viewshed + FSPL only"
    return calc_store, overlay_selection, {"request_id": request_id, "done": True}, f"RSSI overlay updated {mode}.", f"done-{request_id}"


@app.callback(
    Output("rssi-progress-bar", "value"),
    Output("rssi-progress-label", "children"),
    Output("rssi-progress-interval", "disabled"),
    Input("rssi-progress-interval", "n_intervals"),
    Input("rssi-progress-meta-store", "data"),
    Input("rssi-progress-complete-store", "data"),
)
def update_rssi_progress(_n_intervals, progress_meta, progress_complete):
    if not progress_meta:
        return "0", "", True

    if progress_complete and progress_complete.get("request_id") == progress_meta.get("request_id"):
        return "100", "RSSI overlay complete.", True

    start_ms = float(progress_meta.get("start_ms", 0))
    eta_sec = max(float(progress_meta.get("eta_sec", 1)), 1.0)
    elapsed_sec = max((pd.Timestamp.utcnow().timestamp() * 1000.0 - start_ms) / 1000.0, 0.0)
    progress = min(95, int((elapsed_sec / eta_sec) * 100))
    remaining = max(0, int(round(eta_sec - elapsed_sec)))
    label = f"Computing RSSI overlay... {elapsed_sec:.0f}s elapsed, about {remaining}s remaining."
    return str(progress), label, False


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="startRssiOverlay"),
    Output("rssi-progress-meta-store", "data"),
    Output("rssi-progress-interval", "disabled", allow_duplicate=True),
    Output("rssi-progress-complete-store", "data", allow_duplicate=True),
    Output("rssi-run-request-store", "data"),
    Input("draw-rssi-overlay", "n_clicks"),
    State("nodes-store", "data"),
    State("include-rssi-ground-loss", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("map-bounds-section", "open"),
    Input("update_graph", "n_clicks"),
    prevent_initial_call=True,
)
def collapse_map_bounds(_n_clicks):
    return False


@app.callback(
    Output("path-profile-graph", "figure"),
    Output("path-profile-stats", "children"),
    Input("point-path-store", "data"),
    Input("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
    State("bbox-store", "data"),
    State("global-rx-height-agl", "value"),
    State("global-rx-gain-dbi", "value"),
)
def update_path_profile(point_path_data, selected_node_ids, nodes, bbox_data, global_rx_height_agl, global_rx_gain_dbi):
    def build_stats_panel(stats):
        return html.Div(
            [
                html.Div(
                    item,
                    style={
                        "padding": "6px 10px",
                        "backgroundColor": "#1f2937",
                        "borderRadius": "6px",
                        "color": "#e5e7eb",
                        "border": "1px solid #374151",
                    },
                )
                for item in stats
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "8px"},
        )

    nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in nodes])
    color_lookup = node_color_map(nodes)
    if not bbox_data:
        return empty_path_profile_figure("Load a map area to inspect path profiles."), html.Div()
    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox)
        ensure_analysis_context(bundle)
    except Exception as exc:
        return empty_path_profile_figure(f"Path profile unavailable: {exc}"), html.Div()

    if len(selected_node_ids) == 2:
        primary_node_id = selected_node_ids[-1]
        secondary_node_id = selected_node_ids[0]
        source = find_node(nodes, primary_node_id)
        target = find_node(nodes, secondary_node_id)
        if source is None or target is None:
            return empty_path_profile_figure("Selected path no longer exists."), html.Div()

        result = compute_path_loss(
            (source["longitude"], source["latitude"]),
            (target["longitude"], target["latitude"]),
            tx_height=source["height_agl_m"],
            rx_height=target["height_agl_m"],
            tx_power_dbm=source["tx_power_dbm"],
            tx_gain_dbi=source["antenna_gain_dbi"],
            rx_gain_dbi=target["antenna_gain_dbi"],
        )
        stats = [
            f"Path: {source['name']} -> {target['name']}",
            f"RSSI: {result['rssi_dbm']:.1f} dBm",
            f"Distance: {result['link_distance_km']:.2f} km",
            f"Path attenuation: {result['path_loss_db']:.1f} dB",
            f"Attenuation events: {result['attenuation_event_count']}",
            f"Terrain collision events: {result['terrain_collision_event_count']}",
            f"Direct LOS collisions: {result['direct_los_collision_count']}",
            (
                f"Source TX height/gain/power: {source['height_agl_m']:.1f} m / "
                f"{source['antenna_gain_dbi']:.1f} dBi / {source['tx_power_dbm']:.1f} dBm"
            ),
            (
                f"Target RX height/gain: {target['height_agl_m']:.1f} m / "
                f"{target['antenna_gain_dbi']:.1f} dBi"
            ),
        ]
        return (
            build_path_profile_figure(
                source["name"],
                target["name"],
                result,
                source_color=color_lookup.get(str(source["id"]), "#22c55e"),
                target_color=color_lookup.get(str(target["id"]), "#f59e0b"),
            ),
            build_stats_panel(stats),
        )

    if point_path_data:
        source = find_node(nodes, point_path_data.get("source_node_id"))
        target_lon = point_path_data.get("target_longitude")
        target_lat = point_path_data.get("target_latitude")
        if source is not None and target_lon is not None and target_lat is not None:
            result = compute_path_loss(
                (source["longitude"], source["latitude"]),
                (float(target_lon), float(target_lat)),
                tx_height=source["height_agl_m"],
                rx_height=float(global_rx_height_agl or DEFAULT_NODE_HEIGHT_M),
                tx_power_dbm=source["tx_power_dbm"],
                tx_gain_dbi=source["antenna_gain_dbi"],
                rx_gain_dbi=float(global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI),
            )
            stats = [
                f"Path: {source['name']} -> Map Point",
                f"Endpoint: {float(target_lon):.5f}, {float(target_lat):.5f}",
                f"RSSI: {result['rssi_dbm']:.1f} dBm",
                f"Distance: {result['link_distance_km']:.2f} km",
                f"Path attenuation: {result['path_loss_db']:.1f} dB",
                f"Attenuation events: {result['attenuation_event_count']}",
                f"Terrain collision events: {result['terrain_collision_event_count']}",
                f"Direct LOS collisions: {result['direct_los_collision_count']}",
                (
                    f"Source TX height/gain/power: {source['height_agl_m']:.1f} m / "
                    f"{source['antenna_gain_dbi']:.1f} dBi / {source['tx_power_dbm']:.1f} dBm"
                ),
                (
                    f"Global RX height/gain: {float(global_rx_height_agl or DEFAULT_NODE_HEIGHT_M):.1f} m / "
                    f"{float(global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI):.1f} dBi"
                ),
            ]
            return (
                build_path_profile_figure(
                    source["name"],
                    "Map Point",
                    result,
                    source_color=color_lookup.get(str(source["id"]), "#22c55e"),
                    target_color="#f59e0b",
                ),
                build_stats_panel(stats),
            )

    return empty_path_profile_figure("Select exactly two nodes or use Draw Path To Map Point to inspect the path profile."), html.Div()


if __name__ == "__main__":
    app.run(debug=True)
