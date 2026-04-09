import asyncio
import base64
import gc
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import rasterio
import requests
import xarray as xr
import dask
import dask.array as da
from dask.callbacks import Callback
from PIL import Image, ImageColor
from dash import ALL, ClientsideFunction, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import Response
try:
    from numba import njit, prange
except ModuleNotFoundError:
    njit = None
    prange = range
from plotly.colors import qualitative, sample_colorscale
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from scipy.interpolate import RBFInterpolator
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class _NullTqdm:
        def __init__(self, iterable=None, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, _n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **_kwargs):
        return _NullTqdm(iterable=iterable, **_kwargs)
from xrspatial import viewshed

try:
    from desktop_notifier import DesktopNotifier
except ModuleNotFoundError:
    DesktopNotifier = None

PROJECTED_CRS = "EPSG:26912"  # NAD83 / UTM zone 12N, suitable for Utah
GEOGRAPHIC_CRS = "EPSG:4326"
WEB_MERCATOR_CRS = "EPSG:3857"
WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_VERSION = "v200"
WORLDCOVER_YEAR = "2021"
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_TILE_SIZE = 256
OSM_MIN_ZOOM = 2
OSM_MAX_ZOOM = 17
OSM_MAX_TILE_COUNT = 64
OSM_TILE_HEADERS = {"User-Agent": "NF-MTP/1.0"}

DEFAULT_BBOX = {
    "min_lon": -113.90,
    "min_lat": 39.8667,
    "max_lon": -111.15,
    "max_lat": 42.65,
}

DISPLAY_MAX_DIM = 2400
DISPLAY_MIN_DIM = 512
MIN_BBOX_RESOLUTION_M = 50.0
MAX_BBOX_RESOLUTION_M = 200.0
DEFAULT_BBOX_RESOLUTION_M = 100.0
BBOX_RESOLUTION_STEP_M = 25.0
DASK_STREAM_MEMMAP_MIN_BYTES = 16 * 1024 * 1024
DASK_STREAM_MEMMAP_DIR = os.path.join(tempfile.gettempdir(), "mesh_terrain_dask_memmaps")
DEFAULT_NODE_HEIGHT_M = 2.0
DEFAULT_GLOBAL_RX_HEIGHT_M = 2.0
DEFAULT_FREQ_MHZ = 915.0
DEFAULT_TX_POWER_DBM = 22.0
DEFAULT_TX_GAIN_DBI = 5.0
DEFAULT_RX_GAIN_DBI = 2.0
DEFAULT_OTHER_LOSSES_DB = 3.0
DEFAULT_RSSI_PATH_SAMPLE_SPACING_M = DEFAULT_BBOX_RESOLUTION_M
MIN_LINK_RSSI_DBM = -140.0
VIEWSHED_ASSESSMENT_RADIUS_M = 100.0
VIEWSHED_ASSESSMENT_OBSERVER_HEIGHT_AGL_M = 0.0
VIEWSHED_ASSESSMENT_MAX_SAMPLES = 160
VIEWSHED_ASSESSMENT_TARGET_RESOLUTION_M = 1.0
VIEWSHED_ASSESSMENT_MIN_DIM = 64
VIEWSHED_ASSESSMENT_MAX_DIM = 128
VIEWSHED_ASSESSMENT_METRIC_VERSION = "global-visible-cell-count-v5-circular-thin-plate-spline"
VIEWSHED_SAMPLE_COUNT_OPTIONS = [7, 19, 37, 61, 91, 127]
DEFAULT_VIEWSHED_SAMPLE_COUNT = 37
VIEWSHED_COLOR_SCALE_OPTIONS = ["Turbo", "Viridis", "Magma", "Inferno", "Cividis", "Plasma"]
VIEWSHED_POINT_COLOR = "#ef4444"
VIEWSHED_POINT_OUTLINE_COLOR = "#ffffff"
VIEWSHED_LEGEND_TITLE = "Visible Cells"

NODE_EXCLUDED_COLORS = {"#000000", "#000", "black", "#222a2a"}
COLOR_SEQUENCE = [
    color
    for color in (qualitative.Dark24 + qualitative.Bold + qualitative.Safe + qualitative.Set3)
    if str(color).strip().lower() not in NODE_EXCLUDED_COLORS
]
ELEVATION_COLOR_SCALE_OPTIONS = ["Magma", "Viridis", "Cividis", "Inferno", "Plasma", "Greys"]
RSSI_COLOR_SCALE_OPTIONS = ["Turbo", "Viridis", "Cividis", "Inferno", "Plasma", "RdYlGn"]
BASE_MAP_STYLE_OPTIONS = [
    {"label": "Satellite", "value": "satellite"},
    {"label": "Street", "value": "street"},
]
PATH_GOOD_COLOR = "#1b7f3a"
PATH_BAD_COLOR = "#c23b22"
PATH_PREVIEW_COLOR = "#38bdf8"
ATTENUATION_EVENT_COLOR = "#f97316"
ATTENUATION_EVENT_EDGE_COLOR = "#7c2d12"
TERRAIN_BLOCK_COLOR = "#a855f7"
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
WORLDCOVER_PALETTE = {
    10: "#006400",
    20: "#ffbb22",
    30: "#ffff4c",
    40: "#f096ff",
    50: "#fa0000",
    60: "#b4b4b4",
    70: "#f0f0f0",
    80: "#0064c8",
    90: "#0096a0",
    95: "#00cf75",
    100: "#fae6a0",
}
MAPLIBRE_VERSION = "5.16.0"
MAPLIBRE_JS_URL = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.js"
MAPLIBRE_CSS_URL = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.css"
MAPLIBRE_GLYPHS_URL = "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf"
TERRAIN_TILE_SIZE = 256
TERRAIN_MAX_ZOOM = 15
DEFAULT_TERRAIN_EXAGGERATION = 1.0
DEFAULT_3D_PITCH = 60.0
DEFAULT_3D_BEARING = 20.0
DEFAULT_MAP_LOADING_MESSAGE = "Loading Terrain and WorldCover Data..."
RSSI_MAP_LOADING_MESSAGE = "Performing RSSI and LOS Calculations"
RSSI_RENDER_MODE_OPTIONS = [
    {"label": "Show max RSSI", "value": "max-rssi"},
    {"label": "Color by strongest node", "value": "best-node"},
]
SATELLITE_TILE_SOURCE = [
    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
]

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

EARTH_RADIUS_M = 6_371_000.0
STANDARD_REFRACTION_K_FACTOR = 4.0 / 3.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * STANDARD_REFRACTION_K_FACTOR
PATH_PROFILE_CACHE_MAXSIZE = 8192
RSSI_DASK_MAX_WORKERS = 8
VIEWSHED_DASK_MAX_WORKERS = 8


def maybe_njit(*args, **kwargs):
    def decorator(func):
        if njit is None:
            return func
        return njit(*args, **kwargs)(func)

    return decorator


def build_worldcover_lookup(mapping):
    lookup = np.zeros(256, dtype=np.float32)
    for code, value in mapping.items():
        index = int(code)
        if 0 <= index < lookup.shape[0]:
            lookup[index] = float(value)
    return lookup


OFFSET_LOOKUP = build_worldcover_lookup(OFFSETS)
ATTENUATION_LOOKUP = build_worldcover_lookup(ATTENUATION)


def worldcover_lookup(values, lookup):
    classes = np.asarray(values, dtype=np.int16)
    clipped = np.clip(classes, 0, lookup.shape[0] - 1).astype(np.intp, copy=False)
    return lookup[clipped]


def V_OFFSET(values):
    return worldcover_lookup(values, OFFSET_LOOKUP)


def V_ATTENUATION(values):
    return worldcover_lookup(values, ATTENUATION_LOOKUP)

ANALYSIS_CONTEXT = {}
ANALYSIS_KEY = None
RSSI_OVERLAY_CACHE = {}
RSSI_PROGRESS_STATE = {}
RSSI_PROGRESS_LOCK = threading.Lock()
VIEWSHED_ASSESSMENT_CACHE = {}
TERRAIN_DEM_TOKENS = {}
DESKTOP_NOTIFIER = DesktopNotifier(app_name="Mesh Terrain") if DesktopNotifier is not None else None


class SuppressReloadHashFilter(logging.Filter):
    def filter(self, record):
        return "GET /_reload-hash" not in record.getMessage()


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


def point_path_signature(point_path_data):
    if not point_path_data:
        return "point:none"

    source_node_id = str(point_path_data.get("source_node_id") or "")
    target_longitude = point_path_data.get("target_longitude")
    target_latitude = point_path_data.get("target_latitude")
    if target_longitude is None or target_latitude is None:
        return f"point:{source_node_id}:invalid"
    return (
        f"point:{source_node_id}:"
        f"{float(target_longitude):.6f}:"
        f"{float(target_latitude):.6f}"
    )


def viewshed_point_signature(point_data):
    if not point_data:
        return "viewshed:none"
    longitude = point_data.get("longitude")
    latitude = point_data.get("latitude")
    if longitude is None or latitude is None:
        return "viewshed:invalid"
    return f"viewshed:{float(longitude):.6f}:{float(latitude):.6f}"


def viewshed_assessment_signature(assessment_store):
    if not assessment_store:
        return "viewshed-assessment:none"
    cache_key = assessment_store.get("cache_key")
    if cache_key:
        return f"viewshed-assessment:{cache_key}"
    return "viewshed-assessment:pending"


def native_map_overlay_context_key(
    nodes,
    selected_node_ids=None,
    point_path_data=None,
    viewshed_point_data=None,
    viewshed_assessment_store=None,
):
    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    parts = ["nodes"]
    parts.extend(node_signature(node) for node in normalized_nodes)
    parts.append("selected:" + ",".join(str(node_id) for node_id in selected_node_ids))
    parts.append(point_path_signature(point_path_data))
    parts.append(viewshed_point_signature(viewshed_point_data))
    parts.append(viewshed_assessment_signature(viewshed_assessment_store))
    return "||".join(parts)


def normalize_visual_context_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            str(key): normalize_visual_context_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_visual_context_value(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(float(value), 6)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def normalized_context_bbox(bbox_data):
    if not bbox_data:
        return None
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(
        bbox_data["min_lon"],
        bbox_data["min_lat"],
        bbox_data["max_lon"],
        bbox_data["max_lat"],
    )
    return {
        "min_lon": round(float(min_lon), 6),
        "min_lat": round(float(min_lat), 6),
        "max_lon": round(float(max_lon), 6),
        "max_lat": round(float(max_lat), 6),
    }


def normalize_bbox_resolution_m(resolution_m):
    if resolution_m is None:
        resolution_m = DEFAULT_BBOX_RESOLUTION_M
    try:
        resolution_m = float(resolution_m)
    except (TypeError, ValueError):
        resolution_m = DEFAULT_BBOX_RESOLUTION_M
    if not np.isfinite(resolution_m):
        resolution_m = DEFAULT_BBOX_RESOLUTION_M
    return float(np.clip(resolution_m, MIN_BBOX_RESOLUTION_M, MAX_BBOX_RESOLUTION_M))


def bundle_cache_key(min_lon, min_lat, max_lon, max_lat, target_resolution_m=DEFAULT_BBOX_RESOLUTION_M):
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(min_lon, min_lat, max_lon, max_lat)
    return (
        float(min_lon),
        float(min_lat),
        float(max_lon),
        float(max_lat),
        normalize_bbox_resolution_m(target_resolution_m),
    )


def native_map_visual_context_key(
    bbox_data,
    bbox_resolution_m,
    terrain_alpha,
    terrain_clip_range,
    worldcover_display,
    worldcover_opacity,
    base_map_style,
    elevation_colormap,
    rssi_calculation_store,
    rssi_overlay_selection_store,
    rssi_opacity,
    rssi_colormap,
    rssi_render_mode,
    viewshed_visual_settings=None,
):
    payload = {
        "bbox": normalized_context_bbox(bbox_data),
        "bbox_resolution_m": normalize_visual_context_value(normalize_bbox_resolution_m(bbox_resolution_m)),
        "terrain_alpha": normalize_visual_context_value(terrain_alpha),
        "terrain_clip_range": normalize_visual_context_value(terrain_clip_range),
        "worldcover_enabled": "enabled" in (worldcover_display or []),
        "worldcover_opacity": normalize_visual_context_value(worldcover_opacity),
        "base_map_style": str(base_map_style or "satellite"),
        "elevation_colormap": str(elevation_colormap or "Magma"),
        "rssi_calculation_store": normalize_visual_context_value(rssi_calculation_store),
        "rssi_overlay_selection_store": normalize_visual_context_value(rssi_overlay_selection_store or {}),
        "rssi_opacity": normalize_visual_context_value(rssi_opacity),
        "rssi_colormap": str(rssi_colormap or "Turbo"),
        "rssi_render_mode": str(rssi_render_mode or "max-rssi"),
        "viewshed_visual_settings": normalize_visual_context_value(viewshed_visual_settings or {}),
    }
    return json.dumps(payload, separators=(",", ":"))


def fetch_image_href(meta_url):
    response = requests.get(meta_url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    href = payload.get("href")
    if not href:
        raise ValueError("Image service response did not include an href.")
    return href


def fetch_binary_with_headers(url, headers=None, timeout=240):
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.content


@contextmanager
def open_remote_raster(url):
    data = fetch_binary_with_headers(url, timeout=240)
    with MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            yield dataset


def resolved_raster_transform(dataset, min_x, min_y, max_x, max_y):
    transform = dataset.transform
    identity = rasterio.transform.Affine.identity()
    if transform is None or tuple(transform) == tuple(identity):
        return rasterio.transform.from_bounds(float(min_x), float(min_y), float(max_x), float(max_y), dataset.width, dataset.height)
    return transform


def safe_reproject(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        reproject(**kwargs)


def ensure_dask_stream_memmap_dir():
    os.makedirs(DASK_STREAM_MEMMAP_DIR, exist_ok=True)
    return DASK_STREAM_MEMMAP_DIR


def data_array_memmap_path(bundle, key, dtype):
    dtype = np.dtype(dtype)
    token = hashlib.sha1(
        repr((bundle.get("cache_key"), str(key), dtype.str)).encode("utf-8")
    ).hexdigest()
    return os.path.join(ensure_dask_stream_memmap_dir(), f"{token}.dat")


def materialize_data_array_values(data_array, dtype=None, memmap_path=None):
    data = data_array.data
    target_dtype = np.dtype(dtype or data_array.dtype)
    if dask.is_dask_collection(data):
        shape = tuple(int(size) for size in data_array.shape)
        nbytes = int(np.prod(shape, dtype=np.int64)) * int(target_dtype.itemsize)
        if memmap_path and nbytes >= int(DASK_STREAM_MEMMAP_MIN_BYTES):
            array = np.memmap(memmap_path, mode="w+", dtype=target_dtype, shape=shape)
            da.store(da.asarray(data).astype(target_dtype, copy=False), array, lock=False, compute=True)
            array.flush()
            return array
        data = da.asarray(data).astype(target_dtype, copy=False).compute()
    array = np.asarray(data)
    if array.dtype != target_dtype:
        array = array.astype(target_dtype, copy=False)
    return array


def cached_bundle_data_array_values(bundle, key, dtype=None):
    cache_key = f"{key}_values_cache"
    cached = bundle.get(cache_key)
    if cached is not None:
        if dtype is None or cached.dtype == np.dtype(dtype):
            return cached
    memmap_path = None
    if dask.is_dask_collection(bundle[key].data):
        memmap_path = data_array_memmap_path(bundle, key, dtype or bundle[key].dtype)
        bundle[f"{key}_values_memmap_path"] = memmap_path
    values = materialize_data_array_values(bundle[key], dtype=dtype, memmap_path=memmap_path)
    bundle[cache_key] = values
    return values


def cached_bundle_eager_data_array(bundle, key, dtype=None):
    cache_key = f"{key}_eager_cache"
    cached = bundle.get(cache_key)
    if cached is not None:
        if dtype is None or cached.dtype == np.dtype(dtype):
            return cached
    source = bundle[key]
    if not dask.is_dask_collection(source.data) and (dtype is None or source.dtype == np.dtype(dtype)):
        bundle[cache_key] = source
        return source
    values = cached_bundle_data_array_values(bundle, key, dtype=dtype)
    eager = source.copy(data=values)
    bundle[cache_key] = eager
    return eager


def raster_window_from_bounds(min_x, min_y, max_x, max_y, transform_affine, width, height, padding=1):
    raw_window = rasterio.windows.from_bounds(
        float(min_x),
        float(min_y),
        float(max_x),
        float(max_y),
        transform=transform_affine,
    )
    col_start = max(int(math.floor(raw_window.col_off)) - int(padding), 0)
    row_start = max(int(math.floor(raw_window.row_off)) - int(padding), 0)
    col_stop = min(int(math.ceil(raw_window.col_off + raw_window.width)) + int(padding), int(width))
    row_stop = min(int(math.ceil(raw_window.row_off + raw_window.height)) + int(padding), int(height))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return rasterio.windows.Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def subset_raster_for_bounds(values, transform_affine, min_x, min_y, max_x, max_y, padding=1):
    array = np.asarray(values)
    if array.ndim != 2:
        return array, transform_affine

    window = raster_window_from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform_affine,
        array.shape[1],
        array.shape[0],
        padding=padding,
    )
    if window is None:
        return array, transform_affine

    row_start = int(window.row_off)
    row_stop = row_start + int(window.height)
    col_start = int(window.col_off)
    col_stop = col_start + int(window.width)
    subset = array[row_start:row_stop, col_start:col_stop]
    if subset.size == 0:
        return array, transform_affine
    return subset, rasterio.windows.transform(window, transform_affine)


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


def nearest_axis_index(axis, value):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size == 0:
        raise ValueError("Axis must contain at least one value.")
    if axis.size == 1:
        return 0

    target = float(value)
    ascending = bool(axis[0] <= axis[-1])
    search_axis = axis if ascending else -axis
    search_value = target if ascending else -target
    insert_index = int(np.searchsorted(search_axis, search_value, side="left"))
    if insert_index <= 0:
        return 0
    if insert_index >= axis.size:
        return int(axis.size - 1)

    left_index = insert_index - 1
    right_index = insert_index
    left_distance = abs(target - float(axis[left_index]))
    right_distance = abs(float(axis[right_index]) - target)
    return int(left_index if left_distance <= right_distance else right_index)


def normalize_bbox(min_lon, min_lat, max_lon, max_lat):
    values = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("Bounding box min values must be smaller than max values.")
    return tuple(round(value, 6) for value in values)


def bbox_dict(bounds):
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    )
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


def next_revision(current_revision):
    return int(current_revision or 0) + 1


def normalize_rssi_path_sample_spacing(sample_spacing_m):
    return normalize_bbox_resolution_m(sample_spacing_m)


def click_mode_value(click_mode):
    return str((click_mode or {}).get("mode", "none"))


def toggle_click_mode_state(click_mode, target_mode):
    mode = click_mode_value(click_mode)
    if mode == str(target_mode):
        return {"mode": "none", "node_id": None}
    return {"mode": str(target_mode), "node_id": None}


def click_mode_button_copy(click_mode, target_mode, active_label, active_status, inactive_label, inactive_status):
    if click_mode_value(click_mode) == str(target_mode):
        return active_label, active_status
    return inactive_label, inactive_status


def fit_bbox_for_nodes(nodes, padding_fraction=0.12, min_padding_deg=0.01):
    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    if not normalized_nodes:
        return None

    longitudes = [float(node["longitude"]) for node in normalized_nodes]
    latitudes = [float(node["latitude"]) for node in normalized_nodes]
    min_lon = min(longitudes)
    max_lon = max(longitudes)
    min_lat = min(latitudes)
    max_lat = max(latitudes)

    lon_span = max(max_lon - min_lon, min_padding_deg)
    lat_span = max(max_lat - min_lat, min_padding_deg)
    lon_pad = max(lon_span * padding_fraction, min_padding_deg)
    lat_pad = max(lat_span * padding_fraction, min_padding_deg)

    return {
        "min_lon": round(min_lon - lon_pad, 6),
        "min_lat": round(min_lat - lat_pad, 6),
        "max_lon": round(max_lon + lon_pad, 6),
        "max_lat": round(max_lat + lat_pad, 6),
    }


def point_in_bbox(longitude, latitude, bbox):
    if not bbox:
        return False
    return (
        float(bbox["min_lon"]) <= float(longitude) <= float(bbox["max_lon"])
        and float(bbox["min_lat"]) <= float(latitude) <= float(bbox["max_lat"])
    )


def nodes_outside_loaded_bbox(nodes, bbox):
    outside = []
    for node in nodes or []:
        if not point_in_bbox(node["longitude"], node["latitude"], bbox):
            outside.append(str(node["name"]))
    return outside


def clip_mercator_lat(latitude):
    return max(min(float(latitude), 85.05112878), -85.05112878)


def lon_to_tile_x(longitude, zoom):
    return (float(longitude) + 180.0) / 360.0 * (2**zoom)


def lat_to_tile_y(latitude, zoom):
    lat_rad = math.radians(clip_mercator_lat(latitude))
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2**zoom)


def choose_osm_zoom(min_lon, min_lat, max_lon, max_lat):
    for zoom in range(OSM_MAX_ZOOM, OSM_MIN_ZOOM - 1, -1):
        x0 = int(math.floor(lon_to_tile_x(min_lon, zoom)))
        x1 = int(math.floor(lon_to_tile_x(max_lon, zoom)))
        y0 = int(math.floor(lat_to_tile_y(max_lat, zoom)))
        y1 = int(math.floor(lat_to_tile_y(min_lat, zoom)))
        x_count = max(1, x1 - x0 + 1)
        y_count = max(1, y1 - y0 + 1)
        if x_count * y_count <= OSM_MAX_TILE_COUNT and x_count <= 8 and y_count <= 8:
            return zoom
    return OSM_MIN_ZOOM


def extent_to_pixel_shape(min_x, min_y, max_x, max_y, max_dim=DISPLAY_MAX_DIM, min_dim=DISPLAY_MIN_DIM):
    x_span = max(float(max_x) - float(min_x), 1e-6)
    y_span = max(float(max_y) - float(min_y), 1e-6)
    if x_span >= y_span:
        width = int(max_dim)
        height = max(int(min_dim), int(round(float(max_dim) * y_span / x_span)))
    else:
        height = int(max_dim)
        width = max(int(min_dim), int(round(float(max_dim) * x_span / y_span)))
    return width, height


def geographic_bbox_to_pixel_shape(min_lon, min_lat, max_lon, max_lat, max_dim=DISPLAY_MAX_DIM, min_dim=DISPLAY_MIN_DIM):
    return extent_to_pixel_shape(min_lon, min_lat, max_lon, max_lat, max_dim=max_dim, min_dim=min_dim)


@lru_cache(maxsize=16)
def default_geographic_pixel_size(target_projected_pixel_size_m=DEFAULT_BBOX_RESOLUTION_M):
    target_projected_pixel_size_m = normalize_bbox_resolution_m(target_projected_pixel_size_m)
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(
        DEFAULT_BBOX["min_lon"],
        DEFAULT_BBOX["min_lat"],
        DEFAULT_BBOX["max_lon"],
        DEFAULT_BBOX["max_lat"],
    )
    min_x, min_y, max_x, max_y = transform_bounds(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    x_span_m = max((max_x - min_x), 1e-6)
    y_span_m = max((max_y - min_y), 1e-6)
    width = max(int(math.ceil(x_span_m / target_projected_pixel_size_m)), 1)
    height = max(int(math.ceil(y_span_m / target_projected_pixel_size_m)), 1)
    return (
        max((max_lon - min_lon) / max(width, 1), 1e-9),
        max((max_lat - min_lat) / max(height, 1), 1e-9),
    )


@lru_cache(maxsize=16)
def default_projected_pixel_size(target_projected_pixel_size_m=DEFAULT_BBOX_RESOLUTION_M):
    target_projected_pixel_size_m = normalize_bbox_resolution_m(target_projected_pixel_size_m)
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(
        DEFAULT_BBOX["min_lon"],
        DEFAULT_BBOX["min_lat"],
        DEFAULT_BBOX["max_lon"],
        DEFAULT_BBOX["max_lat"],
    )
    min_x, min_y, max_x, max_y = transform_bounds(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    x_span_m = max((max_x - min_x), 1e-6)
    y_span_m = max((max_y - min_y), 1e-6)
    width = max(int(math.ceil(x_span_m / target_projected_pixel_size_m)), 1)
    height = max(int(math.ceil(y_span_m / target_projected_pixel_size_m)), 1)
    return (
        max((max_x - min_x) / max(width, 1), 1e-6),
        max((max_y - min_y) / max(height, 1), 1e-6),
    )


def target_pixel_shape_for_extent(min_x, min_y, max_x, max_y, pixel_size, max_dim=DISPLAY_MAX_DIM, min_dim=DISPLAY_MIN_DIM):
    x_span = max(float(max_x) - float(min_x), 1e-9)
    y_span = max(float(max_y) - float(min_y), 1e-9)
    pixel_size_x, pixel_size_y = pixel_size
    width = int(math.ceil(x_span / max(float(pixel_size_x), 1e-9)))
    height = int(math.ceil(y_span / max(float(pixel_size_y), 1e-9)))
    return max(width, 1), max(height, 1)


def mercator_axes_to_geographic(x_axis, y_axis):
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    if x_axis.size:
        lon_axis, _ = transform(
            WEB_MERCATOR_CRS,
            GEOGRAPHIC_CRS,
            x_axis.tolist(),
            [0.0] * int(x_axis.size),
        )
        lon_axis = np.asarray(lon_axis, dtype=np.float64)
    else:
        lon_axis = np.empty(0, dtype=np.float64)
    if y_axis.size:
        _, lat_axis = transform(
            WEB_MERCATOR_CRS,
            GEOGRAPHIC_CRS,
            [0.0] * int(y_axis.size),
            y_axis.tolist(),
        )
        lat_axis = np.asarray(lat_axis, dtype=np.float64)
    else:
        lat_axis = np.empty(0, dtype=np.float64)
    return lon_axis, lat_axis


def split_dimension(total, max_chunk):
    total = int(total)
    max_chunk = int(max_chunk)
    if total <= 0:
        return []
    chunk_count = max(1, int(math.ceil(total / max(max_chunk, 1))))
    base = total // chunk_count
    remainder = total % chunk_count
    return [base + (1 if index < remainder else 0) for index in range(chunk_count)]


def raster_request_specs(min_x, min_y, max_x, max_y, width, height, max_request_dim=DISPLAY_MAX_DIM):
    width_splits = split_dimension(width, max_request_dim)
    height_splits = split_dimension(height, max_request_dim)
    x_span = float(max_x) - float(min_x)
    y_span = float(max_y) - float(min_y)
    total_width = max(int(width), 1)
    total_height = max(int(height), 1)

    specs = []
    row_offset = 0
    for chunk_height in height_splits:
        next_row_offset = row_offset + chunk_height
        chunk_max_y = float(max_y) - (row_offset / total_height) * y_span
        chunk_min_y = float(max_y) - (next_row_offset / total_height) * y_span
        col_offset = 0
        for chunk_width in width_splits:
            next_col_offset = col_offset + chunk_width
            chunk_min_x = float(min_x) + (col_offset / total_width) * x_span
            chunk_max_x = float(min_x) + (next_col_offset / total_width) * x_span
            specs.append(
                {
                    "min_x": chunk_min_x,
                    "min_y": chunk_min_y,
                    "max_x": chunk_max_x,
                    "max_y": chunk_max_y,
                    "width": int(chunk_width),
                    "height": int(chunk_height),
                    "row_offset": int(row_offset),
                    "col_offset": int(col_offset),
                }
            )
            col_offset = next_col_offset
        row_offset = next_row_offset
    return specs


def span_to_sample_dim(span_m, target_resolution_m, min_dim, max_dim):
    target_resolution_m = max(float(target_resolution_m), 1e-6)
    sample_dim = int(math.ceil(float(span_m) / target_resolution_m))
    return int(np.clip(sample_dim, int(min_dim), int(max_dim)))


def elevation_export_image_href(min_x, min_y, max_x, max_y, crs, width, height):
    meta_url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        f"?bbox={float(min_x)},{float(min_y)},{float(max_x)},{float(max_y)}"
        f"&bboxSR={str(crs).split(':')[-1]}"
        f"&imageSR={str(crs).split(':')[-1]}"
        f"&size={int(width)},{int(height)}"
        "&format=tiff"
        "&pixelType=F32"
        "&interpolation=RSP_BilinearInterpolation"
        "&f=json"
    )
    return fetch_image_href(meta_url)


def fetch_elevation_raster_mosaic(min_x, min_y, max_x, max_y, crs, width, height):
    width = max(int(width), 1)
    height = max(int(height), 1)
    full_transform = rasterio.transform.from_bounds(float(min_x), float(min_y), float(max_x), float(max_y), width, height)
    mosaic = np.full((height, width), np.nan, dtype=np.float32)

    for spec in raster_request_specs(min_x, min_y, max_x, max_y, width, height):
        href = elevation_export_image_href(
            spec["min_x"],
            spec["min_y"],
            spec["max_x"],
            spec["max_y"],
            crs,
            spec["width"],
            spec["height"],
        )
        with open_remote_raster(href) as src:
            source = src.read(1).astype(np.float32)
            src_transform = rasterio.transform.from_bounds(
                spec["min_x"],
                spec["min_y"],
                spec["max_x"],
                spec["max_y"],
                src.width,
                src.height,
            )
            window = rasterio.windows.Window(
                col_off=spec["col_offset"],
                row_off=spec["row_offset"],
                width=spec["width"],
                height=spec["height"],
            )
            destination = mosaic[
                spec["row_offset"]:spec["row_offset"] + spec["height"],
                spec["col_offset"]:spec["col_offset"] + spec["width"],
            ]
            safe_reproject(
                source=source,
                destination=destination,
                src_transform=src_transform,
                src_crs=crs,
                src_nodata=np.nan,
                dst_transform=rasterio.windows.transform(window, full_transform),
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    return mosaic


def raster_layer_coordinates(bbox_data):
    return [
        [float(bbox_data["min_lon"]), float(bbox_data["max_lat"])],
        [float(bbox_data["max_lon"]), float(bbox_data["max_lat"])],
        [float(bbox_data["max_lon"]), float(bbox_data["min_lat"])],
        [float(bbox_data["min_lon"]), float(bbox_data["min_lat"])],
    ]


def feature_collection(features=None):
    return {
        "type": "FeatureCollection",
        "features": list(features or []),
    }


def build_maplibre_style(base_map_style):
    style_name = str(base_map_style or "satellite")
    if style_name == "satellite":
        return {
            "version": 8,
            "glyphs": MAPLIBRE_GLYPHS_URL,
            "sources": {
                "basemap": {
                    "type": "raster",
                    "tiles": SATELLITE_TILE_SOURCE,
                    "tileSize": 256,
                    "attribution": "United States Geological Survey",
                    "maxzoom": 18,
                }
            },
            "layers": [{"id": "basemap", "type": "raster", "source": "basemap"}],
        }

    return {
        "version": 8,
        "glyphs": MAPLIBRE_GLYPHS_URL,
        "sources": {
            "basemap": {
                "type": "raster",
                "tiles": [OSM_TILE_URL],
                "tileSize": 256,
                "attribution": "© OpenStreetMap contributors",
                "maxzoom": 19,
            }
        },
        "layers": [{"id": "basemap", "type": "raster", "source": "basemap"}],
    }


def colorscale_gradient_css(colorscale_name, sample_count=7):
    colors = sample_colorscale(colorscale_name, np.linspace(0.0, 1.0, sample_count))
    return "linear-gradient(90deg, " + ", ".join(colors) + ")"


def build_gradient_legend_card(title, colorscale_name, minimum, maximum):
    return html.Div(
        [
            html.Div(title, className="native-map-legend-title"),
            html.Div(
                className="native-map-legend-gradient",
                style={"background": colorscale_gradient_css(colorscale_name)},
            ),
            html.Div(
                [
                    html.Span(f"{float(minimum):.0f}"),
                    html.Span(f"{float(maximum):.0f}"),
                ],
                className="native-map-legend-range",
            ),
        ],
        className="native-map-legend-card",
    )


def build_discrete_legend_card(title, items):
    return html.Div(
        [
            html.Div(title, className="native-map-legend-title"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="native-map-legend-swatch",
                                style={"backgroundColor": str(item["color"])},
                            ),
                            html.Span(str(item["label"])),
                        ],
                        className="native-map-legend-item",
                    )
                    for item in (items or [])
                ],
                className="native-map-legend-list",
            ),
        ],
        className="native-map-legend-card",
    )


def build_native_map_legends(terrain_legend=None, worldcover_legend=None, rssi_legend=None, viewshed_legend=None):
    cards = []
    if terrain_legend is not None:
        cards.append(
            build_gradient_legend_card(
                "Elevation (m)",
                terrain_legend["colorscale"],
                terrain_legend["min"],
                terrain_legend["max"],
            )
        )
    if worldcover_legend is not None:
        cards.append(
            build_discrete_legend_card(
                str(worldcover_legend.get("title") or "WorldCover"),
                worldcover_legend.get("items") or [],
            )
        )
    if rssi_legend is not None:
        if rssi_legend.get("type") == "categorical":
            cards.append(
                build_discrete_legend_card(
                    str(rssi_legend.get("title") or "Best Serving Node"),
                    rssi_legend.get("items") or [],
                )
            )
        else:
            cards.append(
                build_gradient_legend_card(
                    str(rssi_legend.get("title") or "Max RSSI (dBm)"),
                    rssi_legend["colorscale"],
                    rssi_legend["min"],
                    rssi_legend["max"],
                )
            )
    if viewshed_legend is not None:
        cards.append(
            build_gradient_legend_card(
                str(viewshed_legend.get("title") or VIEWSHED_LEGEND_TITLE),
                viewshed_legend["colorscale"],
                viewshed_legend["min"],
                viewshed_legend["max"],
            )
        )
    if not cards:
        return html.Div()
    stack_class = (
        "native-map-legend-stack native-map-legend-stack--bottom-right"
        if worldcover_legend is not None or viewshed_legend is not None
        else "native-map-legend-stack"
    )
    return html.Div(cards, className=stack_class)


def build_node_feature_collection(nodes, selected_node_ids):
    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    outline_colors, outline_widths = node_marker_outline_arrays(normalized_nodes, selected_node_ids)

    features = []
    for index, node in enumerate(normalized_nodes):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(node["longitude"]), float(node["latitude"])],
                },
                "properties": {
                    "kind": "node",
                    "node_id": str(node["id"]),
                    "name": str(node["name"]),
                    "color": node_color(index),
                    "outline_color": outline_colors[index],
                    "outline_width": float(outline_widths[index]),
                },
            }
        )
    return feature_collection(features)


def build_viewshed_point_feature_collection(point_data):
    if not point_data:
        return feature_collection()
    longitude = point_data.get("longitude")
    latitude = point_data.get("latitude")
    if longitude is None or latitude is None:
        return feature_collection()
    return feature_collection(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(longitude), float(latitude)],
                },
                "properties": {
                    "kind": "viewshed-point",
                    "name": "Viewshed Assessment Point",
                    "color": VIEWSHED_POINT_COLOR,
                    "outline_color": VIEWSHED_POINT_OUTLINE_COLOR,
                },
            }
        ]
    )


def build_viewshed_radius_feature_collection(point_data, radius_m, segments=96):
    if not point_data:
        return feature_collection()
    longitude = point_data.get("longitude")
    latitude = point_data.get("latitude")
    if longitude is None or latitude is None:
        return feature_collection()
    radius_m = float(radius_m or 0.0)
    if radius_m <= 0.0:
        return feature_collection()

    center_x, center_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        [float(longitude)],
        [float(latitude)],
    )
    center_x = float(center_x[0])
    center_y = float(center_y[0])
    ring_angles = np.linspace(0.0, 2.0 * math.pi, int(max(12, segments)) + 1, dtype=np.float64)
    ring_x = center_x + radius_m * np.cos(ring_angles)
    ring_y = center_y + radius_m * np.sin(ring_angles)
    ring_lon, ring_lat = transform(
        PROJECTED_CRS,
        GEOGRAPHIC_CRS,
        ring_x.tolist(),
        ring_y.tolist(),
    )
    coordinates = [[float(lon), float(lat)] for lon, lat in zip(ring_lon, ring_lat, strict=False)]
    return feature_collection(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "kind": "viewshed-radius",
                    "radius_m": radius_m,
                    "color": VIEWSHED_POINT_COLOR,
                    "outline_color": VIEWSHED_POINT_OUTLINE_COLOR,
                },
            }
        ]
    )


def build_viewshed_sample_feature_collection(point_data, radius_m, sample_count):
    if not point_data:
        return feature_collection()
    longitude = point_data.get("longitude")
    latitude = point_data.get("latitude")
    if longitude is None or latitude is None:
        return feature_collection()
    radius_m = float(radius_m or 0.0)
    sample_count = effective_viewshed_sample_count(sample_count)
    if radius_m <= 0.0 or sample_count <= 0:
        return feature_collection()

    center_x, center_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        [float(longitude)],
        [float(latitude)],
    )
    sample_points = generate_circular_samples(float(center_x[0]), float(center_y[0]), radius_m, sample_count)
    sample_lon, sample_lat = transform(
        PROJECTED_CRS,
        GEOGRAPHIC_CRS,
        sample_points[:, 0].tolist(),
        sample_points[:, 1].tolist(),
    )
    features = []
    for sample_index, (sample_lon_value, sample_lat_value) in enumerate(zip(sample_lon, sample_lat, strict=False)):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(sample_lon_value), float(sample_lat_value)],
                },
                "properties": {
                    "kind": "viewshed-sample",
                    "index": int(sample_index),
                    "color": VIEWSHED_POINT_COLOR,
                    "outline_color": VIEWSHED_POINT_OUTLINE_COLOR,
                },
            }
        )
    return feature_collection(features)


def build_loaded_bbox_feature_collection(loaded_bbox):
    if not loaded_bbox:
        return feature_collection()
    return feature_collection(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(loaded_bbox["min_lon"]), float(loaded_bbox["max_lat"])],
                        [float(loaded_bbox["max_lon"]), float(loaded_bbox["max_lat"])],
                        [float(loaded_bbox["max_lon"]), float(loaded_bbox["min_lat"])],
                        [float(loaded_bbox["min_lon"]), float(loaded_bbox["min_lat"])],
                        [float(loaded_bbox["min_lon"]), float(loaded_bbox["max_lat"])],
                    ],
                },
                "properties": {
                    "kind": "loaded-bbox",
                    "color": "#38bdf8",
                },
            }
        ]
    )


def sample_grid_value(values, lon_axis, lat_axis, longitude, latitude):
    array = np.asarray(values)
    lon_axis = np.asarray(lon_axis, dtype=np.float64)
    lat_axis = np.asarray(lat_axis, dtype=np.float64)
    if array.ndim != 2 or lon_axis.size == 0 or lat_axis.size == 0:
        return None

    column_index = nearest_axis_index(lon_axis, longitude)
    row_index = nearest_axis_index(lat_axis, latitude)
    sample = float(array[row_index, column_index])
    if not np.isfinite(sample):
        return None
    return sample


def build_native_path_line_feature(start_lon, start_lat, end_lon, end_lat, color, dashed=False, result=None):
    _ = result
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(start_lon), float(start_lat)], [float(end_lon), float(end_lat)]],
        },
        "properties": {
            "kind": "path-line",
            "color": color,
            "dashed": bool(dashed),
        },
    }


def empty_native_path_overlay():
    return {
        "line": feature_collection(),
        "attenuation_points": feature_collection(),
        "terrain_block_points": feature_collection(),
    }


def build_native_path_overlay(start_lon, start_lat, end_lon, end_lat, color, dashed=False, result=None):
    line = feature_collection(
        [build_native_path_line_feature(start_lon, start_lat, end_lon, end_lat, color, dashed=dashed, result=result)]
    )
    if result is None:
        attenuation_points = feature_collection()
        terrain_block_points = feature_collection()
    else:
        attenuation_points, terrain_block_points = build_native_path_event_feature_collections(
            result,
            start_lon,
            start_lat,
            end_lon,
            end_lat,
        )
    return {
        "line": line,
        "attenuation_points": attenuation_points,
        "terrain_block_points": terrain_block_points,
    }


def build_path_sample_fractions(result):
    distance_along = np.asarray(result.get("distance_along_km", []), dtype=np.float64)
    if distance_along.size:
        total_distance = float(distance_along[-1])
        if total_distance > 1e-9:
            return np.clip(distance_along / total_distance, 0.0, 1.0)
        if distance_along.size == 1:
            return np.array([0.0], dtype=np.float64)
        return np.linspace(0.0, 1.0, distance_along.size, dtype=np.float64)

    path_longitude = np.asarray(result.get("path_longitude", []), dtype=np.float64)
    if path_longitude.size == 0:
        return np.array([], dtype=np.float64)
    if path_longitude.size == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, path_longitude.size, dtype=np.float64)


def path_coordinate_arrays(result, start_lon, start_lat, end_lon, end_lat):
    if result is not None:
        path_longitude = np.asarray(result.get("path_longitude", []), dtype=np.float64)
        path_latitude = np.asarray(result.get("path_latitude", []), dtype=np.float64)
        if path_longitude.size >= 2 and path_longitude.size == path_latitude.size:
            return path_longitude, path_latitude

    path_fraction = build_path_sample_fractions(result or {})
    if path_fraction.size:
        return interpolate_path_coordinates(
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            path_fraction,
        )

    return (
        np.asarray([float(start_lon), float(end_lon)], dtype=np.float64),
        np.asarray([float(start_lat), float(end_lat)], dtype=np.float64),
    )


def interpolate_path_coordinates(start_lon, start_lat, end_lon, end_lat, fractions):
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    longitude = float(start_lon) + (float(end_lon) - float(start_lon)) * fractions
    latitude = float(start_lat) + (float(end_lat) - float(start_lat)) * fractions
    return longitude.astype(np.float64), latitude.astype(np.float64)


def build_native_path_event_feature_collections(result, start_lon, start_lat, end_lon, end_lat):
    path_fraction = build_path_sample_fractions(result)
    blockage_fraction = np.asarray(result.get("blockage_fraction", []), dtype=np.float64)
    attenuation_per_sample_db = np.asarray(result.get("attenuation_per_sample_db", []), dtype=np.float64)
    terrain_only_blocked = np.asarray(result.get("terrain_only_blocked", []), dtype=bool)

    attenuation_features = []
    terrain_block_features = []
    if path_fraction.size == 0:
        return feature_collection(), feature_collection()
    path_longitude, path_latitude = path_coordinate_arrays(
        result,
        start_lon,
        start_lat,
        end_lon,
        end_lat,
    )
    if path_longitude.size != path_fraction.size or path_latitude.size != path_fraction.size:
        return feature_collection(), feature_collection()
    line_longitude, line_latitude = interpolate_path_coordinates(
        start_lon,
        start_lat,
        end_lon,
        end_lat,
        path_fraction,
    )

    attenuation_mask = (blockage_fraction > 0) & ~terrain_only_blocked
    for sample_index in np.flatnonzero(attenuation_mask):
        blocked_fraction = blockage_fraction[sample_index]
        added_loss_db = attenuation_per_sample_db[sample_index]
        attenuation_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(path_longitude[sample_index]), float(path_latitude[sample_index])],
                },
                "properties": {
                    "kind": "attenuation-event",
                    "blocked_fraction": float(blocked_fraction),
                    "added_loss_db": float(added_loss_db),
                    "sample_fraction": float(path_fraction[sample_index]),
                    "radius": float(np.clip(4.5 + 5.0 * blocked_fraction, 4.5, 9.0)),
                    "color": ATTENUATION_EVENT_COLOR,
                },
            }
        )

    for sample_index in np.flatnonzero(terrain_only_blocked):
        added_loss_db = attenuation_per_sample_db[sample_index]
        terrain_block_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(line_longitude[sample_index]), float(line_latitude[sample_index])],
                },
                "properties": {
                    "kind": "terrain-los-block",
                    "added_loss_db": float(added_loss_db),
                    "sample_fraction": float(path_fraction[sample_index]),
                    "radius": 6.5,
                    "color": TERRAIN_BLOCK_COLOR,
                },
            }
        )

    return feature_collection(attenuation_features), feature_collection(terrain_block_features)


def compute_native_map_path_overlay(nodes, selected_node_ids, point_path_data, bundle, global_rx_height_agl, global_rx_gain_dbi):
    nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in nodes])
    if bundle is not None:
        ensure_analysis_context(bundle)

    if point_path_data:
        source = find_node(nodes, point_path_data.get("source_node_id"))
        target_lon = point_path_data.get("target_longitude")
        target_lat = point_path_data.get("target_latitude")
        if source is None or target_lon is None or target_lat is None:
            return empty_native_path_overlay()
        if bundle is None:
            return build_native_path_overlay(
                source["longitude"],
                source["latitude"],
                float(target_lon),
                float(target_lat),
                PATH_PREVIEW_COLOR,
            )
        try:
            result = compute_path_loss(
                (source["longitude"], source["latitude"]),
                (float(target_lon), float(target_lat)),
                tx_height=source["height_agl_m"],
                rx_height=float(global_rx_height_agl or DEFAULT_GLOBAL_RX_HEIGHT_M),
                tx_power_dbm=source["tx_power_dbm"],
                tx_gain_dbi=source["antenna_gain_dbi"],
                rx_gain_dbi=float(global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI),
            )
        except Exception:
            return build_native_path_overlay(
                source["longitude"],
                source["latitude"],
                float(target_lon),
                float(target_lat),
                PATH_BAD_COLOR,
                dashed=True,
            )
        return build_native_path_overlay(
            source["longitude"],
            source["latitude"],
            float(target_lon),
            float(target_lat),
            PATH_GOOD_COLOR if result["rssi_dbm"] > MIN_LINK_RSSI_DBM else PATH_BAD_COLOR,
            dashed=result["rssi_dbm"] <= MIN_LINK_RSSI_DBM,
            result=result,
        )

    if len(selected_node_ids) == 2:
        source = find_node(nodes, selected_node_ids[0])
        target = find_node(nodes, selected_node_ids[1])
        if source is None or target is None:
            return empty_native_path_overlay()
        if bundle is None:
            return build_native_path_overlay(
                source["longitude"],
                source["latitude"],
                target["longitude"],
                target["latitude"],
                PATH_PREVIEW_COLOR,
            )
        try:
            link_result = compute_bidirectional_link_result(source, target)
        except Exception:
            return build_native_path_overlay(
                source["longitude"],
                source["latitude"],
                target["longitude"],
                target["latitude"],
                PATH_BAD_COLOR,
                dashed=True,
            )
        return build_native_path_overlay(
            source["longitude"],
            source["latitude"],
            target["longitude"],
            target["latitude"],
            PATH_GOOD_COLOR if link_result["good_link"] else PATH_BAD_COLOR,
            dashed=not link_result["good_link"],
            result=link_result["forward_result"],
        )

    return empty_native_path_overlay()


def build_native_map_spec(
    bundle,
    nodes,
    terrain_alpha,
    terrain_clip_range=None,
    elevation_colorscale="Magma",
    worldcover_enabled=False,
    worldcover_opacity=0.0,
    viewshed_point_data=None,
    viewshed_radius_m=None,
    viewshed_sample_count=None,
    viewshed_overlay=None,
    viewshed_colorscale="Turbo",
    viewshed_opacity=0.0,
    rssi_overlay=None,
    rssi_colorscale="Turbo",
    loaded_bbox=None,
    selected_node_ids=None,
    base_map_style="satellite",
    rssi_opacity=0.55,
    path_overlay=None,
    overlay_context_key=None,
    visual_context_key=None,
):
    normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
    selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
    path_overlay = path_overlay or empty_native_path_overlay()

    terrain_layer = None
    terrain_dem = None
    terrain_legend = None
    worldcover_layer = None
    worldcover_legend = None
    viewshed_layer = None
    viewshed_legend = None
    if bundle is not None:
        terrain_bounds = bundle["terrain_display_bounds"]
        terrain_display = bundle["terrain_display"]
        terrain_valid = terrain_display[np.isfinite(terrain_display)]
        terrain_min = float(np.nanmin(terrain_valid)) if terrain_valid.size else 0.0
        terrain_max = float(np.nanmax(terrain_valid)) if terrain_valid.size else terrain_min + 1.0
        if terrain_max <= terrain_min:
            terrain_max = terrain_min + 1.0

        clip_min = terrain_min
        clip_max = terrain_max
        if terrain_clip_range and len(terrain_clip_range) == 2:
            clip_min = min(max(float(terrain_clip_range[0]), terrain_min), terrain_max)
            clip_max = min(max(float(terrain_clip_range[1]), clip_min), terrain_max)

        terrain_legend = {
            "colorscale": elevation_colorscale,
            "min": clip_min,
            "max": clip_max,
        }
        token = register_terrain_dem_bundle(bundle)
        terrain_dem = {
            "token": token,
            "tiles": [f"/terrain-dem/{token}/{{z}}/{{x}}/{{y}}.png"],
            "bounds": [
                float(terrain_bounds["min_lon"]),
                float(terrain_bounds["min_lat"]),
                float(terrain_bounds["max_lon"]),
                float(terrain_bounds["max_lat"]),
            ],
            "tile_size": TERRAIN_TILE_SIZE,
            "maxzoom": TERRAIN_MAX_ZOOM,
            "encoding": "terrarium",
            "exaggeration": DEFAULT_TERRAIN_EXAGGERATION,
            "pitch": DEFAULT_3D_PITCH,
            "bearing": DEFAULT_3D_BEARING,
            "color_relief": {
                "expression": maplibre_color_relief_expression(
                    elevation_colorscale,
                    clip_min,
                    clip_max,
                    alpha=float(terrain_alpha or 0.0),
                ),
                "opacity": 1.0,
            },
        }

        if worldcover_enabled:
            worldcover_source = bundle.get("worldcover_display")
            if worldcover_source is None:
                worldcover_source = cached_bundle_data_array_values(bundle, "worldcover_projected", dtype=np.int16)
            worldcover_values = np.asarray(worldcover_source, dtype=np.int16)
            worldcover_codes = [
                int(code)
                for code in np.unique(worldcover_values)
                if int(code) in WORLDCOVER_PALETTE and int(code) != 0
            ]
            worldcover_uri = colorize_value_map_to_png_uri(
                worldcover_values,
                {code: WORLDCOVER_PALETTE[code] for code in worldcover_codes},
                alpha=float(worldcover_opacity or 0.0),
            )
            if worldcover_uri is not None:
                worldcover_layer = {
                    "image": worldcover_uri,
                    "coordinates": raster_layer_coordinates(terrain_bounds),
                    "opacity": 1.0,
                }
                worldcover_legend = {
                    "type": "categorical",
                    "title": "WorldCover",
                    "items": [
                        {"label": WORLDCOVER_CLASSES.get(code, str(code)), "color": WORLDCOVER_PALETTE[code]}
                        for code in worldcover_codes
                    ],
                }

        if viewshed_overlay is not None:
            visible_cell_count = np.asarray(viewshed_overlay.get("visible_cell_count"), dtype=np.float32)
            finite_cell_count = visible_cell_count[np.isfinite(visible_cell_count)]
            if finite_cell_count.size:
                cell_count_min = float(np.nanmin(finite_cell_count))
                cell_count_max = float(np.nanmax(finite_cell_count))
                if cell_count_max <= cell_count_min:
                    cell_count_max = cell_count_min + 1.0
                viewshed_bounds = viewshed_overlay.get("display_bounds") or terrain_bounds
                viewshed_uri = colorize_array_to_png_uri(
                    visible_cell_count,
                    viewshed_colorscale,
                    cell_count_min,
                    cell_count_max,
                    alpha=float(viewshed_opacity or 0.0),
                )
                if viewshed_uri is not None:
                    viewshed_layer = {
                        "image": viewshed_uri,
                        "coordinates": raster_layer_coordinates(viewshed_bounds),
                        "opacity": 1.0,
                    }
                    viewshed_legend = {
                        "type": "gradient",
                        "title": VIEWSHED_LEGEND_TITLE,
                        "colorscale": viewshed_colorscale,
                        "min": cell_count_min,
                        "max": cell_count_max,
                    }

    rssi_layer = None
    rssi_legend = None
    if rssi_overlay is not None:
        overlay_mode = str(rssi_overlay.get("mode") or "max-rssi")
        if overlay_mode == "best-node":
            rssi_uri = colorize_category_array_to_png_uri(
                rssi_overlay["owner_index"],
                [item["color"] for item in rssi_overlay.get("legend_items", [])],
                alpha=float(rssi_opacity or 0.0),
            )
            rssi_legend = {
                "type": "categorical",
                "title": "Best Serving Node",
                "items": [
                    {"label": item["label"], "color": item["color"]}
                    for item in rssi_overlay.get("legend_items", [])
                ],
            }
        else:
            rssi_uri = colorize_array_to_png_uri(
                rssi_overlay["max_rssi"],
                rssi_colorscale,
                -140.0,
                -60.0,
                alpha=float(rssi_opacity or 0.0),
            )
            rssi_legend = {
                "type": "gradient",
                "title": "Max RSSI (dBm)",
                "colorscale": rssi_colorscale,
                "min": -140.0,
                "max": -60.0,
            }
        if rssi_uri is not None and bundle is not None:
            rssi_layer = {
                "image": rssi_uri,
                "coordinates": raster_layer_coordinates(bundle["terrain_display_bounds"]),
                "opacity": 1.0,
                "resampling": "nearest",
            }

    return {
        "base_style_key": str(base_map_style or "satellite"),
        "base_style": build_maplibre_style(base_map_style),
        "terrain_dem": terrain_dem,
        "terrain_layer": terrain_layer,
        "worldcover_layer": worldcover_layer,
        "viewshed_layer": viewshed_layer,
        "rssi_layer": rssi_layer,
        "nodes": build_node_feature_collection(normalized_nodes, selected_node_ids),
        "viewshed_point": build_viewshed_point_feature_collection(viewshed_point_data),
        "viewshed_radius_outline": build_viewshed_radius_feature_collection(viewshed_point_data, viewshed_radius_m),
        "viewshed_samples": build_viewshed_sample_feature_collection(viewshed_point_data, viewshed_radius_m, viewshed_sample_count),
        "loaded_bbox": build_loaded_bbox_feature_collection(loaded_bbox),
        "path_line": path_overlay.get("line", feature_collection()),
        "attenuation_points": path_overlay.get("attenuation_points", feature_collection()),
        "terrain_block_points": path_overlay.get("terrain_block_points", feature_collection()),
        "terrain_legend": terrain_legend,
        "worldcover_legend": worldcover_legend,
        "viewshed_legend": viewshed_legend,
        "rssi_legend": rssi_legend,
        "overlay_context_key": str(overlay_context_key or ""),
        "visual_context_key": str(visual_context_key or ""),
    }


def colorize_array_to_png_uri(values, colorscale_name, zmin, zmax, alpha=1.0):
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return None

    scale_range = max(float(zmax) - float(zmin), 1e-9)
    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite_mask] = np.clip((values[finite_mask] - float(zmin)) / scale_range, 0.0, 1.0)
    lut = np.array(
        [ImageColor.getcolor(color, "RGBA") for color in sample_colorscale(colorscale_name, np.linspace(0.0, 1.0, 256))],
        dtype=np.uint8,
    )
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    lut_index = np.zeros(values.shape, dtype=np.uint8)
    lut_index[finite_mask] = np.rint(normalized[finite_mask] * 255.0).astype(np.uint8)
    rgba[finite_mask] = lut[lut_index[finite_mask]]
    alpha_value = int(round(np.clip(float(alpha), 0.0, 1.0) * 255.0))
    rgba[..., 3] = np.where(finite_mask, alpha_value, 0).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def colorize_category_array_to_png_uri(values, colors, alpha=1.0):
    category_array = np.asarray(values)
    if category_array.ndim != 2 or not colors:
        return None

    alpha_value = int(round(np.clip(float(alpha), 0.0, 1.0) * 255.0))
    if alpha_value <= 0:
        return None

    valid_mask = np.isfinite(category_array) & (category_array >= 0)
    if not np.any(valid_mask):
        return None

    rgba = np.zeros(category_array.shape + (4,), dtype=np.uint8)
    palette = np.array([ImageColor.getcolor(str(color), "RGBA") for color in colors], dtype=np.uint8)
    safe_category_array = np.where(valid_mask, category_array, 0).astype(np.int32)
    clipped_index = np.clip(safe_category_array, 0, len(palette) - 1)
    rgba[valid_mask] = palette[clipped_index[valid_mask]]
    rgba[..., 3] = np.where(valid_mask, alpha_value, 0).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def colorize_value_map_to_png_uri(values, value_to_color, alpha=1.0):
    category_array = np.asarray(values)
    if category_array.ndim != 2 or not value_to_color:
        return None

    alpha_value = int(round(np.clip(float(alpha), 0.0, 1.0) * 255.0))
    if alpha_value <= 0:
        return None

    rgba = np.zeros(category_array.shape + (4,), dtype=np.uint8)
    valid_values = np.array([int(value) for value in value_to_color], dtype=np.int16)
    valid_mask = np.isin(category_array, valid_values)
    if not np.any(valid_mask):
        return None

    for value, color in value_to_color.items():
        rgba[category_array == int(value)] = ImageColor.getcolor(str(color), "RGBA")
    rgba[..., 3] = np.where(valid_mask, alpha_value, 0).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def rgba_with_alpha(color, alpha):
    red, green, blue, _ = ImageColor.getcolor(str(color), "RGBA")
    alpha_value = int(round(np.clip(float(alpha), 0.0, 1.0) * 255.0))
    return f"rgba({red}, {green}, {blue}, {alpha_value / 255.0:.6f})"


def maplibre_color_relief_expression(colorscale_name, zmin, zmax, alpha=1.0, sample_count=16):
    lower = float(zmin)
    upper = max(float(zmax), lower + 1e-6)
    stops = np.linspace(lower, upper, sample_count)
    colors = sample_colorscale(colorscale_name, np.linspace(0.0, 1.0, sample_count))
    expression = ["interpolate", ["linear"], ["elevation"]]
    for stop, color in zip(stops, colors, strict=False):
        expression.extend([float(stop), rgba_with_alpha(color, alpha)])
    return expression


def terrain_dem_token(cache_key):
    return hashlib.sha1(repr(tuple(cache_key)).encode("utf-8")).hexdigest()[:16]


def register_terrain_dem_bundle(bundle):
    cache_key = tuple(bundle["cache_key"])
    token = terrain_dem_token(cache_key)
    TERRAIN_DEM_TOKENS[token] = cache_key
    return token


def tile_lon_lat_bounds(x, y, z):
    tile_count = 2**int(z)
    west = float(x) / tile_count * 360.0 - 180.0
    east = float(x + 1) / tile_count * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - (2.0 * float(y)) / tile_count))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - (2.0 * float(y + 1)) / tile_count))))
    return west, south, east, north


def terrain_tile_intersects_bbox(tile_bbox, terrain_bbox):
    west, south, east, north = tile_bbox
    return not (
        east <= float(terrain_bbox["min_lon"])
        or west >= float(terrain_bbox["max_lon"])
        or north <= float(terrain_bbox["min_lat"])
        or south >= float(terrain_bbox["max_lat"])
    )


def encode_terrarium_rgb(elevations):
    values = np.asarray(elevations, dtype=np.float32)
    safe_values = np.where(np.isfinite(values), values, 0.0) + 32768.0
    red = np.floor(safe_values / 256.0)
    green = np.floor(safe_values - red * 256.0)
    blue = np.floor((safe_values - np.floor(safe_values)) * 256.0)
    stacked = np.stack(
        [
            np.clip(red, 0, 255).astype(np.uint8),
            np.clip(green, 0, 255).astype(np.uint8),
            np.clip(blue, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return stacked


@lru_cache(maxsize=2048)
def build_terrain_dem_tile_png(token, z, x, y):
    cache_key = TERRAIN_DEM_TOKENS.get(str(token))
    if cache_key is None:
        raise KeyError(f"Unknown terrain DEM token: {token}")

    bundle = get_map_bundle(*cache_key)
    terrain_projected = cached_bundle_data_array_values(bundle, "terrain_projected", dtype=np.float32)
    source_transform = bundle["projected_transform"]
    fill_value = float(bundle.get("terrain_fill_value", 0.0))

    tile_bbox = tile_lon_lat_bounds(int(x), int(y), int(z))
    if not terrain_tile_intersects_bbox(tile_bbox, bundle["terrain_display_bounds"]):
        rgb = encode_terrarium_rgb(np.full((TERRAIN_TILE_SIZE, TERRAIN_TILE_SIZE), fill_value, dtype=np.float32))
        image = Image.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    west, south, east, north = tile_bbox
    mercator_bounds = transform_bounds(GEOGRAPHIC_CRS, WEB_MERCATOR_CRS, west, south, east, north)
    projected_bounds = transform_bounds(WEB_MERCATOR_CRS, PROJECTED_CRS, *mercator_bounds)
    destination = np.full((TERRAIN_TILE_SIZE, TERRAIN_TILE_SIZE), np.nan, dtype=np.float32)
    source_values, source_transform = subset_raster_for_bounds(
        terrain_projected,
        source_transform,
        *projected_bounds,
        padding=1,
    )
    safe_reproject(
        source=source_values,
        destination=destination,
        src_transform=source_transform,
        src_crs=PROJECTED_CRS,
        src_nodata=np.nan,
        dst_transform=rasterio.transform.from_bounds(*mercator_bounds, TERRAIN_TILE_SIZE, TERRAIN_TILE_SIZE),
        dst_crs=WEB_MERCATOR_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    destination = np.where(np.isfinite(destination), destination, fill_value)
    rgb = encode_terrarium_rgb(destination)
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@lru_cache(maxsize=512)
def fetch_osm_tile(z, x, y):
    tile_count = 2**z
    wrapped_x = int(x) % tile_count
    clipped_y = max(0, min(int(y), tile_count - 1))
    tile_url = OSM_TILE_URL.format(z=z, x=wrapped_x, y=clipped_y)
    data = fetch_binary_with_headers(tile_url, headers=OSM_TILE_HEADERS, timeout=120)
    return Image.open(io.BytesIO(data)).convert("RGB")


@lru_cache(maxsize=64)
def get_osm_background_image(min_lon, min_lat, max_lon, max_lat):
    min_lon, min_lat, max_lon, max_lat = normalize_bbox(min_lon, min_lat, max_lon, max_lat)
    zoom = choose_osm_zoom(min_lon, min_lat, max_lon, max_lat)

    x0f = lon_to_tile_x(min_lon, zoom)
    x1f = lon_to_tile_x(max_lon, zoom)
    y0f = lat_to_tile_y(max_lat, zoom)
    y1f = lat_to_tile_y(min_lat, zoom)

    x0 = int(math.floor(x0f))
    x1 = int(math.floor(x1f))
    y0 = int(math.floor(y0f))
    y1 = int(math.floor(y1f))

    stitched = Image.new("RGB", ((x1 - x0 + 1) * OSM_TILE_SIZE, (y1 - y0 + 1) * OSM_TILE_SIZE))
    for tile_x in range(x0, x1 + 1):
        for tile_y in range(y0, y1 + 1):
            tile = fetch_osm_tile(zoom, tile_x, tile_y)
            stitched.paste(tile, ((tile_x - x0) * OSM_TILE_SIZE, (tile_y - y0) * OSM_TILE_SIZE))

    left = int(math.floor((x0f - x0) * OSM_TILE_SIZE))
    upper = int(math.floor((y0f - y0) * OSM_TILE_SIZE))
    right = int(math.ceil((x1f - x0) * OSM_TILE_SIZE))
    lower = int(math.ceil((y1f - y0) * OSM_TILE_SIZE))
    cropped = stitched.crop((left, upper, max(left + 1, right), max(upper + 1, lower)))

    src_width, src_height = cropped.size
    src_transform = rasterio.transform.from_bounds(
        *transform_bounds(GEOGRAPHIC_CRS, WEB_MERCATOR_CRS, min_lon, min_lat, max_lon, max_lat),
        src_width,
        src_height,
    )
    dst_width, dst_height = geographic_bbox_to_pixel_shape(min_lon, min_lat, max_lon, max_lat)
    dst_transform = rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, dst_width, dst_height)
    source = np.asarray(cropped, dtype=np.uint8).transpose(2, 0, 1)
    destination = np.zeros((3, dst_height, dst_width), dtype=np.uint8)

    for band_index in range(3):
        safe_reproject(
            source=source[band_index],
            destination=destination[band_index],
            src_transform=src_transform,
            src_crs=WEB_MERCATOR_CRS,
            dst_transform=dst_transform,
            dst_crs=GEOGRAPHIC_CRS,
            resampling=Resampling.bilinear,
        )

    return Image.fromarray(destination.transpose(1, 2, 0))


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
def _get_map_bundle_cached(min_lon, min_lat, max_lon, max_lat, target_resolution_m):
    target_resolution_m = normalize_bbox_resolution_m(target_resolution_m)
    min_x, min_y, max_x, max_y = transform_bounds(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    display_width, display_height = target_pixel_shape_for_extent(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        default_geographic_pixel_size(target_resolution_m),
    )
    projected_width, projected_height = target_pixel_shape_for_extent(
        min_x,
        min_y,
        max_x,
        max_y,
        default_projected_pixel_size(target_resolution_m),
    )

    terrain_display = fetch_elevation_raster_mosaic(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        GEOGRAPHIC_CRS,
        display_width,
        display_height,
    )
    terrain_display_transform = rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, display_width, display_height)
    terrain_display_lon_axis, terrain_display_lat_axis = raster_axes(terrain_display_transform, display_height, display_width)
    terrain_display_shape = (display_height, display_width)
    terrain_display_bounds = {
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }
    map_min_x, map_min_y, map_max_x, map_max_y = transform_bounds(
        GEOGRAPHIC_CRS,
        WEB_MERCATOR_CRS,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    map_display_width = max(int(math.ceil((map_max_x - map_min_x) / max(float(target_resolution_m), 1e-9))), 1)
    map_display_height = max(int(math.ceil((map_max_y - map_min_y) / max(float(target_resolution_m), 1e-9))), 1)
    map_display_transform = rasterio.transform.from_bounds(
        map_min_x,
        map_min_y,
        map_max_x,
        map_max_y,
        map_display_width,
        map_display_height,
    )
    map_display_x_axis, map_display_y_axis = raster_axes(
        map_display_transform,
        map_display_height,
        map_display_width,
    )
    map_display_lon_axis, map_display_lat_axis = mercator_axes_to_geographic(
        map_display_x_axis,
        map_display_y_axis,
    )
    map_display_shape = (map_display_height, map_display_width)

    terrain_band = fetch_elevation_raster_mosaic(
        min_x,
        min_y,
        max_x,
        max_y,
        PROJECTED_CRS,
        projected_width,
        projected_height,
    )
    projected_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, projected_width, projected_height)
    projected_shape = (projected_height, projected_width)
    terrain_x_axis, terrain_y_axis = raster_axes(projected_transform, projected_height, projected_width)
    terrain_finite = terrain_band[np.isfinite(terrain_band)]
    terrain_fill_value = float(np.nanmin(terrain_finite)) if terrain_finite.size else 0.0

    worldcover_band = np.zeros_like(terrain_band, dtype=np.uint8)
    for worldcover_url in worldcover_tile_urls(min_lon, min_lat, max_lon, max_lat):
        with open_remote_raster(worldcover_url) as src:
            source_band = src.read(1)
            safe_reproject(
                source=source_band,
                destination=worldcover_band,
                src_transform=resolved_raster_transform(src, float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=projected_transform,
                dst_crs=PROJECTED_CRS,
                dst_nodata=0,
                resampling=Resampling.nearest,
                init_dest_nodata=False,
            )
    worldcover_display = np.zeros(terrain_display_shape, dtype=np.uint8)
    safe_reproject(
        source=worldcover_band,
        destination=worldcover_display,
        src_transform=projected_transform,
        src_crs=PROJECTED_CRS,
        src_nodata=0,
        dst_transform=terrain_display_transform,
        dst_crs=GEOGRAPHIC_CRS,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=False,
    )

    terrain_projected = xr.DataArray(
        terrain_band,
        dims=("y", "x"),
        coords={"y": terrain_y_axis, "x": terrain_x_axis},
        name="elevation",
    ).chunk({'x': 1024, 'y': 1024}).sortby("y", ascending=False)

    worldcover_projected = xr.DataArray(
        worldcover_band,
        dims=("y", "x"),
        coords={"y": terrain_y_axis, "x": terrain_x_axis},
        name="worldcover",
    ).chunk({'x': 1024, 'y': 1024}).sortby("y", ascending=False)

    return {
        "cache_key": (min_lon, min_lat, max_lon, max_lat, target_resolution_m),
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
        "target_resolution_m": float(target_resolution_m),
        "terrain_display": terrain_display,
        "terrain_display_lon_axis": terrain_display_lon_axis.astype(np.float64),
        "terrain_display_lat_axis": terrain_display_lat_axis.astype(np.float64),
        "terrain_display_transform": terrain_display_transform,
        "terrain_display_shape": terrain_display_shape,
        "terrain_display_bounds": terrain_display_bounds,
        "map_display_transform": map_display_transform,
        "map_display_shape": map_display_shape,
        "map_display_lon_axis": map_display_lon_axis.astype(np.float64),
        "map_display_lat_axis": map_display_lat_axis.astype(np.float64),
        "terrain_fill_value": terrain_fill_value,
        "terrain_projected": terrain_projected,
        "worldcover_display": worldcover_display,
        "worldcover_projected": worldcover_projected,
        "projected_transform": projected_transform,
        "projected_shape": projected_shape,
    }


def get_map_bundle(min_lon, min_lat, max_lon, max_lat, target_resolution_m=DEFAULT_BBOX_RESOLUTION_M):
    return _get_map_bundle_cached(*bundle_cache_key(min_lon, min_lat, max_lon, max_lat, target_resolution_m))


get_map_bundle.cache_clear = _get_map_bundle_cached.cache_clear


@lru_cache(maxsize=128)
def get_viewshed_assessment_terrain(longitude, latitude, radius_m):
    longitude = float(longitude)
    latitude = float(latitude)
    radius_m = max(float(radius_m), 1.0)

    projected_x, projected_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        [longitude],
        [latitude],
    )
    center_x = float(projected_x[0])
    center_y = float(projected_y[0])
    min_x = center_x - radius_m
    min_y = center_y - radius_m
    max_x = center_x + radius_m
    max_y = center_y + radius_m
    span_m = max(max_x - min_x, max_y - min_y)
    projected_dim = span_to_sample_dim(
        span_m,
        VIEWSHED_ASSESSMENT_TARGET_RESOLUTION_M,
        VIEWSHED_ASSESSMENT_MIN_DIM,
        VIEWSHED_ASSESSMENT_MAX_DIM,
    )

    terrain_href = elevation_export_image_href(
        min_x,
        min_y,
        max_x,
        max_y,
        PROJECTED_CRS,
        projected_dim,
        projected_dim,
    )
    with open_remote_raster(terrain_href) as src:
        terrain_values = src.read(1).astype(np.float32)
        projected_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, src.width, src.height)
        terrain_x_axis, terrain_y_axis = raster_axes(projected_transform, src.height, src.width)
        projected_shape = (src.height, src.width)

    display_min_lon, display_min_lat, display_max_lon, display_max_lat = transform_bounds(
        PROJECTED_CRS,
        GEOGRAPHIC_CRS,
        min_x,
        min_y,
        max_x,
        max_y,
    )
    display_width, display_height = geographic_bbox_to_pixel_shape(
        display_min_lon,
        display_min_lat,
        display_max_lon,
        display_max_lat,
        max_dim=max(projected_shape),
        min_dim=min(projected_shape),
    )
    display_transform = rasterio.transform.from_bounds(
        display_min_lon,
        display_min_lat,
        display_max_lon,
        display_max_lat,
        display_width,
        display_height,
    )
    display_lon_axis, display_lat_axis = raster_axes(display_transform, display_height, display_width)

    return {
        "terrain_da": xr.DataArray(
            terrain_values,
            dims=("y", "x"),
            coords={"y": terrain_y_axis, "x": terrain_x_axis},
            name="elevation",
        ).sortby("y", ascending=False),
        "projected_transform": projected_transform,
        "projected_shape": projected_shape,
        "display_transform": display_transform,
        "display_shape": (display_height, display_width),
        "display_bounds": {
            "min_lon": float(display_min_lon),
            "min_lat": float(display_min_lat),
            "max_lon": float(display_max_lon),
            "max_lat": float(display_max_lat),
        },
        "display_lon_axis": display_lon_axis.astype(np.float64),
        "display_lat_axis": display_lat_axis.astype(np.float64),
        "center_x": center_x,
        "center_y": center_y,
    }


def ensure_analysis_context(bundle):
    global ANALYSIS_KEY

    bundle_key = bundle["cache_key"]
    if ANALYSIS_KEY == bundle_key:
        return

    if ANALYSIS_CONTEXT:
        clear_analysis_context(run_gc=True)

    terrain_projected = cached_bundle_eager_data_array(bundle, "terrain_projected", dtype=np.float32)
    worldcover_projected = cached_bundle_eager_data_array(bundle, "worldcover_projected", dtype=np.int16)
    terrain_values = cached_bundle_data_array_values(bundle, "terrain_projected", dtype=np.float32)
    worldcover_values = cached_bundle_data_array_values(bundle, "worldcover_projected", dtype=np.int16)

    ANALYSIS_CONTEXT.update(
        {
            "terrain_da": terrain_projected,
            "worldcover_da": worldcover_projected,
            "terrain_values": terrain_values,
            "worldcover_values": worldcover_values,
            "terrain_x": terrain_projected.x.values.astype(np.float64),
            "terrain_y": terrain_projected.y.values.astype(np.float64),
            "terrain_valid_mask": np.isfinite(terrain_values),
        }
    )
    ANALYSIS_KEY = bundle_key


def clear_analysis_context(run_gc=False):
    global ANALYSIS_KEY

    for key in tuple(ANALYSIS_CONTEXT):
        ANALYSIS_CONTEXT.pop(key, None)
    _get_path_profile_by_index_cached.cache_clear()
    ANALYSIS_KEY = None
    if run_gc:
        gc.collect()


def dist3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def dist2d(p1, p2):
    x1, y1, _z1 = p1
    x2, y2, _z2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def earth_curvature(distance_along_m, total_distance_m, effective_radius_m=EFFECTIVE_EARTH_RADIUS_M):
    total_distance_m = max(float(total_distance_m), 1e-6)
    distance_along_m = np.asarray(distance_along_m, dtype=np.float64)
    remaining_m = np.maximum(total_distance_m - distance_along_m, 0.0)
    return (distance_along_m * remaining_m) / (2.0 * float(effective_radius_m))


def planar_distance_grid_km(x_axis, y_axis, origin_x, origin_y):
    x_values = np.asarray(x_axis, dtype=np.float64)
    y_values = np.asarray(y_axis, dtype=np.float64)
    dx = x_values[np.newaxis, :].astype(np.float64, copy=True)
    dx -= float(origin_x)
    dy = y_values[:, np.newaxis].astype(np.float64, copy=True)
    dy -= float(origin_y)

    distance_km = np.hypot(dx, dy)
    distance_km /= 1000.0
    np.maximum(distance_km, 1e-6, out=distance_km)
    return distance_km


@maybe_njit(cache=False)
def _earth_curvature_numba(distance_along_m, total_distance_m, effective_radius_m):
    if total_distance_m <= 0.0:
        return 0.0
    remaining_m = total_distance_m - distance_along_m
    if remaining_m < 0.0:
        remaining_m = 0.0
    return (distance_along_m * remaining_m) / (2.0 * effective_radius_m)



def snap_point(x_coord, y_coord):
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    terrain_values = ANALYSIS_CONTEXT["terrain_values"]
    col = nearest_axis_index(terrain_x, x_coord)
    row = nearest_axis_index(terrain_y, y_coord)
    return row, col, float(terrain_x[col]), float(terrain_y[row]), float(terrain_values[row, col])


@maybe_njit(cache=False)
def _rounded_path_index(start_index, end_index, sample_index, sample_count):
    if sample_count <= 1:
        return int(start_index)
    if sample_index <= 0:
        return int(start_index)
    if sample_index >= sample_count - 1:
        return int(end_index)
    step = (end_index - start_index) / (sample_count - 1.0)
    return int(np.rint(start_index + step * sample_index))


@maybe_njit(cache=False)
def _lookup_value(lookup, raw_value):
    index = int(raw_value)
    if index < 0:
        index = 0
    elif index >= lookup.shape[0]:
        index = lookup.shape[0] - 1
    return lookup[index]


@maybe_njit(cache=False)
def _path_profile_metrics_by_index(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    row1,
    col1,
    row2,
    col2,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    reverse_requested = (row2 < row1) or (row2 == row1 and col2 < col1)
    if reverse_requested:
        original_row1 = row1
        original_col1 = col1
        original_observer_height = observer_height
        row1 = row2
        col1 = col2
        row2 = original_row1
        col2 = original_col1
        observer_height = target_height
        target_height = original_observer_height

    x1 = float(terrain_x[col1])
    y1 = float(terrain_y[row1])
    z1 = float(terrain_values[row1, col1] + observer_height)
    x2 = float(terrain_x[col2])
    y2 = float(terrain_y[row2])
    z2 = float(terrain_values[row2, col2] + target_height)

    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    spacing_m = max(float(sample_spacing_m), 1e-6)
    sample_count = max(int(min_samples), int(math.ceil(dist / spacing_m)))
    if sample_count < 1:
        sample_count = 1
    legacy_spacing_m = max(float(legacy_sample_spacing_m), 1e-6)
    legacy_sample_count = max(int(min_samples), int(math.ceil(dist / legacy_spacing_m)))
    current_active_samples = max(sample_count - 2, 1)
    legacy_active_samples = max(legacy_sample_count - 2, 1)
    sample_weight = float(legacy_active_samples) / float(current_active_samples)

    total_distance_m = 0.0
    prev_x = x1
    prev_y = y1
    for sample_index in range(1, sample_count):
        row = _rounded_path_index(row1, row2, sample_index, sample_count)
        col = _rounded_path_index(col1, col2, sample_index, sample_count)
        x = float(terrain_x[col])
        y = float(terrain_y[row])
        step_dx = x - prev_x
        step_dy = y - prev_y
        total_distance_m += math.sqrt(step_dx * step_dx + step_dy * step_dy)
        prev_x = x
        prev_y = y

    total_distance_m = max(total_distance_m, 1e-6)
    distance_along_m = 0.0
    prev_x = x1
    prev_y = y1
    path_loss_db = 0.0
    direct_los_blocked = False

    for sample_index in range(sample_count):
        row = _rounded_path_index(row1, row2, sample_index, sample_count)
        col = _rounded_path_index(col1, col2, sample_index, sample_count)
        x = float(terrain_x[col])
        y = float(terrain_y[row])
        if sample_index > 0:
            step_dx = x - prev_x
            step_dy = y - prev_y
            distance_along_m += math.sqrt(step_dx * step_dx + step_dy * step_dy)
        prev_x = x
        prev_y = y

        path_fraction = distance_along_m / total_distance_m
        curvature_bulge = _earth_curvature_numba(
            distance_along_m,
            total_distance_m,
            EFFECTIVE_EARTH_RADIUS_M,
        )
        line_z = z1 + (z2 - z1) * path_fraction - curvature_bulge

        dx1 = x - x1
        dy1 = y - y1
        dz1 = line_z - z1
        dx2 = x - x2
        dy2 = y - y2
        dz2 = line_z - z2
        d1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
        d2 = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)
        fresnel = math.sqrt((0.3304 * d1 * d2) / max(d1 + d2, 1e-6))

        terrain_height = float(terrain_values[row, col])
        worldcover_class = worldcover_values[row, col]
        obstruction_top = terrain_height + _lookup_value(offset_lookup, worldcover_class)
        clearance_height = line_z - fresnel * 0.6
        fresnel_zone_height = max(line_z - clearance_height, 1e-6)
        blockage_fraction = (obstruction_top - clearance_height) / fresnel_zone_height
        if blockage_fraction < 0.0:
            blockage_fraction = 0.0
        elif blockage_fraction > 1.0:
            blockage_fraction = 1.0
        terrain_only_blocked = terrain_height >= line_z

        if sample_index == 0 or sample_index == sample_count - 1:
            blockage_fraction = 0.0
            terrain_only_blocked = False
        elif terrain_only_blocked:
            direct_los_blocked = True

        path_loss_db += _lookup_value(attenuation_lookup, worldcover_class) * blockage_fraction * sample_weight
        if terrain_only_blocked:
            path_loss_db += 20.0 * sample_weight

    return path_loss_db, direct_los_blocked


@maybe_njit(cache=False)
def _path_loss_only_by_index(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    row1,
    col1,
    row2,
    col2,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    path_loss_db, _direct_los_blocked = _path_profile_metrics_by_index(
        terrain_values,
        worldcover_values,
        terrain_x,
        terrain_y,
        row1,
        col1,
        row2,
        col2,
        observer_height,
        target_height,
        min_samples,
        sample_spacing_m,
        legacy_sample_spacing_m,
        offset_lookup,
        attenuation_lookup,
    )
    return path_loss_db


@maybe_njit(parallel=True, cache=False)
def _compute_ground_loss_batch_numba(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    observer_row,
    observer_col,
    target_rows,
    target_cols,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    losses = np.empty(target_rows.shape[0], dtype=np.float32)
    for index in prange(target_rows.shape[0]):
        losses[index] = _path_loss_only_by_index(
            terrain_values,
            worldcover_values,
            terrain_x,
            terrain_y,
            int(observer_row),
            int(observer_col),
            int(target_rows[index]),
            int(target_cols[index]),
            observer_height,
            target_height,
            min_samples,
            sample_spacing_m,
            legacy_sample_spacing_m,
            offset_lookup,
            attenuation_lookup,
    )
    return losses


@maybe_njit(parallel=True, cache=False)
def _compute_ground_loss_visibility_batch_numba(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    observer_row,
    observer_col,
    target_rows,
    target_cols,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    losses = np.empty(target_rows.shape[0], dtype=np.float32)
    visible = np.empty(target_rows.shape[0], dtype=np.bool_)
    for index in prange(target_rows.shape[0]):
        loss_db, direct_los_blocked = _path_profile_metrics_by_index(
            terrain_values,
            worldcover_values,
            terrain_x,
            terrain_y,
            int(observer_row),
            int(observer_col),
            int(target_rows[index]),
            int(target_cols[index]),
            observer_height,
            target_height,
            min_samples,
            sample_spacing_m,
            legacy_sample_spacing_m,
            offset_lookup,
            attenuation_lookup,
        )
        losses[index] = loss_db
        visible[index] = not direct_los_blocked
    return losses, visible


@maybe_njit(cache=False)
def _compute_ground_loss_batch_numba_serial(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    observer_row,
    observer_col,
    target_rows,
    target_cols,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    losses = np.empty(target_rows.shape[0], dtype=np.float32)
    for index in range(target_rows.shape[0]):
        losses[index] = _path_loss_only_by_index(
            terrain_values,
            worldcover_values,
            terrain_x,
            terrain_y,
            int(observer_row),
            int(observer_col),
            int(target_rows[index]),
            int(target_cols[index]),
            observer_height,
            target_height,
            min_samples,
            sample_spacing_m,
            legacy_sample_spacing_m,
            offset_lookup,
            attenuation_lookup,
        )
    return losses


@maybe_njit(cache=False)
def _compute_ground_loss_visibility_batch_numba_serial(
    terrain_values,
    worldcover_values,
    terrain_x,
    terrain_y,
    observer_row,
    observer_col,
    target_rows,
    target_cols,
    observer_height,
    target_height,
    min_samples,
    sample_spacing_m,
    legacy_sample_spacing_m,
    offset_lookup,
    attenuation_lookup,
):
    losses = np.empty(target_rows.shape[0], dtype=np.float32)
    visible = np.empty(target_rows.shape[0], dtype=np.bool_)
    for index in range(target_rows.shape[0]):
        loss_db, direct_los_blocked = _path_profile_metrics_by_index(
            terrain_values,
            worldcover_values,
            terrain_x,
            terrain_y,
            int(observer_row),
            int(observer_col),
            int(target_rows[index]),
            int(target_cols[index]),
            observer_height,
            target_height,
            min_samples,
            sample_spacing_m,
            legacy_sample_spacing_m,
            offset_lookup,
            attenuation_lookup,
        )
        losses[index] = loss_db
        visible[index] = not direct_los_blocked
    return losses, visible


def normalize_target_indices(target_rows_or_cells, target_cols=None):
    if target_cols is not None:
        rows = np.asarray(target_rows_or_cells, dtype=np.int32).reshape(-1)
        cols = np.asarray(target_cols, dtype=np.int32).reshape(-1)
        return rows, cols

    if target_rows_or_cells is None:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    cells = np.asarray(target_rows_or_cells)
    if cells.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("Target cells must be an array-like collection of (row, col) pairs.")
    return cells[:, 0].astype(np.int32, copy=False), cells[:, 1].astype(np.int32, copy=False)


@lru_cache(maxsize=PATH_PROFILE_CACHE_MAXSIZE)
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
        total_distance_m = 1e-6
        path_fraction = np.array([0.0], dtype=np.float64)

    line_z = pos1[2] + (pos2[2] - pos1[2]) * path_fraction - earth_curvature(distance_along_m, total_distance_m)
    d1 = np.sqrt((x_path - pos1[0]) ** 2 + (y_path - pos1[1]) ** 2 + (line_z - pos1[2]) ** 2)
    d2 = np.sqrt((x_path - pos2[0]) ** 2 + (y_path - pos2[1]) ** 2 + (line_z - pos2[2]) ** 2)
    fresnel = np.sqrt((0.3304 * d1 * d2) / np.maximum(d1 + d2, 1e-6))

    ground_surface = worldcover_values[row_path, col_path]
    ground_surface_offset = V_OFFSET(ground_surface)
    terrain_profile = terrain_values[row_path, col_path]
    terrain_only_blocked = terrain_profile >= line_z
    terrain_only_blocked[0] = False
    terrain_only_blocked[-1] = False

    obstruction_top = terrain_profile + ground_surface_offset
    clearance_height = line_z - fresnel * 0.6
    fresnel_zone_height = np.maximum(line_z - clearance_height, 1e-6)
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


def compute_ground_loss_for_chunk(args, use_parallel_numba=True):
    if len(args) == 5:
        observer_row, observer_col, cells, observer_height, target_height = args
        rows, cols = normalize_target_indices(cells)
        sample_spacing_m = DEFAULT_RSSI_PATH_SAMPLE_SPACING_M
    elif len(args) == 6:
        observer_row, observer_col, rows, cols, observer_height, target_height = args
        sample_spacing_m = DEFAULT_RSSI_PATH_SAMPLE_SPACING_M
        rows, cols = normalize_target_indices(rows, cols)
    elif len(args) == 7:
        observer_row, observer_col, rows, cols, observer_height, target_height, sample_spacing_m = args
        rows, cols = normalize_target_indices(rows, cols)
    else:
        raise ValueError("Expected 5, 6, or 7 arguments for compute_ground_loss_for_chunk.")

    if rows.size == 0:
        return rows, cols, np.empty(0, dtype=np.float32)
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)

    compute_fn = _compute_ground_loss_batch_numba if use_parallel_numba else _compute_ground_loss_batch_numba_serial
    losses = compute_fn(
        ANALYSIS_CONTEXT["terrain_values"],
        ANALYSIS_CONTEXT["worldcover_values"],
        ANALYSIS_CONTEXT["terrain_x"],
        ANALYSIS_CONTEXT["terrain_y"],
        int(observer_row),
        int(observer_col),
        rows,
        cols,
        float(observer_height),
        float(target_height),
        2,
        float(sample_spacing_m),
        float(DEFAULT_RSSI_PATH_SAMPLE_SPACING_M),
        OFFSET_LOOKUP,
        ATTENUATION_LOOKUP,
    )
    return rows, cols, np.asarray(losses, dtype=np.float32)


def compute_ground_loss_visibility_for_chunk(args, use_parallel_numba=True):
    if len(args) == 5:
        observer_row, observer_col, cells, observer_height, target_height = args
        rows, cols = normalize_target_indices(cells)
        sample_spacing_m = DEFAULT_RSSI_PATH_SAMPLE_SPACING_M
    elif len(args) == 6:
        observer_row, observer_col, rows, cols, observer_height, target_height = args
        sample_spacing_m = DEFAULT_RSSI_PATH_SAMPLE_SPACING_M
        rows, cols = normalize_target_indices(rows, cols)
    elif len(args) == 7:
        observer_row, observer_col, rows, cols, observer_height, target_height, sample_spacing_m = args
        rows, cols = normalize_target_indices(rows, cols)
    else:
        raise ValueError("Expected 5, 6, or 7 arguments for compute_ground_loss_visibility_for_chunk.")

    if rows.size == 0:
        return rows, cols, np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)

    compute_fn = (
        _compute_ground_loss_visibility_batch_numba
        if use_parallel_numba
        else _compute_ground_loss_visibility_batch_numba_serial
    )
    losses, visible = compute_fn(
        ANALYSIS_CONTEXT["terrain_values"],
        ANALYSIS_CONTEXT["worldcover_values"],
        ANALYSIS_CONTEXT["terrain_x"],
        ANALYSIS_CONTEXT["terrain_y"],
        int(observer_row),
        int(observer_col),
        rows,
        cols,
        float(observer_height),
        float(target_height),
        2,
        float(sample_spacing_m),
        float(DEFAULT_RSSI_PATH_SAMPLE_SPACING_M),
        OFFSET_LOOKUP,
        ATTENUATION_LOOKUP,
    )
    return rows, cols, np.asarray(losses, dtype=np.float32), np.asarray(visible, dtype=bool)


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
        total_distance_m = 1e-6
        path_fraction = np.array([0.0], dtype=np.float64)

    curvature_bulge = earth_curvature(distance_along_m, total_distance_m)
    los_line = tx_z + (rx_z - tx_z) * path_fraction
    curved_terrain_profile = terrain_profile + curvature_bulge
    curved_obstruction_top = obstruction_top + curvature_bulge
    p1 = (x1, y1, tx_z)
    p2 = (x2, y2, rx_z)

    d1 = np.sqrt((x_path - x1) ** 2 + (y_path - y1) ** 2 + (los_line - tx_z) ** 2)
    d2 = np.sqrt((x_path - x2) ** 2 + (y_path - y2) ** 2 + (los_line - rx_z) ** 2)
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
        "direct_los_collision_count": int(np.count_nonzero(curved_obstruction_top >= los_line)),
        "path_loss_db": float(path_loss_db),
        "link_distance_km": float(link_distance_km),
        "distance_along_km": distance_along_m / 1000.0,
        "terrain_profile_m": curved_terrain_profile,
        "obstruction_top_m": curved_obstruction_top,
        "los_line_m": los_line,
        "fresnel_upper_m": fresnel_upper,
        "fresnel_lower_m": fresnel_lower,
        "blockage_fraction": blockage_fraction,
        "attenuation_per_sample_db": attenuation_per_sample,
        "terrain_only_blocked": terrain_only_blocked,
        "worldcover_classes": worldcover_profile,
        "direct_los_hits": curved_obstruction_top >= los_line,
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


def bounded_cache_put(cache, key, value, max_size=8):
    cache[key] = value
    while len(cache) > int(max_size):
        cache.pop(next(iter(cache)))


def bounded_dask_worker_count(task_count, max_workers):
    if int(task_count) <= 1:
        return 1
    cpu_total = max(int(os.cpu_count() or 1), 1)
    return max(1, min(int(task_count), cpu_total, int(max_workers)))


def rssi_dask_worker_count(task_count):
    return bounded_dask_worker_count(task_count, RSSI_DASK_MAX_WORKERS)


def make_dask_progress_callback(delayed_tasks, progress_callback):
    if progress_callback is None:
        return nullcontext()

    pending_keys = {getattr(task, "key", None) for task in delayed_tasks}
    pending_keys.discard(None)
    if not pending_keys:
        return nullcontext()

    callback_lock = threading.Lock()

    def posttask(key, _result, _dsk, _state, _worker_id):
        with callback_lock:
            if key not in pending_keys:
                return
            pending_keys.remove(key)
        progress_callback(1, key)

    return Callback(posttask=posttask)


def compute_dask_tasks(tasks, use_threads=True, progress_callback=None):
    delayed_tasks = list(tasks or [])
    if not delayed_tasks:
        return tuple()
    with make_dask_progress_callback(delayed_tasks, progress_callback):
        if not use_threads:
            return dask.compute(
                *delayed_tasks,
                scheduler="single-threaded",
            )
        if len(delayed_tasks) == 1:
            return (
                delayed_tasks[0].compute(
                    scheduler="threads",
                    num_workers=1,
                ),
            )
        return dask.compute(
            *delayed_tasks,
            scheduler="threads",
            num_workers=rssi_dask_worker_count(len(delayed_tasks)),
        )


def set_rssi_progress_state(request_id, total_nodes, completed_nodes=0, failed=False):
    with RSSI_PROGRESS_LOCK:
        RSSI_PROGRESS_STATE[int(request_id)] = {
            "total_nodes": max(int(total_nodes), 0),
            "completed_nodes": max(int(completed_nodes), 0),
            "failed": bool(failed),
        }


def increment_rssi_progress_state(request_id, delta=1):
    with RSSI_PROGRESS_LOCK:
        state = RSSI_PROGRESS_STATE.setdefault(
            int(request_id),
            {"total_nodes": 0, "completed_nodes": 0, "failed": False},
        )
        state["completed_nodes"] = max(int(state.get("completed_nodes", 0)) + int(delta), 0)


def get_rssi_progress_state(request_id):
    with RSSI_PROGRESS_LOCK:
        state = RSSI_PROGRESS_STATE.get(int(request_id))
        return dict(state) if state is not None else None


def numba_parallel_threads_safe():
    if njit is None:
        return True
    try:
        from numba import threading_layer

        return str(threading_layer()).lower() in {"tbb", "omp"}
    except Exception:
        return False


def format_elapsed_time(seconds):
    elapsed = max(float(seconds), 0.0)
    if elapsed < 1.0:
        return f"{elapsed * 1000.0:.0f} ms"
    if elapsed < 60.0:
        return f"{elapsed:.2f} s"
    minutes = int(elapsed // 60.0)
    remaining_seconds = elapsed - (minutes * 60.0)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def send_desktop_notification(title, message):
    if DESKTOP_NOTIFIER is None:
        return False

    async def _send():
        await DESKTOP_NOTIFIER.send(
            title=str(title),
            message=str(message),
        )

    def _worker():
        try:
            asyncio.run(_send())
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()
    return True


def effective_viewshed_sample_count(sample_count=None, max_samples=VIEWSHED_ASSESSMENT_MAX_SAMPLES):
    resolved_sample_count = int(sample_count or DEFAULT_VIEWSHED_SAMPLE_COUNT)
    if resolved_sample_count in VIEWSHED_SAMPLE_COUNT_OPTIONS:
        return int(min(int(max_samples), resolved_sample_count))
    return int(min(int(max_samples), DEFAULT_VIEWSHED_SAMPLE_COUNT))


def centered_circular_ring_count(target_sample_count):
    target_sample_count = max(1, int(target_sample_count))
    if target_sample_count <= 1:
        return 0

    ring_count = int(round((math.sqrt((4.0 * float(target_sample_count)) - 1.0) - 1.0) / 2.0))
    if 1 + 3 * ring_count * (ring_count + 1) == target_sample_count:
        return ring_count

    closest_ring_count = 1
    closest_error = None
    for candidate_ring_count in range(1, 16):
        candidate_count = 1 + 3 * candidate_ring_count * (candidate_ring_count + 1)
        candidate_error = abs(candidate_count - target_sample_count)
        if closest_error is None or candidate_error < closest_error:
            closest_ring_count = candidate_ring_count
            closest_error = candidate_error
    return closest_ring_count


def generate_circular_samples(center_x, center_y, radius_m, target_sample_count):
    center_x = float(center_x)
    center_y = float(center_y)
    radius_m = float(radius_m)
    target_sample_count = max(1, int(target_sample_count))
    if target_sample_count <= 1 or radius_m <= 0:
        return np.asarray([[center_x, center_y]], dtype=np.float64)

    ring_count = centered_circular_ring_count(target_sample_count)

    samples = [(center_x, center_y)]
    for ring_index in range(1, ring_count + 1):
        ring_sample_count = 6 * ring_index
        ring_radius = radius_m * (float(ring_index) / float(ring_count))
        angular_offset = math.pi / float(ring_sample_count)
        for sample_index in range(ring_sample_count):
            angle = angular_offset + (2.0 * math.pi * float(sample_index) / float(ring_sample_count))
            sample_x = center_x + ring_radius * math.cos(angle)
            sample_y = center_y + ring_radius * math.sin(angle)
            samples.append((sample_x, sample_y))
    return np.asarray(samples, dtype=np.float64)


def interpolate_viewshed_rbf(sample_points, sample_scores, target_points):
    sample_points = np.asarray(sample_points, dtype=np.float64)
    sample_scores = np.asarray(sample_scores, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if sample_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    if sample_points.shape[0] == 1:
        return np.full(target_points.shape[0], float(sample_scores[0]), dtype=np.float32)

    interpolator = RBFInterpolator(
        sample_points,
        sample_scores,
        kernel="thin_plate_spline",
        smoothing=0.0,
    )
    return np.asarray(interpolator(target_points), dtype=np.float32)


def viewshed_assessment_cache_key(bundle, point_data, radius_m, observer_height_agl, sample_count):
    payload = "|".join(
        [
            VIEWSHED_ASSESSMENT_METRIC_VERSION,
            str(bundle["cache_key"]),
            viewshed_point_signature(point_data),
            f"{float(radius_m):.3f}",
            f"{float(observer_height_agl):.3f}",
            str(int(sample_count)),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def resolve_viewshed_assessment(bundle, assessment_store, radius_m=None, observer_height_agl=None, sample_count=None):
    if not bundle or not assessment_store:
        return None
    if tuple(assessment_store.get("bundle_key", ())) != bundle["cache_key"]:
        return None
    if radius_m is not None:
        stored_radius = assessment_store.get("radius_m")
        if stored_radius is None or not math.isclose(float(stored_radius), float(radius_m), rel_tol=0.0, abs_tol=1e-6):
            return None
    if observer_height_agl is not None:
        stored_height = assessment_store.get("observer_height_agl")
        if stored_height is None or not math.isclose(
            float(stored_height),
            float(observer_height_agl),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return None
    if sample_count is not None:
        stored_sample_count = assessment_store.get("sample_count")
        if stored_sample_count is None or int(stored_sample_count) != int(sample_count):
            return None
    cache_key = str(assessment_store.get("cache_key") or "")
    if not cache_key:
        return None
    return VIEWSHED_ASSESSMENT_CACHE.get(cache_key)


def compute_viewshed_assessment(
    point_data,
    bundle,
    radius_m=VIEWSHED_ASSESSMENT_RADIUS_M,
    observer_height_agl=VIEWSHED_ASSESSMENT_OBSERVER_HEIGHT_AGL_M,
    max_samples=VIEWSHED_ASSESSMENT_MAX_SAMPLES,
    sample_count=DEFAULT_VIEWSHED_SAMPLE_COUNT,
):
    if not point_data or not bundle:
        return None, None

    longitude = point_data.get("longitude")
    latitude = point_data.get("latitude")
    if longitude is None or latitude is None:
        return None, None

    effective_sample_count = effective_viewshed_sample_count(sample_count, max_samples=max_samples)
    cache_key = viewshed_assessment_cache_key(bundle, point_data, radius_m, observer_height_agl, effective_sample_count)
    if cache_key in VIEWSHED_ASSESSMENT_CACHE:
        return cache_key, VIEWSHED_ASSESSMENT_CACHE[cache_key]

    ensure_analysis_context(bundle)
    full_terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    full_valid_mask = ANALYSIS_CONTEXT["terrain_valid_mask"]

    terrain_window = get_viewshed_assessment_terrain(longitude, latitude, radius_m)
    terrain_da = terrain_window["terrain_da"]
    terrain_values = np.asarray(terrain_da.values, dtype=np.float32)
    terrain_x = terrain_da.x.values.astype(np.float64)
    terrain_y = terrain_da.y.values.astype(np.float64)
    if terrain_values.ndim != 2 or terrain_x.size < 2 or terrain_y.size < 2:
        return None, None

    projected_x, projected_y = transform(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        [float(longitude)],
        [float(latitude)],
    )
    center_row = nearest_axis_index(terrain_y, float(projected_y[0]))
    center_col = nearest_axis_index(terrain_x, float(projected_x[0]))
    x_step = float(abs(terrain_x[1] - terrain_x[0]))
    y_step = float(abs(terrain_y[1] - terrain_y[0]))
    radius_m = max(float(radius_m), max(x_step, y_step))

    local_terrain = terrain_da
    local_values = terrain_values
    local_x = terrain_x
    local_y = terrain_y

    dx = local_x[None, :] - float(projected_x[0])
    dy = local_y[:, None] - float(projected_y[0])
    candidate_domain_mask = (dx * dx + dy * dy) <= float(radius_m) ** 2
    if not np.any(candidate_domain_mask):
        candidate_domain_mask[max(0, center_row), max(0, center_col)] = True

    valid_local_mask = np.isfinite(local_values)
    finite_local = local_values[valid_local_mask & candidate_domain_mask]
    if finite_local.size == 0:
        return None, None

    local_min_x = float(np.min(local_x) - (x_step / 2.0))
    local_max_x = float(np.max(local_x) + (x_step / 2.0))
    local_min_y = float(np.min(local_y) - (y_step / 2.0))
    local_max_y = float(np.max(local_y) + (y_step / 2.0))
    sample_points = generate_circular_samples(
        float(projected_x[0]),
        float(projected_y[0]),
        radius_m,
        effective_sample_count,
    )
    valid_sample_points = []
    for sample_x, sample_y in sample_points:
        if sample_x < local_min_x or sample_x > local_max_x or sample_y < local_min_y or sample_y > local_max_y:
            continue
        valid_sample_points.append((float(sample_x), float(sample_y)))

    sampled_points = valid_sample_points
    sampled_scores = []
    if sampled_points:
        sample_tasks = [
            dask.delayed(compute_viewshed_sample_score)(
                sample_x,
                sample_y,
                full_terrain_da,
                observer_height_agl,
                full_valid_mask,
            )
            for sample_x, sample_y in sampled_points
        ]
        sampled_scores = list(
            dask.compute(
                *sample_tasks,
                scheduler="threads",
                num_workers=bounded_dask_worker_count(len(sample_tasks), VIEWSHED_DASK_MAX_WORKERS),
            )
        )

    if not sampled_scores:
        return None, None

    local_score = np.full(local_values.shape, np.nan, dtype=np.float32)
    target_rows, target_cols = np.where(candidate_domain_mask & valid_local_mask)
    target_points = np.column_stack((local_x[target_cols], local_y[target_rows]))
    interpolated_scores = interpolate_viewshed_rbf(
        sampled_points,
        sampled_scores,
        target_points,
    )
    local_score[target_rows, target_cols] = interpolated_scores
    projected_score = np.where(candidate_domain_mask & valid_local_mask, local_score, np.nan)

    display_score = np.full(terrain_window["display_shape"], np.nan, dtype=np.float32)
    safe_reproject(
        source=projected_score,
        destination=display_score,
        src_transform=terrain_window["projected_transform"],
        src_crs=PROJECTED_CRS,
        src_nodata=np.nan,
        dst_transform=terrain_window["display_transform"],
        dst_crs=GEOGRAPHIC_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    projected_mask = candidate_domain_mask.astype(np.uint8)
    display_mask = np.zeros(terrain_window["display_shape"], dtype=np.uint8)
    safe_reproject(
        source=projected_mask,
        destination=display_mask,
        src_transform=terrain_window["projected_transform"],
        src_crs=PROJECTED_CRS,
        src_nodata=0,
        dst_transform=terrain_window["display_transform"],
        dst_crs=GEOGRAPHIC_CRS,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=False,
    )
    display_score = np.where(display_mask > 0, display_score, np.nan)

    finite_display = display_score[np.isfinite(display_score)]
    if finite_display.size == 0:
        return None, None

    result = {
        "visible_cell_count": display_score,
        "lon_axis": terrain_window["display_lon_axis"],
        "lat_axis": terrain_window["display_lat_axis"],
        "display_bounds": terrain_window["display_bounds"],
        "min": float(np.nanmin(finite_display)),
        "max": float(np.nanmax(finite_display)),
        "radius_m": float(radius_m),
        "observer_height_agl": float(observer_height_agl),
        "sample_count": int(effective_sample_count),
        "point": {
            "longitude": float(longitude),
            "latitude": float(latitude),
        },
    }
    bounded_cache_put(VIEWSHED_ASSESSMENT_CACHE, cache_key, result)
    return cache_key, result


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


def compute_viewshed_sample_score(sample_x, sample_y, terrain_da, observer_height_agl, full_valid_mask):
    visible = viewshed(
        terrain_da,
        x=float(sample_x),
        y=float(sample_y),
        observer_elev=float(observer_height_agl),
        target_elev=0.0,
    )
    visible_mask = np.asarray(visible > 0)
    return float(np.count_nonzero(visible_mask & full_valid_mask))


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


def compute_observer_projected_rssi(
    observer,
    terrain_da,
    terrain_x,
    terrain_y,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    include_ground_loss=False,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
    minimum_rssi_dbm=None,
    min_distance_km=None,
    use_parallel_ground_loss_numba=True,
):
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)
    vs = viewshed(
        terrain_da,
        x=observer.x,
        y=observer.y,
        observer_elev=observer.observer_elev,
        target_elev=float(global_rx_height_agl_m),
    )
    visible_mask = np.asarray(vs > 0)
    distance_km = planar_distance_grid_km(terrain_x, terrain_y, observer.x, observer.y)
    ground_loss_db = np.zeros(terrain_da.shape, dtype=np.float32)

    if include_ground_loss:
        observer_row, observer_col, _obs_x, _obs_y, _obs_z = snap_point(observer.x, observer.y)
        visible_rows, visible_cols = np.nonzero(visible_mask)
        if visible_rows.size:
            chunk_rows, chunk_cols, chunk_losses, chunk_visible = compute_ground_loss_visibility_for_chunk(
                (
                    observer_row,
                    observer_col,
                    visible_rows,
                    visible_cols,
                    float(observer.observer_elev),
                    float(global_rx_height_agl_m),
                    float(sample_spacing_m),
                ),
                use_parallel_numba=use_parallel_ground_loss_numba,
            )
            ground_loss_db[chunk_rows, chunk_cols] = chunk_losses
            visible_mask[chunk_rows, chunk_cols] = chunk_visible

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
    if min_distance_km is not None:
        observer_rssi = np.where(distance_km > float(min_distance_km), observer_rssi, np.nan)
    if minimum_rssi_dbm is not None:
        observer_rssi = np.where(observer_rssi >= float(minimum_rssi_dbm), observer_rssi, np.nan)
    return np.asarray(observer_rssi, dtype=np.float32)


def reproject_projected_rssi_to_display(observer_rssi, bundle):
    display_shape = bundle.get("map_display_shape", bundle["terrain_display_shape"])
    display_transform = bundle.get("map_display_transform", bundle["terrain_display_transform"])
    display_lon_axis = bundle.get("map_display_lon_axis", bundle["terrain_display_lon_axis"])
    display_lat_axis = bundle.get("map_display_lat_axis", bundle["terrain_display_lat_axis"])
    display_crs = WEB_MERCATOR_CRS if "map_display_transform" in bundle else GEOGRAPHIC_CRS
    display_rssi = np.full(display_shape, np.nan, dtype=np.float32)
    safe_reproject(
        source=np.asarray(observer_rssi, dtype=np.float32),
        destination=display_rssi,
        src_transform=bundle["projected_transform"],
        src_crs=PROJECTED_CRS,
        src_nodata=np.nan,
        dst_transform=display_transform,
        dst_crs=display_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return {
        "max_rssi": display_rssi,
        "lon_axis": display_lon_axis,
        "lat_axis": display_lat_axis,
    }


def compute_node_rssi_summaries(
    nodes,
    bundle,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
):
    if not nodes:
        return {}

    ensure_analysis_context(bundle)
    observer_frame = build_observer_frame(nodes)
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)
    use_parallel_ground_loss_numba = numba_parallel_threads_safe()

    x_step = float(abs(terrain_x[1] - terrain_x[0])) if terrain_x.size > 1 else 0.0
    y_step = float(abs(terrain_y[1] - terrain_y[0])) if terrain_y.size > 1 else 0.0
    cell_area_km2 = (x_step * y_step) / 1_000_000.0 if x_step and y_step else 0.0

    observers = list(observer_frame.itertuples(index=False))
    tasks = [
        dask.delayed(compute_observer_projected_rssi)(
            observer,
            terrain_da,
            terrain_x,
            terrain_y,
            global_rx_height_agl_m,
            global_rx_gain_dbi,
            include_ground_loss=True,
            sample_spacing_m=sample_spacing_m,
            minimum_rssi_dbm=None,
            min_distance_km=0.05,
            use_parallel_ground_loss_numba=use_parallel_ground_loss_numba,
        )
        for observer in observers
    ]
    observer_rssi_results = compute_dask_tasks(tasks, use_threads=True)

    summaries = {}
    for observer, observer_rssi in zip(observers, observer_rssi_results, strict=False):
        valid = observer_rssi[np.isfinite(observer_rssi)]
        valid_above_floor = valid[valid >= MIN_LINK_RSSI_DBM]
        summaries[str(observer.id)] = {
            "coverage_area_km2": float(np.count_nonzero(np.isfinite(observer_rssi)) * cell_area_km2),
            "peak_rssi_dbm": float(np.nanmax(valid_above_floor)) if valid_above_floor.size else None,
            "p95_rssi_dbm": float(np.nanpercentile(valid_above_floor, 95)) if valid_above_floor.size else None,
        }

    return summaries


def compute_rssi_overlay(
    nodes,
    bundle,
    include_ground_loss,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
):
    if not nodes:
        return None

    normalized_nodes = [with_node_defaults(node) for node in nodes]
    cache_key = overlay_cache_key(
        bundle,
        normalized_nodes,
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
        sample_spacing_m=sample_spacing_m,
    )
    if cache_key in RSSI_OVERLAY_CACHE:
        return RSSI_OVERLAY_CACHE[cache_key]

    ensure_analysis_context(bundle)
    observer_frame = build_observer_frame(normalized_nodes)
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    max_rssi = np.full(terrain_da.shape, np.nan, dtype=np.float32)
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)
    use_parallel_ground_loss_numba = include_ground_loss and numba_parallel_threads_safe()

    observers = list(observer_frame.itertuples(index=False))
    tasks = [
        dask.delayed(compute_observer_projected_rssi)(
            observer,
            terrain_da,
            terrain_x,
            terrain_y,
            global_rx_height_agl_m,
            global_rx_gain_dbi,
            include_ground_loss=include_ground_loss,
            sample_spacing_m=sample_spacing_m,
            minimum_rssi_dbm=MIN_LINK_RSSI_DBM,
            min_distance_km=None,
            use_parallel_ground_loss_numba=use_parallel_ground_loss_numba,
        )
        for observer in observers
    ]
    observer_rssi_results = compute_dask_tasks(tasks, use_threads=True)
    for observer_rssi in observer_rssi_results:
        max_rssi = np.fmax(max_rssi, observer_rssi)

    result = reproject_projected_rssi_to_display(max_rssi, bundle)
    RSSI_OVERLAY_CACHE[cache_key] = result
    return result


def compute_single_node_rssi_overlay_result(
    node,
    bundle,
    include_ground_loss,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
    use_parallel_ground_loss_numba=True,
):
    normalized = with_node_defaults(node)
    cache_key = overlay_cache_key(
        bundle,
        [normalized],
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
        sample_spacing_m=sample_spacing_m,
    )
    observer_frame = build_observer_frame([normalized])
    terrain_da = ANALYSIS_CONTEXT["terrain_da"]
    terrain_x = ANALYSIS_CONTEXT["terrain_x"]
    terrain_y = ANALYSIS_CONTEXT["terrain_y"]
    observer = next(observer_frame.itertuples(index=False))
    observer_rssi = compute_observer_projected_rssi(
        observer,
        terrain_da,
        terrain_x,
        terrain_y,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
        include_ground_loss=include_ground_loss,
        sample_spacing_m=sample_spacing_m,
        minimum_rssi_dbm=MIN_LINK_RSSI_DBM,
        min_distance_km=None,
        use_parallel_ground_loss_numba=use_parallel_ground_loss_numba,
    )
    return cache_key, reproject_projected_rssi_to_display(observer_rssi, bundle)


def compute_single_node_rssi_overlay(
    node,
    bundle,
    include_ground_loss,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
):
    normalized = with_node_defaults(node)
    cache_key = overlay_cache_key(
        bundle,
        [normalized],
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
        sample_spacing_m=sample_spacing_m,
    )
    if cache_key in RSSI_OVERLAY_CACHE:
        return cache_key, RSSI_OVERLAY_CACHE[cache_key]

    ensure_analysis_context(bundle)
    sample_spacing_m = normalize_rssi_path_sample_spacing(sample_spacing_m)
    cache_key, result = compute_single_node_rssi_overlay_result(
        normalized,
        bundle,
        include_ground_loss,
        global_rx_height_agl_m,
        global_rx_gain_dbi,
        sample_spacing_m=sample_spacing_m,
        use_parallel_ground_loss_numba=include_ground_loss and numba_parallel_threads_safe(),
    )
    RSSI_OVERLAY_CACHE[cache_key] = result
    return cache_key, result


def compose_cached_rssi_overlay(bundle, calculation_store, overlay_selection_store):
    if not calculation_store:
        return None
    if tuple(calculation_store.get("bundle_key", ())) != bundle["cache_key"]:
        return None

    node_keys = calculation_store.get("node_overlay_keys", {})
    selection_store = dict(overlay_selection_store or {})
    enabled_nodes = {
        str(node_id)
        for node_id in calculation_store.get("node_order", [])
        if str(node_id) in node_keys and bool(selection_store.get(str(node_id), True))
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
        "mode": "max-rssi",
    }


def get_enabled_rssi_overlay_entries(bundle, nodes, calculation_store, overlay_selection_store):
    if not bundle or not calculation_store:
        return []
    if tuple(calculation_store.get("bundle_key", ())) != bundle["cache_key"]:
        return []

    normalized_node_list = [with_node_defaults(node) for node in (nodes or [])]
    normalized_nodes = {
        str(node["id"]): node
        for node in normalized_node_list
    }
    node_styles = {
        str(node["id"]): {
            "label": str(node["name"]),
            "color": node_color(index),
        }
        for index, node in enumerate(normalized_node_list)
    }
    node_keys = calculation_store.get("node_overlay_keys", {})
    node_signatures = calculation_store.get("node_signatures", {})
    selection_store = dict(overlay_selection_store or {})
    enabled_node_ids = [
        str(node_id)
        for node_id in calculation_store.get("node_order", [])
        if str(node_id) in node_keys and bool(selection_store.get(str(node_id), True))
    ]
    if not enabled_node_ids:
        return []

    include_ground_loss = bool(calculation_store.get("include_ground_loss", False))
    rx_height = float(calculation_store.get("global_rx_height_agl", DEFAULT_GLOBAL_RX_HEIGHT_M))
    rx_gain = float(calculation_store.get("global_rx_gain_dbi", DEFAULT_RX_GAIN_DBI))
    sample_spacing_m = normalize_rssi_path_sample_spacing(
        calculation_store.get("path_sample_spacing_m", DEFAULT_RSSI_PATH_SAMPLE_SPACING_M)
    )

    for node_id in enabled_node_ids:
        expected_key = node_keys.get(node_id)
        if expected_key in RSSI_OVERLAY_CACHE:
            continue
        node = normalized_nodes.get(node_id)
        if node is None:
            continue
        if node_signatures and node_signatures.get(node_id) != node_signature(node):
            continue
        cache_key, _overlay = compute_single_node_rssi_overlay(
            node,
            bundle,
            include_ground_loss,
            rx_height,
            rx_gain,
            sample_spacing_m=sample_spacing_m,
        )
        if expected_key and cache_key != expected_key:
            continue

    entries = []
    for node_id in enabled_node_ids:
        overlay = RSSI_OVERLAY_CACHE.get(node_keys.get(node_id))
        if overlay is None:
            continue
        style = node_styles.get(node_id, {"label": node_id, "color": node_color(len(entries))})
        entries.append(
            {
                "node_id": node_id,
                "label": style["label"],
                "color": style["color"],
                "overlay": overlay,
            }
        )
    return entries


def build_max_rssi_overlay(entries):
    if not entries:
        return None

    combined = np.array(entries[0]["overlay"]["max_rssi"], copy=True)
    lon_axis = entries[0]["overlay"]["lon_axis"]
    lat_axis = entries[0]["overlay"]["lat_axis"]
    for entry in entries[1:]:
        combined = np.fmax(combined, entry["overlay"]["max_rssi"])

    return {
        "max_rssi": combined,
        "lon_axis": lon_axis,
        "lat_axis": lat_axis,
        "mode": "max-rssi",
    }


def build_best_node_rssi_overlay(entries):
    if not entries:
        return None

    template = np.asarray(entries[0]["overlay"]["max_rssi"], dtype=np.float32)
    best_rssi = np.full(template.shape, -np.inf, dtype=np.float32)
    owner_index = np.full(template.shape, -1, dtype=np.int16)

    for entry_index, entry in enumerate(entries):
        values = np.asarray(entry["overlay"]["max_rssi"], dtype=np.float32)
        valid = np.isfinite(values)
        replace = valid & ((owner_index < 0) | (values > best_rssi))
        best_rssi[replace] = values[replace]
        owner_index[replace] = entry_index

    if not np.any(owner_index >= 0):
        return None

    return {
        "mode": "best-node",
        "max_rssi": np.where(owner_index >= 0, best_rssi, np.nan),
        "owner_index": owner_index,
        "lon_axis": entries[0]["overlay"]["lon_axis"],
        "lat_axis": entries[0]["overlay"]["lat_axis"],
        "legend_items": [
            {
                "node_id": entry["node_id"],
                "label": entry["label"],
                "color": entry["color"],
            }
            for entry in entries
        ],
    }


def resolve_rssi_overlay(bundle, nodes, calculation_store, overlay_selection_store):
    if not bundle or not calculation_store:
        return None

    overlay = compose_cached_rssi_overlay(bundle, calculation_store, overlay_selection_store)
    if overlay is not None:
        return overlay

    entries = get_enabled_rssi_overlay_entries(bundle, nodes, calculation_store, overlay_selection_store)
    return build_max_rssi_overlay(entries)


def resolve_rssi_provider_overlay(bundle, nodes, calculation_store, overlay_selection_store):
    entries = get_enabled_rssi_overlay_entries(bundle, nodes, calculation_store, overlay_selection_store)
    return build_best_node_rssi_overlay(entries)


def overlay_cache_key(
    bundle,
    nodes,
    include_ground_loss,
    global_rx_height_agl_m,
    global_rx_gain_dbi,
    sample_spacing_m=DEFAULT_RSSI_PATH_SAMPLE_SPACING_M,
):
    normalized_nodes = [with_node_defaults(node) for node in nodes]
    normalized_spacing = normalize_rssi_path_sample_spacing(sample_spacing_m)
    parts = [
        f"{bundle['cache_key']}",
        str(bool(include_ground_loss)),
        f"{float(global_rx_height_agl_m):.3f}",
        f"{float(global_rx_gain_dbi):.3f}",
    ]
    if include_ground_loss:
        parts.append(f"spacing:{normalized_spacing:.1f}")
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
    attenuation_per_sample_db = np.asarray(result["attenuation_per_sample_db"], dtype=np.float64)
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
                    "color": ATTENUATION_EVENT_COLOR,
                    "line": {"color": ATTENUATION_EVENT_EDGE_COLOR, "width": 0.75},
                    "opacity": 0.88,
                },
                customdata=np.column_stack(
                    (
                        blockage_fraction[attenuation_events],
                        attenuation_per_sample_db[attenuation_events],
                    )
                ),
                hovertemplate=(
                    "Attenuation event<br>"
                    "Distance=%{x:.3f} km<br>"
                    "Blocked fraction=%{customdata[0]:.2f}<br>"
                    "Added loss=%{customdata[1]:.2f} dB<extra></extra>"
                ),
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
                marker={"symbol": "diamond", "size": 8, "color": TERRAIN_BLOCK_COLOR},
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


def build_node_summary(nodes, selected_node_ids, overlay_selection_store):
    if not nodes:
        return html.Div("No nodes added yet.", style={"color": "#cbd5e1"})

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
                key=f"node-card-{str(node['id'])}",
            )
        )

    return html.Div(
        [header, html.Div(summary_cards, style={"display": "flex", "flexDirection": "column", "gap": "8px"})],
        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
    )


def parse_uploaded_nodes(contents):
    _content_type, content_string = contents.split(",", 1)
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


def build_node_upload_component():
    return dcc.Upload(
        id="node-upload",
        children=html.Button("Load Nodes CSV", id="load-nodes-button", style={"width": "100%"}, disabled=True),
        disabled=True,
        multiple=False,
    )


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


app = Dash(
    external_scripts=[MAPLIBRE_JS_URL],
    external_stylesheets=[MAPLIBRE_CSS_URL],
)
app.enable_dev_tools(
    debug=False,
    dev_tools_ui=False,
    dev_tools_serve_dev_bundles=False,
    dev_tools_hot_reload=False,
)
app.title = "Non Flatlander Mesh Terrain Planner (NF-MTP)"
server = app.server

werkzeug_logger = logging.getLogger("werkzeug")
if not any(isinstance(existing_filter, SuppressReloadHashFilter) for existing_filter in werkzeug_logger.filters):
    werkzeug_logger.addFilter(SuppressReloadHashFilter())


@app.server.route("/terrain-dem/<token>/<int:z>/<int:x>/<int:y>.png")
def serve_terrain_dem_tile(token, z, x, y):
    try:
        png_bytes = build_terrain_dem_tile_png(token, z, x, y)
    except Exception as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(png_bytes, mimetype="image/png")

app.layout = [
    dcc.Store(id="nodes-store", data=[]),
    dcc.Store(id="node-counter-store", data=0),
    dcc.Store(id="selected-node-ids-store", data=[]),
    dcc.Store(id="node-delete-request-store", data=None),
    dcc.Store(id="manual-node-entry-ack-store", data=None),
    dcc.Store(id="node-upload-reset-store", data=0),
    dcc.Store(id="map-click-mode-store", data={"mode": "none", "node_id": None}),
    dcc.Store(id="map-camera-store", data=DEFAULT_BBOX),
    dcc.Store(id="map-camera-revision-store", data=0),
    dcc.Store(id="map-view-store", data=DEFAULT_BBOX),
    dcc.Store(id="native-map-spec-store", data=None),
    dcc.Store(id="native-map-hover-store", data=None),
    dcc.Store(id="terrain-ready-store", data=False),
    dcc.Store(id="bbox-store", data=None),
    dcc.Store(id="bbox-preview-store", data=DEFAULT_BBOX),
    dcc.Store(id="rssi-calculation-store", data=None),
    dcc.Store(id="rssi-overlay-selection-store", data={}),
    dcc.Store(id="map-interaction-loading-store", data=False),
    dcc.Store(id="map-interaction-loading-message-store", data=DEFAULT_MAP_LOADING_MESSAGE),
    dcc.Store(id="terrain-load-notification-store", data=None),
    dcc.Store(id="rssi-run-request-store", data=None),
    dcc.Store(id="rssi-progress-meta-store", data=None),
    dcc.Store(id="rssi-progress-complete-store", data=None),
    dcc.Store(id="point-path-store", data=None),
    dcc.Store(id="viewshed-point-store", data=None),
    dcc.Store(id="viewshed-assessment-store", data=None),
    dcc.Download(id="node-download"),
    dcc.Interval(id="rssi-progress-interval", interval=1000, n_intervals=0, disabled=True),
    html.Div(
        [
            html.Img(src=app.get_asset_url("logo.svg"), className="app-logo", alt="NF-MTP logo"),
            html.H1(
                "Non Flatlander Mesh Terrain Planner (NF-MTP)",
                style={"textAlign": "center", "color": "#f9fafb", "margin": "0"},
            ),
        ],
        className="app-header",
    ),
    html.Div(id="top-banner-container"),
    html.Div(
        [
            html.Div(
                [
                    html.Details(
                        [
                            html.Summary(
                                "Terrain Overlay",
                                title="Control terrain visibility, clipping, map styling, and optional WorldCover display.",
                                style={"fontWeight": "600", "cursor": "pointer"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Adjust the terrain layer visibility and the elevation band emphasized on the map.",
                                        style={"fontSize": "12px", "color": "#cbd5e1"},
                                    ),
                                    html.Div(
                                        "Terrain-driven controls unlock after the first successful terrain load.",
                                        style={"fontSize": "12px", "color": "#94a3b8"},
                                    ),
                                    html.Div("Base map style", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.Dropdown(
                                        id="base-map-style",
                                        options=BASE_MAP_STYLE_OPTIONS,
                                        value="satellite",
                                        clearable=False,
                                    ),
                                    html.Div("Terrain overlay alpha", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.Slider(0, 1, step=0.05, value=0.45, id="elevation_alpha", disabled=True, updatemode="mouseup"),
                                    html.Div("Terrain elevation clipping range (m)", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.RangeSlider(
                                        id="elevation_clip_range",
                                        min=0,
                                        max=1,
                                        value=[0, 1],
                                        allowCross=False,
                                        tooltip={"placement": "bottom"},
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Set Clipping To Current View",
                                        id="clip-visible-elevation-range",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Div("Elevation colormap", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.Dropdown(
                                        id="elevation-colormap",
                                        options=[{"label": value, "value": value} for value in
                                                 ELEVATION_COLOR_SCALE_OPTIONS],
                                        value="Magma",
                                        clearable=False,
                                    ),
                                    dcc.Checklist(
                                        id="worldcover-display",
                                        options=[{"label": " Show WorldCover layer", "value": "enabled", "disabled": True}],
                                        value=[],
                                    ),
                                    html.Div("WorldCover opacity", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.Slider(0, 1, step=0.05, value=0.55, id="worldcover-opacity", disabled=True, updatemode="mouseup"),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "10px", "marginTop": "10px"},
                            ),
                        ],
                        id="terrain-overlay-section",
                        open=False,
                    ),
                    html.Details(
                        [
                            html.Summary(
                                "Map Settings",
                                title="Set the longitude and latitude bounds for the terrain area to load.",
                                style={"fontWeight": "600", "cursor": "pointer"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Define the geographic bounds to load, or copy the current native map viewport into the terrain bounding box.",
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
                                    html.Div("BBox resolution (m/pixel)", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                    dcc.Slider(
                                        MIN_BBOX_RESOLUTION_M,
                                        MAX_BBOX_RESOLUTION_M,
                                        step=BBOX_RESOLUTION_STEP_M,
                                        value=DEFAULT_BBOX_RESOLUTION_M,
                                        id="bbox-resolution-m",
                                        marks={
                                            int(MIN_BBOX_RESOLUTION_M): "50",
                                            100: "100",
                                            150: "150",
                                            int(MAX_BBOX_RESOLUTION_M): "200",
                                        },
                                        updatemode="mouseup",
                                    ),
                                    html.Button(
                                        "Set Terrain Bounding Box",
                                        id="toggle-terrain-bbox",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                    ),
                                    html.Div(id="terrain-bbox-status", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                    html.Button(
                                        "Load Elevation + Worldcover",
                                        id="update_graph",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#166534",
                                            "borderColor": "#22c55e",
                                            "color": "#f0fdf4",
                                        },
                                    ),
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
                                    dcc.Slider(0, 1, step=0.05, value=0.55, id="rssi-opacity", updatemode="mouseup"),
                                    html.Div("Global RX Height AGL (m)", style={"fontWeight": "600"}),
                                    dcc.Input(id="global-rx-height-agl", type="number", value=DEFAULT_GLOBAL_RX_HEIGHT_M, style={"width": "100%"}),
                                    html.Div("Global RX Antenna Gain (dBi)", style={"fontWeight": "600"}),
                                    dcc.Input(id="global-rx-gain-dbi", type="number", value=DEFAULT_RX_GAIN_DBI, style={"width": "100%"}),
                                    html.Div("Max RSSI colormap", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.Dropdown(
                                        id="rssi-colormap",
                                        options=[{"label": value, "value": value} for value in RSSI_COLOR_SCALE_OPTIONS],
                                        value="Turbo",
                                        clearable=False,
                                    ),
                                    html.Div("RSSI display mode", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.RadioItems(
                                        id="rssi-render-mode",
                                        options=RSSI_RENDER_MODE_OPTIONS,
                                        value="max-rssi",
                                        labelStyle={"display": "block", "marginBottom": "4px"},
                                        inputStyle={"marginRight": "8px"},
                                    ),
                                    html.Button(
                                        "Draw RSSI For All Nodes",
                                        id="draw-rssi-overlay",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Progress(id="rssi-progress-bar", value="0", max="100", style={"width": "100%", "height": "14px"}),
                                    html.Div(id="rssi-progress-label", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                    dcc.Checklist(
                                        id="include-rssi-ground-loss",
                                        options=[{"label": " Include per-cell path attenuation in RSSI overlay", "value": "enabled"}],
                                        value=["enabled"],
                                    ),
                                    html.Div(
                                        "Ground-loss path sampling now follows the current bbox resolution setting.",
                                        style={"fontSize": "12px", "color": "#cbd5e1"},
                                    ),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        id="rssi-calculations-section",
                        open=False,
                    ),
                    html.Details(
                        [
                            html.Summary(
                                "Viewshed Assessment",
                                title="Place an assessment point on the map and compute a Max LOS heatmap around it.",
                                style={"fontWeight": "600", "cursor": "pointer"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Place marker and select the radius and density of points to be assessed. A viewshed will be calculated for each of the sampling points to identify ideal locations for node placement.",
                                        style={"fontSize": "12px", "color": "#cbd5e1"},
                                    ),
                                    html.Button(
                                        "Place Assessment Point",
                                        id="toggle-viewshed-point",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Max Viewshed calculation",
                                        id="run-viewshed-assessment",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Div("Assessment radius (meters)", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.Slider(
                                        25,
                                        1000,
                                        step=25,
                                        value=int(VIEWSHED_ASSESSMENT_RADIUS_M),
                                        marks={value: str(value) for value in (25, 100, 250, 500, 750, 1000)},
                                        id="viewshed-radius",
                                        disabled=True,
                                        updatemode="mouseup",
                                    ),
                                    html.Div("Assessment height AGL (meters)", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.Input(
                                        id="viewshed-height-agl",
                                        type="number",
                                        min=0,
                                        step=0.5,
                                        value=VIEWSHED_ASSESSMENT_OBSERVER_HEIGHT_AGL_M,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Div("Sampling density", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.Slider(
                                        min(VIEWSHED_SAMPLE_COUNT_OPTIONS),
                                        max(VIEWSHED_SAMPLE_COUNT_OPTIONS),
                                        step=None,
                                        value=DEFAULT_VIEWSHED_SAMPLE_COUNT,
                                        marks={value: str(value) for value in VIEWSHED_SAMPLE_COUNT_OPTIONS},
                                        id="viewshed-sample-count",
                                        disabled=True,
                                        updatemode="mouseup",
                                    ),
                                    html.Div("Viewshed colormap", style={"fontWeight": "600", "marginTop": "8px"}),
                                    dcc.Dropdown(
                                        id="viewshed-colormap",
                                        options=[{"label": value, "value": value} for value in VIEWSHED_COLOR_SCALE_OPTIONS],
                                        value="Turbo",
                                        clearable=False,
                                        disabled=True,
                                    ),
                                    html.Div("Viewshed overlay opacity", style={"fontWeight": "600", "marginTop": "4px"}),
                                    dcc.Slider(
                                        0,
                                        1,
                                        step=0.05,
                                        value=0.6,
                                        id="viewshed-opacity",
                                        disabled=True,
                                        updatemode="mouseup",
                                    ),
                                    html.Button(
                                        "Remove Assessment Point + Overlay",
                                        id="clear-viewshed-assessment",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "backgroundColor": "#7f1d1d",
                                            "borderColor": "#b91c1c",
                                            "color": "#fee2e2",
                                        },
                                        disabled=True,
                                    ),
                                    html.Div(id="viewshed-assessment-status", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        id="viewshed-assessment-section",
                        open=False,
                    ),
                ],
                className="app-main-column app-sidebar-column",
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
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(id="native-map", className="native-map-canvas"),
                                    html.Div(id="native-map-bbox-selection", className="native-map-bbox-selection"),
                                    html.Div(id="native-map-path-overlay", className="native-map-path-overlay"),
                                    html.Div(id="native-map-legend", className="native-map-legend"),
                                    html.Div(
                                        "Move the cursor over the map to inspect coordinates, elevation, and RSSI.",
                                        id="native-map-hover-readout",
                                        className="native-map-hover-panel",
                                    ),
                                    html.Div(id="native-map-render-ack", style={"display": "none"}),
                                ],
                                className="native-map-shell",
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        html.Div(className="map-loading-spinner"),
                                        html.Div(DEFAULT_MAP_LOADING_MESSAGE, id="map-interaction-overlay-label"),
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
                    html.Div(id="rssi-worker", style={"display": "none"}),
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
                className="app-main-column app-map-column",
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
                                    html.Button(
                                        "Enable Click-To-Add",
                                        id="toggle-click-add",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
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
                                    dcc.Input(
                                        id="manual-node-name",
                                        type="text",
                                        placeholder="Node name",
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    dcc.Input(
                                        id="manual-node-lon",
                                        type="number",
                                        placeholder="Longitude",
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    dcc.Input(
                                        id="manual-node-lat",
                                        type="number",
                                        placeholder="Latitude",
                                        style={"width": "100%"},
                                        disabled=True,
                                    ),
                                    html.Button("Add Node", id="add-manual-node", n_clicks=0, style={"width": "100%"}, disabled=True),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        open=False,
                    ),
                    html.Div(
                        dcc.Loading(html.Div(id="node-summary"), type="default"),
                        style={"paddingRight": "4px"},
                    ),
                    html.Details(
                        [
                            html.Summary("CSV Import / Export", style={"fontWeight": "600", "cursor": "pointer"}),
                            html.Div(
                                [
                                    build_node_upload_component(),
                                    html.Button("Save Nodes CSV", id="save-nodes", n_clicks=0, style={"width": "100%"}),
                                    html.Div(id="node-action-message", style={"fontSize": "12px", "color": "#cbd5e1"}),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "10px"},
                            ),
                        ],
                        open=False,
                    ),
                ],
                className="app-main-column app-sidebar-column",
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
        className="app-main-layout",
        style={"display": "flex", "alignItems": "stretch", "gap": "20px", "padding": "0 20px 20px"},
    ),
]


@app.callback(
    Output("node-upload", "contents"),
    Output("node-upload", "filename"),
    Output("node-upload", "last_modified"),
    Input("node-upload-reset-store", "data"),
    prevent_initial_call=True,
)
def clear_node_upload_selection(_reset_revision):
    return None, None, None


@app.callback(
    Output("terrain-ready-store", "data"),
    Input("native-map-spec-store", "data"),
    State("terrain-ready-store", "data"),
)
def update_terrain_ready_store(spec, current_value):
    if current_value:
        return True
    if spec and spec.get("terrain_dem") is not None:
        return True
    return False


@app.callback(
    Output("toggle-click-add", "disabled"),
    Output("manual-node-name", "disabled"),
    Output("manual-node-lon", "disabled"),
    Output("manual-node-lat", "disabled"),
    Output("add-manual-node", "disabled"),
    Output("node-upload", "disabled"),
    Output("load-nodes-button", "disabled"),
    Output("elevation_alpha", "disabled"),
    Output("elevation_clip_range", "disabled"),
    Output("clip-visible-elevation-range", "disabled"),
    Output("worldcover-display", "options"),
    Output("worldcover-opacity", "disabled"),
    Output("draw-rssi-overlay", "disabled"),
    Input("terrain-ready-store", "data"),
)
def update_locked_control_state(terrain_ready):
    disabled = not bool(terrain_ready)
    worldcover_options = [{"label": " Show WorldCover layer", "value": "enabled", "disabled": disabled}]
    return (
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        disabled,
        worldcover_options,
        disabled,
        disabled,
    )


@app.callback(
    Output("bbox-preview-store", "data"),
    Input("min_lon", "value"),
    Input("min_lat", "value"),
    Input("max_lon", "value"),
    Input("max_lat", "value"),
)
def update_bbox_preview_store(min_lon, min_lat, max_lon, max_lat):
    try:
        return bbox_dict(
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            }
        )
    except Exception:
        raise PreventUpdate


@app.callback(
    Output("bbox-store", "data"),
    Output("map-camera-store", "data", allow_duplicate=True),
    Output("map-view-store", "data", allow_duplicate=True),
    Output("map-camera-revision-store", "data", allow_duplicate=True),
    Input("update_graph", "n_clicks"),
    State("min_lon", "value"),
    State("min_lat", "value"),
    State("max_lon", "value"),
    State("max_lat", "value"),
    State("map-camera-revision-store", "data"),
    prevent_initial_call=True,
)
def update_bbox_store(_n_clicks, min_lon, min_lat, max_lon, max_lat, camera_revision):
    bbox = bbox_dict(
        {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
    )
    return bbox, bbox, bbox, next_revision(camera_revision)


@app.callback(
    Output("elevation_clip_range", "min"),
    Output("elevation_clip_range", "max"),
    Output("elevation_clip_range", "value"),
    Output("elevation_clip_range", "marks"),
    Input("bbox-store", "data"),
    Input("bbox-resolution-m", "value"),
)
def update_elevation_clip_controls(bbox_data, bbox_resolution_m):
    if not bbox_data:
        return 0, 1, [0, 1], {0: "0", 1: "1"}

    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox, bbox_resolution_m)
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
    State("map-view-store", "data"),
    State("bbox-store", "data"),
    State("bbox-resolution-m", "value"),
    prevent_initial_call=True,
)
def set_elevation_clip_to_current_view(_n_clicks, map_view_data, bbox_data, bbox_resolution_m):
    if not bbox_data:
        raise PreventUpdate

    bbox = normalize_bbox(
        bbox_data["min_lon"],
        bbox_data["min_lat"],
        bbox_data["max_lon"],
        bbox_data["max_lat"],
    )
    bundle = get_map_bundle(*bbox, bbox_resolution_m)

    terrain_lon = bundle["terrain_display_lon_axis"]
    terrain_lat = bundle["terrain_display_lat_axis"][::-1]
    terrain_display = np.flipud(bundle["terrain_display"])

    current_view = map_view_data or bbox_data
    x_range = (
        float(current_view["min_lon"]),
        float(current_view["max_lon"]),
    )
    y_range = (
        float(current_view["min_lat"]),
        float(current_view["max_lat"]),
    )

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
    Input("bbox-resolution-m", "value"),
    Input("nodes-store", "data"),
    Input("rssi-calculation-store", "data"),
)
def update_top_banner(bbox_data, bbox_resolution_m, nodes, calculation_store):
    outside_nodes = nodes_outside_loaded_bbox(nodes or [], bbox_data) if nodes else []
    if outside_nodes:
        names = ", ".join(outside_nodes[:4])
        suffix = "" if len(outside_nodes) <= 4 else f", and {len(outside_nodes) - 4} more"
        return info_banner(
            f"Node(s) {names}{suffix} are outside the loaded terrain area. Expand or reload the terrain bounds before using terrain-based analysis."
        )

    if not bbox_data:
        return info_banner(
            "The base map is ready. Use the map settings or copy the current native map viewport into the load bounds, then load elevation + worldcover data for terrain analysis.",
            background_color="#1d4ed8",
        )

    if calculation_store:
        current_signatures = {str(node["id"]): node_signature(node) for node in (nodes or [])}
        calculated_signatures = calculation_store.get("node_signatures", {})
        if current_signatures != calculated_signatures:
            return info_banner(
                "Nodes have been added or changed since the last RSSI calculation. Re-run RSSI calculations to refresh the overlay."
            )
        if bbox_data:
            current_bundle_key = bundle_cache_key(
                bbox_data["min_lon"],
                bbox_data["min_lat"],
                bbox_data["max_lon"],
                bbox_data["max_lat"],
                bbox_resolution_m,
            )
            if tuple(calculation_store.get("bundle_key", ())) != current_bundle_key:
                return info_banner(
                    "Terrain bounds or bbox resolution changed since the last RSSI calculation. Re-run RSSI calculations to refresh the overlay."
                )
    return None


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("toggle-click-add", "n_clicks"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_click_add_mode(_n_clicks, click_mode):
    return toggle_click_mode_state(click_mode, "add-node")


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("toggle-terrain-bbox", "n_clicks"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_terrain_bbox_mode(_n_clicks, click_mode):
    return toggle_click_mode_state(click_mode, "terrain-bbox")


@app.callback(
    Output("toggle-click-add", "children"),
    Output("click-add-status", "children"),
    Input("map-click-mode-store", "data"),
)
def update_click_add_text(click_mode):
    return click_mode_button_copy(
        click_mode,
        "add-node",
        "Disable Click-To-Add",
        "Click on the map to add one node. This mode turns off after a successful add.",
        "Enable Click-To-Add",
        "Click-to-add is off.",
    )


@app.callback(
    Output("toggle-terrain-bbox", "children"),
    Output("terrain-bbox-status", "children"),
    Input("map-click-mode-store", "data"),
)
def update_terrain_bbox_text(click_mode):
    return click_mode_button_copy(
        click_mode,
        "terrain-bbox",
        "Cancel Terrain Bounding Box Selection",
        "Drag a box on the native map. Release to fill the terrain bounding box.",
        "Set Terrain Bounding Box",
        "Use this to drag out the terrain bounding box directly on the map.",
    )


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("toggle-viewshed-point", "n_clicks"),
    State("map-click-mode-store", "data"),
    prevent_initial_call=True,
)
def toggle_viewshed_point_mode(_n_clicks, click_mode):
    return toggle_click_mode_state(click_mode, "viewshed-point")


@app.callback(
    Output("toggle-viewshed-point", "children"),
    Output("toggle-viewshed-point", "disabled"),
    Output("run-viewshed-assessment", "disabled"),
    Output("clear-viewshed-assessment", "disabled"),
    Output("viewshed-radius", "disabled"),
    Output("viewshed-height-agl", "disabled"),
    Output("viewshed-sample-count", "disabled"),
    Output("viewshed-colormap", "disabled"),
    Output("viewshed-opacity", "disabled"),
    Output("viewshed-assessment-status", "children"),
    Input("map-click-mode-store", "data"),
    Input("viewshed-point-store", "data"),
    Input("viewshed-assessment-store", "data"),
    Input("terrain-ready-store", "data"),
    Input("bbox-store", "data"),
    Input("bbox-resolution-m", "value"),
    Input("viewshed-radius", "value"),
    Input("viewshed-height-agl", "value"),
    Input("viewshed-sample-count", "value"),
)
def update_viewshed_controls(
    click_mode,
    point_data,
    assessment_store,
    terrain_ready,
    bbox_data,
    bbox_resolution_m,
    viewshed_radius,
    viewshed_height_agl,
    viewshed_sample_count,
):
    mode_active = click_mode_value(click_mode) == "viewshed-point"
    terrain_ready = bool(terrain_ready)
    has_point = bool(point_data and point_data.get("longitude") is not None and point_data.get("latitude") is not None)
    viewshed_radius = float(viewshed_radius or VIEWSHED_ASSESSMENT_RADIUS_M)
    viewshed_height_agl = float(viewshed_height_agl or 0.0)
    effective_sample_count = effective_viewshed_sample_count(viewshed_sample_count)
    current_bundle_key = None
    if bbox_data:
        try:
            current_bundle_key = bundle_cache_key(
                bbox_data["min_lon"],
                bbox_data["min_lat"],
                bbox_data["max_lon"],
                bbox_data["max_lat"],
                bbox_resolution_m,
            )
        except Exception:
            current_bundle_key = None
    radius_matches = bool(
        assessment_store
        and assessment_store.get("radius_m") is not None
        and math.isclose(float(assessment_store["radius_m"]), viewshed_radius, rel_tol=0.0, abs_tol=1e-6)
    )
    height_matches = bool(
        assessment_store
        and assessment_store.get("observer_height_agl") is not None
        and math.isclose(
            float(assessment_store["observer_height_agl"]),
            viewshed_height_agl,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    sample_count_matches = bool(
        assessment_store
        and assessment_store.get("sample_count") is not None
        and int(assessment_store["sample_count"]) == int(effective_sample_count)
    )
    has_current_assessment = bool(
        assessment_store
        and assessment_store.get("cache_key")
        and radius_matches
        and height_matches
        and sample_count_matches
        and current_bundle_key is not None
        and tuple(assessment_store.get("bundle_key", ())) == tuple(current_bundle_key)
    )
    button_label = "Cancel Assessment Point Placement" if mode_active else "Place Assessment Point"

    if not terrain_ready:
        status = "Load terrain before running a viewshed assessment."
    elif assessment_store and assessment_store.get("error"):
        status = str(assessment_store.get("error"))
    elif mode_active:
        status = "Click once on the map to place the viewshed assessment point."
    elif has_point and has_current_assessment:
        status = (
            f"Assessment point set at ({float(point_data['longitude']):.5f}, "
            f"{float(point_data['latitude']):.5f}) with a {int(round(viewshed_radius))} m radius at "
            f"{viewshed_height_agl:.1f} m AGL using {effective_sample_count} samples. Max LOS overlay is displayed."
        )
    elif has_point and assessment_store and assessment_store.get("cache_key"):
        status = (
            f"Assessment point set at ({float(point_data['longitude']):.5f}, "
            f"{float(point_data['latitude']):.5f}). Re-run Max LOS calculation for the current terrain, "
            f"{int(round(viewshed_radius))} m radius, {viewshed_height_agl:.1f} m AGL, and {effective_sample_count} samples."
        )
    elif has_point:
        status = (
            f"Assessment point set at ({float(point_data['longitude']):.5f}, "
            f"{float(point_data['latitude']):.5f}). Click Max LOS calculation to generate the "
            f"{int(round(viewshed_radius))} m overlay at {viewshed_height_agl:.1f} m AGL using {effective_sample_count} samples."
        )
    else:
        status = (
            f"No viewshed assessment point placed. Current radius is {int(round(viewshed_radius))} m "
            f"at {viewshed_height_agl:.1f} m AGL using {effective_sample_count} samples."
        )
    return (
        button_label,
        not terrain_ready,
        (not terrain_ready) or (not has_point),
        (not has_point) and not bool(assessment_store and assessment_store.get("cache_key")),
        not terrain_ready,
        not terrain_ready,
        not terrain_ready,
        not has_current_assessment,
        not has_current_assessment,
        status,
    )


@app.callback(
    Output("viewshed-point-store", "data", allow_duplicate=True),
    Output("viewshed-assessment-store", "data", allow_duplicate=True),
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("clear-viewshed-assessment", "n_clicks"),
    prevent_initial_call=True,
)
def clear_viewshed_assessment(_n_clicks):
    if not _n_clicks:
        raise PreventUpdate
    return None, None, {"mode": "none", "node_id": None}


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
    Output("node-delete-request-store", "data"),
    Input({"type": "delete-node-button", "node_id": ALL}, "n_clicks_timestamp"),
    State({"type": "delete-node-button", "node_id": ALL}, "id"),
    State("node-delete-request-store", "data"),
    prevent_initial_call=True,
)
def capture_node_delete_request(_timestamps, button_ids, current_request):
    timestamps = _timestamps or []
    button_ids = button_ids or []
    latest_timestamp = int((current_request or {}).get("timestamp") or -1)
    next_request = None

    for button_id, timestamp in zip(button_ids, timestamps, strict=False):
        if timestamp is None:
            continue
        timestamp = int(timestamp)
        if timestamp <= latest_timestamp:
            continue
        next_request = {
            "node_id": str(button_id.get("node_id")),
            "timestamp": timestamp,
        }

    if next_request is None:
        raise PreventUpdate
    return next_request


@app.callback(
    Output("nodes-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("node-delete-request-store", "data"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def delete_node(delete_request, nodes):
    if not delete_request or "node_id" not in delete_request:
        raise PreventUpdate

    node_id = str(delete_request["node_id"])
    updated_nodes = [node for node in (nodes or []) if str(node["id"]) != node_id]
    if len(updated_nodes) == len(nodes or []):
        raise PreventUpdate
    return updated_nodes, "Node deleted.", {"mode": "none", "node_id": None}


@app.callback(
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Input({"type": "node-select", "node_id": ALL}, "n_clicks"),
    State("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def update_selected_nodes(_n_clicks, selected_node_ids, nodes):
    triggered = ctx.triggered_id
    valid_ids = [str(node["id"]) for node in (nodes or [])]
    if isinstance(triggered, dict) and "node_id" in triggered:
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        node_id = str(triggered["node_id"])
        if node_id not in set(valid_ids):
            raise PreventUpdate
        return select_primary_node(selected_node_ids, node_id)
    raise PreventUpdate


@app.callback(
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Output("selected-node-ids-store", "data", allow_duplicate=True),
    Output("point-path-store", "data", allow_duplicate=True),
    Output("node-action-message", "children", allow_duplicate=True),
    Input("draw-point-path", "n_clicks"),
    Input({"type": "move-node-button", "node_id": ALL}, "n_clicks"),
    State("selected-node-ids-store", "data"),
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def update_map_click_mode(draw_point_clicks, _move_clicks, selected_node_ids, nodes):
    triggered = ctx.triggered_id

    if triggered == "draw-point-path":
        if not draw_point_clicks:
            raise PreventUpdate
        normalized_nodes = [with_node_defaults(node) for node in (nodes or [])]
        selected_node_ids = normalize_selected_node_ids(selected_node_ids, [node["id"] for node in normalized_nodes])
        primary_id = str(selected_node_ids[-1]) if selected_node_ids else None
        primary = find_node(normalized_nodes, primary_id)
        if primary is None:
            return no_update, no_update, no_update, "Select a primary node before drawing a path to a map point."
        return {
            "mode": "point-path",
            "node_id": str(primary["id"]),
        }, [str(primary["id"])], None, f"Click once on the map to trace from {primary['name']}."

    if isinstance(triggered, dict) and triggered.get("type") == "move-node-button":
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        node = find_node(nodes or [], triggered["node_id"])
        if node is None:
            raise PreventUpdate
        return {
            "mode": "move-node",
            "node_id": str(node["id"]),
        }, no_update, no_update, f"Click once on the map to move {node['name']}."

    raise PreventUpdate


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
    State("nodes-store", "data"),
    prevent_initial_call=True,
)
def update_node_config(_values, nodes):
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
    State("rssi-overlay-selection-store", "data"),
    prevent_initial_call=True,
)
def update_rssi_overlay_selection(_values, selection_store):
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
    Output("map-camera-store", "data", allow_duplicate=True),
    Output("map-view-store", "data", allow_duplicate=True),
    Output("map-camera-revision-store", "data", allow_duplicate=True),
    Output("node-upload-reset-store", "data", allow_duplicate=True),
    Output("manual-node-name", "value"),
    Output("manual-node-lon", "value"),
    Output("manual-node-lat", "value"),
    Output("node-action-message", "children"),
    Input("node-upload", "contents"),
    State("node-upload", "filename"),
    State("nodes-store", "data"),
    State("node-counter-store", "data"),
    State("selected-node-ids-store", "data"),
    State("node-upload-reset-store", "data"),
    State("map-view-store", "data"),
    State("map-camera-revision-store", "data"),
    prevent_initial_call=True,
)
def manage_nodes(
    upload_contents,
    upload_filename,
    nodes,
    counter,
    selected_node_ids,
    node_upload_reset_revision,
    current_map_view,
    camera_revision,
):
    nodes = list(nodes or [])
    counter = int(counter or 0)
    if not upload_contents:
        raise PreventUpdate
    try:
        frame = parse_uploaded_nodes(upload_contents)
    except Exception as exc:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            f"CSV load failed: {exc}",
        )
    loaded_nodes = [with_node_defaults(node) for node in nodes]
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
    message = f"Loaded {len(frame)} nodes from {upload_filename or 'CSV'}."
    next_selected = select_primary_node(selected_node_ids, last_loaded_id) if last_loaded_id is not None else (selected_node_ids or [])
    next_view = fit_bbox_for_nodes(loaded_nodes) or current_map_view or DEFAULT_BBOX
    return (
        loaded_nodes,
        counter,
        next_selected,
        next_view,
        next_view,
        next_revision(camera_revision),
        next_revision(node_upload_reset_revision),
        "",
        None,
        None,
        message,
    )


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
    Output("native-map-spec-store", "data"),
    Output("native-map-legend", "children"),
    Input("bbox-store", "data"),
    Input("bbox-resolution-m", "value"),
    Input("bbox-preview-store", "data"),
    Input("viewshed-point-store", "data"),
    Input("viewshed-assessment-store", "data"),
    Input("nodes-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("elevation_clip_range", "value"),
    Input("elevation_alpha", "value"),
    Input("worldcover-opacity", "value"),
    Input("viewshed-opacity", "value"),
    Input("viewshed-radius", "value"),
    Input("viewshed-height-agl", "value"),
    Input("viewshed-sample-count", "value"),
    Input("rssi-opacity", "value"),
    Input("worldcover-display", "value"),
    Input("base-map-style", "value"),
    Input("elevation-colormap", "value"),
    Input("viewshed-colormap", "value"),
    Input("rssi-calculation-store", "data"),
    Input("rssi-overlay-selection-store", "data"),
    Input("rssi-colormap", "value"),
    Input("rssi-render-mode", "value"),
    Input("point-path-store", "data"),
    State("global-rx-height-agl", "value"),
    State("global-rx-gain-dbi", "value"),
)
def update_native_map_spec(
    bbox_data,
    bbox_resolution_m,
    bbox_preview_data,
    viewshed_point_data,
    viewshed_assessment_store,
    nodes,
    selected_node_ids,
    terrain_clip_range,
    terrain_alpha,
    worldcover_opacity,
    viewshed_opacity,
    viewshed_radius,
    viewshed_height_agl,
    viewshed_sample_count,
    rssi_opacity,
    worldcover_display,
    base_map_style,
    elevation_colormap,
    viewshed_colormap,
    rssi_calculation_store,
    rssi_overlay_selection_store,
    rssi_colormap,
    rssi_render_mode,
    point_path_data,
    global_rx_height_agl,
    global_rx_gain_dbi,
):
    nodes = list(nodes or [])
    bundle = None
    rssi_overlay = None
    viewshed_overlay = None
    bbox_outline = bbox_preview_data or bbox_data
    overlay_context_key = native_map_overlay_context_key(
        nodes,
        selected_node_ids,
        point_path_data,
        viewshed_point_data,
        viewshed_assessment_store,
    )
    visual_context_key = native_map_visual_context_key(
        bbox_data,
        bbox_resolution_m,
        terrain_alpha,
        terrain_clip_range,
        worldcover_display,
        worldcover_opacity,
        base_map_style,
        elevation_colormap,
        rssi_calculation_store,
        rssi_overlay_selection_store,
        rssi_opacity,
        rssi_colormap,
        rssi_render_mode,
        {
            "viewshed_opacity": viewshed_opacity,
            "viewshed_colormap": viewshed_colormap,
            "viewshed_radius": viewshed_radius,
            "viewshed_height_agl": viewshed_height_agl,
            "viewshed_sample_count": viewshed_sample_count,
        },
    )
    try:
        if bbox_data:
            terrain_bbox = normalize_bbox(
                bbox_data["min_lon"],
                bbox_data["min_lat"],
                bbox_data["max_lon"],
                bbox_data["max_lat"],
            )
            bundle = get_map_bundle(*terrain_bbox, bbox_resolution_m)
            if str(rssi_render_mode or "max-rssi") == "best-node":
                rssi_overlay = resolve_rssi_provider_overlay(
                    bundle,
                    nodes,
                    rssi_calculation_store,
                    rssi_overlay_selection_store or {},
                )
            else:
                rssi_overlay = resolve_rssi_overlay(bundle, nodes, rssi_calculation_store, rssi_overlay_selection_store or {})
            viewshed_overlay = resolve_viewshed_assessment(
                bundle,
                viewshed_assessment_store,
                viewshed_radius,
                viewshed_height_agl,
                effective_viewshed_sample_count(viewshed_sample_count),
            )

        path_overlay = compute_native_map_path_overlay(
            nodes,
            selected_node_ids,
            point_path_data,
            bundle,
            global_rx_height_agl,
            global_rx_gain_dbi,
        )

        spec = build_native_map_spec(
            bundle,
            nodes,
            0.0 if terrain_alpha is None else float(terrain_alpha),
            terrain_clip_range=terrain_clip_range,
            elevation_colorscale=elevation_colormap or "Magma",
            worldcover_enabled="enabled" in (worldcover_display or []),
            worldcover_opacity=0.0 if worldcover_opacity is None else float(worldcover_opacity),
            viewshed_point_data=viewshed_point_data,
            viewshed_radius_m=viewshed_radius,
            viewshed_sample_count=viewshed_sample_count,
            viewshed_overlay=viewshed_overlay,
            viewshed_colorscale=viewshed_colormap or "Turbo",
            viewshed_opacity=0.0 if viewshed_opacity is None else float(viewshed_opacity),
            rssi_overlay=rssi_overlay,
            rssi_colorscale=rssi_colormap or "Turbo",
            loaded_bbox=bbox_outline,
            selected_node_ids=selected_node_ids,
            base_map_style=base_map_style or "satellite",
            rssi_opacity=0.55 if rssi_opacity is None else float(rssi_opacity),
            path_overlay=path_overlay,
            overlay_context_key=overlay_context_key,
            visual_context_key=visual_context_key,
        )
        legends = build_native_map_legends(
            spec.get("terrain_legend"),
            spec.get("worldcover_legend"),
            spec.get("rssi_legend"),
            spec.get("viewshed_legend"),
        )
        return spec, legends
    except Exception as exc:
        try:
            fallback_path_overlay = compute_native_map_path_overlay(
                nodes,
                selected_node_ids,
                point_path_data,
                None,
                global_rx_height_agl,
                global_rx_gain_dbi,
            )
        except Exception:
            fallback_path_overlay = empty_native_path_overlay()

        if bundle is not None:
            try:
                fallback_spec = build_native_map_spec(
                    bundle,
                    nodes,
                    0.0 if terrain_alpha is None else float(terrain_alpha),
                    terrain_clip_range=terrain_clip_range,
                    elevation_colorscale=elevation_colormap or "Magma",
                    worldcover_enabled="enabled" in (worldcover_display or []),
                    worldcover_opacity=0.0 if worldcover_opacity is None else float(worldcover_opacity),
                    viewshed_point_data=viewshed_point_data,
                    viewshed_radius_m=viewshed_radius,
                    viewshed_sample_count=viewshed_sample_count,
                    viewshed_overlay=None,
                    viewshed_colorscale=viewshed_colormap or "Turbo",
                    viewshed_opacity=0.0 if viewshed_opacity is None else float(viewshed_opacity),
                    rssi_overlay=None,
                    rssi_colorscale=rssi_colormap or "Turbo",
                    loaded_bbox=bbox_outline,
                    selected_node_ids=selected_node_ids,
                    base_map_style=base_map_style or "satellite",
                    rssi_opacity=0.55 if rssi_opacity is None else float(rssi_opacity),
                    path_overlay=fallback_path_overlay,
                    overlay_context_key=overlay_context_key,
                    visual_context_key=visual_context_key,
                )
                fallback_spec["error"] = f"Failed to refresh native map overlays: {exc}"
                legends = build_native_map_legends(
                    fallback_spec.get("terrain_legend"),
                    fallback_spec.get("worldcover_legend"),
                    fallback_spec.get("rssi_legend"),
                    fallback_spec.get("viewshed_legend"),
                )
                return fallback_spec, legends
            except Exception:
                pass

        fallback_spec = {
            "base_style_key": str(base_map_style or "satellite"),
            "base_style": build_maplibre_style(base_map_style),
            "terrain_dem": None,
            "terrain_layer": None,
            "worldcover_layer": None,
            "viewshed_layer": None,
            "rssi_layer": None,
            "nodes": build_node_feature_collection(nodes, selected_node_ids),
            "viewshed_point": build_viewshed_point_feature_collection(viewshed_point_data),
            "viewshed_radius_outline": build_viewshed_radius_feature_collection(viewshed_point_data, viewshed_radius),
            "viewshed_samples": build_viewshed_sample_feature_collection(viewshed_point_data, viewshed_radius, viewshed_sample_count),
            "loaded_bbox": build_loaded_bbox_feature_collection(loaded_bbox=bbox_outline),
            "path_line": fallback_path_overlay.get("line", feature_collection()),
            "attenuation_points": fallback_path_overlay.get("attenuation_points", feature_collection()),
            "terrain_block_points": fallback_path_overlay.get("terrain_block_points", feature_collection()),
            "terrain_legend": None,
            "worldcover_legend": None,
            "viewshed_legend": None,
            "rssi_legend": None,
            "overlay_context_key": overlay_context_key,
            "visual_context_key": visual_context_key,
            "error": f"Failed to refresh native map overlays: {exc}",
        }
        return fallback_spec, html.Div()

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="renderNativeMap"),
    Output("native-map-render-ack", "children"),
    Input("native-map-spec-store", "data"),
    Input("map-camera-store", "data"),
    Input("map-camera-revision-store", "data"),
    Input("nodes-store", "data"),
    Input("node-counter-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("point-path-store", "data"),
    Input("map-click-mode-store", "data"),
    State("viewshed-point-store", "data"),
    State("viewshed-assessment-store", "data"),
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="applyManualNodeEntry"),
    Output("manual-node-name", "value", allow_duplicate=True),
    Output("manual-node-lon", "value", allow_duplicate=True),
    Output("manual-node-lat", "value", allow_duplicate=True),
    Output("manual-node-entry-ack-store", "data", allow_duplicate=True),
    Input("add-manual-node", "n_clicks"),
    State("manual-node-name", "value"),
    State("manual-node-lon", "value"),
    State("manual-node-lat", "value"),
    State("nodes-store", "data"),
    State("node-counter-store", "data"),
    State("selected-node-ids-store", "data"),
    prevent_initial_call=True,
)

@app.callback(
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Output("map-interaction-loading-message-store", "data", allow_duplicate=True),
    Input("update_graph", "n_clicks"),
    Input("bbox-resolution-m", "value"),
    State("bbox-store", "data"),
    prevent_initial_call=True,
)
def start_map_loading(_n_clicks, _bbox_resolution_m, bbox_data):
    if ctx.triggered_id == "bbox-resolution-m" and not bbox_data:
        raise PreventUpdate
    return True, DEFAULT_MAP_LOADING_MESSAGE


@app.callback(
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Output("map-interaction-loading-message-store", "data", allow_duplicate=True),
    Input("run-viewshed-assessment", "n_clicks"),
    prevent_initial_call=True,
)
def start_viewshed_loading(_n_clicks):
    return True, "Computing Max LOS assessment..."


@app.callback(
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Output("map-interaction-loading-message-store", "data", allow_duplicate=True),
    Input("native-map-render-ack", "children"),
    State("map-interaction-loading-store", "data"),
    prevent_initial_call=True,
)
def clear_map_interaction_loading(_ack, is_loading):
    if is_loading:
        return False, DEFAULT_MAP_LOADING_MESSAGE
    raise PreventUpdate


@app.callback(
    Output("terrain-load-notification-store", "data"),
    Input("native-map-render-ack", "children"),
    State("map-interaction-loading-store", "data"),
    State("map-interaction-loading-message-store", "data"),
    State("bbox-store", "data"),
    State("bbox-resolution-m", "value"),
    State("terrain-load-notification-store", "data"),
    prevent_initial_call=True,
)
def notify_terrain_load_complete(_ack, is_loading, loading_message, bbox_data, bbox_resolution_m, notification_state):
    if not is_loading or str(loading_message or "") != DEFAULT_MAP_LOADING_MESSAGE:
        raise PreventUpdate
    ack_value = str(_ack or "")
    current_state = notification_state or {}
    if not ack_value or current_state.get("ack") == ack_value:
        raise PreventUpdate

    message = "Elevation and WorldCover data finished loading."
    if bbox_data:
        try:
            bbox = normalize_bbox(
                bbox_data["min_lon"],
                bbox_data["min_lat"],
                bbox_data["max_lon"],
                bbox_data["max_lat"],
            )
            message = (
                "Elevation and WorldCover data finished loading for "
                f"({bbox[0]:.4f}, {bbox[1]:.4f}) to ({bbox[2]:.4f}, {bbox[3]:.4f}) "
                f"at {normalize_bbox_resolution_m(bbox_resolution_m):.0f} m/pixel."
            )
        except Exception:
            pass

    send_desktop_notification("Terrain Load Complete", message)
    return {"ack": ack_value}


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
    Output("map-interaction-overlay-label", "children"),
    Input("map-interaction-loading-message-store", "data"),
)
def update_map_interaction_overlay_label(message):
    return str(message or DEFAULT_MAP_LOADING_MESSAGE)


@app.callback(
    Output("native-map-hover-readout", "children"),
    Input("native-map-hover-store", "data"),
    State("bbox-store", "data"),
    State("bbox-resolution-m", "value"),
    State("nodes-store", "data"),
    State("rssi-calculation-store", "data"),
    State("rssi-overlay-selection-store", "data"),
)
def update_native_map_hover_readout(hover_data, bbox_data, bbox_resolution_m, nodes, calculation_store, overlay_selection_store):
    if not hover_data:
        return "Move the cursor over the map to inspect coordinates, elevation, and RSSI."

    longitude = hover_data.get("longitude")
    latitude = hover_data.get("latitude")
    if longitude is None or latitude is None:
        return "Move the cursor over the map to inspect coordinates, elevation, and RSSI."

    feature = hover_data.get("feature") or {}
    rows = []
    if feature.get("name"):
        rows.append(html.Div(str(feature["name"]), className="native-map-hover-title"))
    rows.extend(
        [
            html.Div(f"Longitude {float(longitude):.5f}"),
            html.Div(f"Latitude {float(latitude):.5f}"),
        ]
    )

    terrain_value = None
    rssi_value = None
    if bbox_data:
        try:
            bbox = normalize_bbox(
                bbox_data["min_lon"],
                bbox_data["min_lat"],
                bbox_data["max_lon"],
                bbox_data["max_lat"],
            )
            bundle = get_map_bundle(*bbox, bbox_resolution_m)
            terrain_value = sample_grid_value(
                bundle["terrain_display"],
                bundle["terrain_display_lon_axis"],
                bundle["terrain_display_lat_axis"],
                longitude,
                latitude,
            )
            rssi_overlay = resolve_rssi_overlay(bundle, nodes or [], calculation_store, overlay_selection_store or {})
            if rssi_overlay is not None:
                rssi_value = sample_grid_value(
                    rssi_overlay["max_rssi"],
                    rssi_overlay["lon_axis"],
                    rssi_overlay["lat_axis"],
                    longitude,
                    latitude,
                )
        except Exception:
            terrain_value = None
            rssi_value = None

    rows.append(html.Div(f"Elevation {terrain_value:.1f} m" if terrain_value is not None else "Elevation n/a"))
    rows.append(html.Div(f"RSSI {rssi_value:.1f} dBm" if rssi_value is not None else "RSSI n/a"))

    if feature.get("kind") == "attenuation-event":
        rows.append(html.Div(f"Blocked fraction {float(feature.get('blocked_fraction', 0.0)):.2f}"))
        rows.append(html.Div(f"Added loss {float(feature.get('added_loss_db', 0.0)):.2f} dB"))
    elif feature.get("kind") == "terrain-los-block":
        rows.append(html.Div(f"Added loss {float(feature.get('added_loss_db', 0.0)):.2f} dB"))

    return rows


@app.callback(
    Output("viewshed-assessment-store", "data"),
    Output("map-click-mode-store", "data", allow_duplicate=True),
    Input("run-viewshed-assessment", "n_clicks"),
    State("viewshed-point-store", "data"),
    State("bbox-store", "data"),
    State("bbox-resolution-m", "value"),
    State("viewshed-radius", "value"),
    State("viewshed-height-agl", "value"),
    State("viewshed-sample-count", "value"),
    prevent_initial_call=True,
)
def generate_viewshed_assessment(
    run_clicks,
    point_data,
    bbox_data,
    bbox_resolution_m,
    viewshed_radius,
    viewshed_height_agl,
    viewshed_sample_count,
):
    if not run_clicks:
        raise PreventUpdate
    if not point_data or point_data.get("longitude") is None or point_data.get("latitude") is None:
        return {"error": "Place a viewshed assessment point before running Max LOS calculation."}, no_update
    if not bbox_data:
        return {"error": "Load elevation + worldcover data before running Max LOS calculation."}, {"mode": "none", "node_id": None}
    viewshed_radius = float(viewshed_radius or VIEWSHED_ASSESSMENT_RADIUS_M)
    viewshed_height_agl = float(viewshed_height_agl or VIEWSHED_ASSESSMENT_OBSERVER_HEIGHT_AGL_M)
    effective_sample_count = effective_viewshed_sample_count(viewshed_sample_count)
    started_at = time.perf_counter()

    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox, bbox_resolution_m)
        cache_key, result = compute_viewshed_assessment(
            point_data,
            bundle,
            radius_m=viewshed_radius,
            observer_height_agl=viewshed_height_agl,
            sample_count=effective_sample_count,
        )
        if not cache_key or result is None:
            return {"error": "Max LOS calculation did not produce any visible-cell values."}, {"mode": "none", "node_id": None}
        elapsed = time.perf_counter() - started_at
        send_desktop_notification(
            "Viewshed Assessment Complete",
            (
                f"Computed Max LOS for {int(round(viewshed_radius))} m radius at "
                f"{viewshed_height_agl:.1f} m AGL using {int(effective_sample_count)} samples in "
                f"{format_elapsed_time(elapsed)}."
            ),
        )
        return {
            "request_id": int(run_clicks),
            "bundle_key": list(bundle["cache_key"]),
            "cache_key": cache_key,
            "radius_m": float(viewshed_radius),
            "observer_height_agl": float(viewshed_height_agl),
            "sample_count": int(effective_sample_count),
            "point": {
                "longitude": float(point_data["longitude"]),
                "latitude": float(point_data["latitude"]),
            },
        }, {"mode": "none", "node_id": None}
    except Exception as exc:
        return {"error": f"Max LOS calculation failed: {exc}"}, {"mode": "none", "node_id": None}


@app.callback(
    Output("node-summary", "children"),
    Input("nodes-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("rssi-calculation-store", "data"),
    Input("rssi-overlay-selection-store", "data"),
)
def update_node_summary(nodes, selected_node_ids, _calculation_store, overlay_selection_store):
    return build_node_summary(
        list(nodes or []),
        [str(value) for value in (selected_node_ids or [])][:2],
        overlay_selection_store or {},
    )


@app.callback(
    Output("rssi-calculation-store", "data"),
    Output("rssi-overlay-selection-store", "data"),
    Output("rssi-progress-complete-store", "data"),
    Output("node-action-message", "children", allow_duplicate=True),
    Output("rssi-worker", "children"),
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Output("map-interaction-loading-message-store", "data", allow_duplicate=True),
    Input("rssi-run-request-store", "data"),
    State("bbox-store", "data"),
    State("bbox-resolution-m", "value"),
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
    bbox_resolution_m,
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
        return (
            no_update,
            no_update,
            {"request_id": request_id, "done": True},
            "Add at least one node before drawing RSSI.",
            f"noop-{request_id}",
            False,
            DEFAULT_MAP_LOADING_MESSAGE,
        )
    if not bbox_data:
        return (
            no_update,
            no_update,
            {"request_id": request_id, "done": True},
            "Load elevation + worldcover data before drawing RSSI.",
            f"noop-{request_id}",
            False,
            DEFAULT_MAP_LOADING_MESSAGE,
        )

    bbox = normalize_bbox(
        bbox_data["min_lon"],
        bbox_data["min_lat"],
        bbox_data["max_lon"],
        bbox_data["max_lat"],
    )
    bundle = get_map_bundle(*bbox, bbox_resolution_m)
    include_ground_loss = "enabled" in (include_rssi_ground_loss or [])
    rx_height = global_rx_height_agl or DEFAULT_GLOBAL_RX_HEIGHT_M
    rx_gain = global_rx_gain_dbi or DEFAULT_RX_GAIN_DBI
    sample_spacing_m = normalize_rssi_path_sample_spacing(bbox_resolution_m)
    node_overlay_keys = {}
    node_signatures = {}
    started_at = time.perf_counter()
    set_rssi_progress_state(request_id, len(nodes), completed_nodes=0, failed=False)
    try:
        ensure_analysis_context(bundle)
        use_parallel_ground_loss_numba = include_ground_loss and numba_parallel_threads_safe()
        progress = tqdm(
            total=len(nodes),
            desc="RSSI overlay",
            unit="node",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            scheduled_nodes = [with_node_defaults(node) for node in nodes]
            overlay_tasks = [
                dask.delayed(compute_single_node_rssi_overlay_result)(
                    node,
                    bundle,
                    include_ground_loss,
                    rx_height,
                    rx_gain,
                    sample_spacing_m=sample_spacing_m,
                    use_parallel_ground_loss_numba=use_parallel_ground_loss_numba,
                )
                for node in scheduled_nodes
            ]
            def on_task_progress(delta, _task_key):
                progress.update(int(delta))
                increment_rssi_progress_state(request_id, delta=int(delta))

            overlay_results = compute_dask_tasks(
                overlay_tasks,
                use_threads=True,
                progress_callback=on_task_progress,
            )
            for node, (cache_key, overlay_result) in zip(scheduled_nodes, overlay_results, strict=False):
                RSSI_OVERLAY_CACHE[cache_key] = overlay_result
                node_id = str(node["id"])
                node_overlay_keys[node_id] = cache_key
                node_signatures[node_id] = node_signature(node)
        finally:
            if hasattr(progress, "close"):
                progress.close()
    except Exception as exc:
        set_rssi_progress_state(request_id, len(nodes), completed_nodes=0, failed=True)
        return (
            no_update,
            no_update,
            {"request_id": request_id, "done": True},
            f"RSSI overlay failed: {exc}",
            f"error-{request_id}",
            False,
            DEFAULT_MAP_LOADING_MESSAGE,
        )
    set_rssi_progress_state(request_id, len(nodes), completed_nodes=len(nodes), failed=False)

    calc_store = {
        "request_id": request_id,
        "bundle_key": list(bundle["cache_key"]),
        "include_ground_loss": include_ground_loss,
        "global_rx_height_agl": float(rx_height),
        "global_rx_gain_dbi": float(rx_gain),
        "path_sample_spacing_m": float(sample_spacing_m),
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
    elapsed = time.perf_counter() - started_at
    sampling_text = f" with {sample_spacing_m:.0f} m path samples" if include_ground_loss else ""
    send_desktop_notification(
        "RSSI Overlay Complete",
        (
            f"Computed RSSI overlay for {len(nodes)} nodes {mode} at "
            f"{float(rx_height):.1f} m AGL / {float(rx_gain):.1f} dBi"
            f"{sampling_text}"
            f" in {format_elapsed_time(elapsed)}."
        ),
    )
    return (
        calc_store,
        overlay_selection,
        {"request_id": request_id, "done": True},
        f"RSSI overlay updated {mode}.",
        f"done-{request_id}",
        no_update,
        no_update,
    )


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
        completed_state = get_rssi_progress_state(progress_meta.get("request_id"))
        if completed_state and completed_state.get("total_nodes"):
            total_nodes = int(completed_state["total_nodes"])
            return "100", f"RSSI overlay complete. {total_nodes}/{total_nodes} nodes finished.", True
        return "100", "RSSI overlay complete.", True

    progress_state = get_rssi_progress_state(progress_meta.get("request_id"))
    if progress_state and int(progress_state.get("total_nodes", 0)) > 0:
        total_nodes = int(progress_state["total_nodes"])
        completed_nodes = min(int(progress_state.get("completed_nodes", 0)), total_nodes)
        progress = int(round((completed_nodes / max(total_nodes, 1)) * 100))
        label = f"Computing RSSI overlay... {completed_nodes}/{total_nodes} nodes complete."
        return str(progress), label, False

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
    Output("map-interaction-loading-store", "data", allow_duplicate=True),
    Output("map-interaction-loading-message-store", "data", allow_duplicate=True),
    Input("draw-rssi-overlay", "n_clicks"),
    State("nodes-store", "data"),
    State("include-rssi-ground-loss", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("terrain-overlay-section", "open"),
    Output("map-bounds-section", "open"),
    Output("rssi-calculations-section", "open"),
    Input("update_graph", "n_clicks"),
    prevent_initial_call=True,
)
def collapse_map_bounds(_n_clicks):
    return True, False, True


@app.callback(
    Output("path-profile-graph", "figure"),
    Output("path-profile-stats", "children"),
    Input("point-path-store", "data"),
    Input("selected-node-ids-store", "data"),
    Input("bbox-resolution-m", "value"),
    State("nodes-store", "data"),
    State("bbox-store", "data"),
    State("global-rx-height-agl", "value"),
    State("global-rx-gain-dbi", "value"),
)
def update_path_profile(
    point_path_data,
    selected_node_ids,
    bbox_resolution_m,
    nodes,
    bbox_data,
    global_rx_height_agl,
    global_rx_gain_dbi,
):
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
        return empty_path_profile_figure("Load elevation + worldcover data to inspect path profiles."), html.Div()
    try:
        bbox = normalize_bbox(
            bbox_data["min_lon"],
            bbox_data["min_lat"],
            bbox_data["max_lon"],
            bbox_data["max_lat"],
        )
        bundle = get_map_bundle(*bbox, bbox_resolution_m)
        ensure_analysis_context(bundle)
    except Exception as exc:
        return empty_path_profile_figure(f"Path profile unavailable: {exc}"), html.Div()

    if point_path_data:
        source = find_node(nodes, point_path_data.get("source_node_id"))
        target_lon = point_path_data.get("target_longitude")
        target_lat = point_path_data.get("target_latitude")
        if source is not None and target_lon is not None and target_lat is not None:
            result = compute_path_loss(
                (source["longitude"], source["latitude"]),
                (float(target_lon), float(target_lat)),
                tx_height=source["height_agl_m"],
                rx_height=float(global_rx_height_agl or DEFAULT_GLOBAL_RX_HEIGHT_M),
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
                    f"Global RX height/gain: {float(global_rx_height_agl or DEFAULT_GLOBAL_RX_HEIGHT_M):.1f} m / "
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

    return empty_path_profile_figure("Select exactly two nodes or use Draw Path To Map Point to inspect the path profile."), html.Div()


if __name__ == "__main__":
    app.run(debug=False, dev_tools_hot_reload=False, use_reloader=False)
