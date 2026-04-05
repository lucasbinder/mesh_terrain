(function () {
  const NODE_SOURCE_ID = "mesh-terrain-nodes";
  const LOADED_BBOX_SOURCE_ID = "mesh-terrain-loaded-bbox";
  const TERRAIN_SOURCE_ID = "mesh-terrain-terrain";
  const TERRAIN_DEM_SOURCE_ID = "mesh-terrain-dem";
  const RSSI_SOURCE_ID = "mesh-terrain-rssi";

  const NODE_LAYER_ID = "mesh-terrain-node-circles";
  const NODE_LABEL_LAYER_ID = "mesh-terrain-node-labels";
  const LOADED_BBOX_LAYER_ID = "mesh-terrain-loaded-bbox-line";
  const TERRAIN_LAYER_ID = "mesh-terrain-terrain-layer";
  const TERRAIN_COLOR_RELIEF_LAYER_ID = "mesh-terrain-color-relief";
  const RSSI_LAYER_ID = "mesh-terrain-rssi-layer";
  const DEFAULT_NODE_HEIGHT_M = 8.0;
  const DEFAULT_TX_GAIN_DBI = 6.0;
  const DEFAULT_TX_POWER_DBM = 30.0;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const COORDINATE_CAPTURE_MODES = new Set(["add-node", "move-node", "point-path", "terrain-bbox"]);

  function emptyFeatureCollection() {
    return {type: "FeatureCollection", features: []};
  }

  function getControllerState() {
    if (!window.__meshTerrainNativeMapState) {
      window.__meshTerrainNativeMapState = {
        activeStyleKey: null,
        appliedCameraRevision: null,
        hoverPushMs: 0,
        lastHoverKey: null,
        map: null,
        pendingSpec: null,
        pendingSpecRevision: 0,
        appliedSpecRevision: -1,
        needsStyleRebuild: false,
        lastTerrainToken: null,
        projectedOverlay: null,
        projectedOverlayFeatures: [],
        runtime: {
          nodes: [],
          counter: 0,
          selectedNodeIds: [],
          clickMode: {mode: "none", node_id: null},
        },
      };
    }
    return window.__meshTerrainNativeMapState;
  }

  function setProp(componentId, prop, value) {
    if (!window.dash_clientside || typeof window.dash_clientside.set_props !== "function") {
      return;
    }
    const payload = {};
    payload[prop] = value;
    window.dash_clientside.set_props(componentId, payload);
  }

  function normalizeSelectedNodeIds(selectedNodeIds, validIds) {
    const validSet = validIds ? new Set((validIds || []).map((value) => String(value))) : null;
    const normalized = [];
    (selectedNodeIds || []).forEach((value) => {
      const nodeId = String(value);
      if (validSet && !validSet.has(nodeId)) {
        return;
      }
      const existingIndex = normalized.indexOf(nodeId);
      if (existingIndex !== -1) {
        normalized.splice(existingIndex, 1);
      }
      normalized.push(nodeId);
    });
    return normalized.slice(-2);
  }

  function selectPrimaryNode(currentSelected, nodeId) {
    const normalized = normalizeSelectedNodeIds(currentSelected);
    const previousPrimary = normalized.length ? normalized[normalized.length - 1] : null;
    const nextId = String(nodeId);
    if (previousPrimary === nextId || previousPrimary === null) {
      return [nextId];
    }
    return [previousPrimary, nextId];
  }

  function parseMaybeNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatContextNumber(value, fallback, digits) {
    const parsed = parseMaybeNumber(value);
    const normalized = parsed === null ? Number(fallback) : parsed;
    return normalized.toFixed(digits);
  }

  function nodeContextSignature(node) {
    const current = node || {};
    return [
      String(current.id),
      formatContextNumber(current.longitude, 0, 6),
      formatContextNumber(current.latitude, 0, 6),
      formatContextNumber(current.height_agl_m, DEFAULT_NODE_HEIGHT_M, 3),
      formatContextNumber(current.antenna_gain_dbi, DEFAULT_TX_GAIN_DBI, 3),
      formatContextNumber(current.tx_power_dbm, DEFAULT_TX_POWER_DBM, 3),
    ].join("|");
  }

  function pointPathContextSignature(pointPathData) {
    if (!pointPathData) {
      return "point:none";
    }

    const sourceNodeId = String(pointPathData.source_node_id || "");
    const targetLongitude = parseMaybeNumber(pointPathData.target_longitude);
    const targetLatitude = parseMaybeNumber(pointPathData.target_latitude);
    if (targetLongitude === null || targetLatitude === null) {
      return `point:${sourceNodeId}:invalid`;
    }
    return `point:${sourceNodeId}:${targetLongitude.toFixed(6)}:${targetLatitude.toFixed(6)}`;
  }

  function overlayContextKey(nodes, selectedNodeIds, pointPathData) {
    const currentNodes = nodes || [];
    const validIds = currentNodes.map((node) => String(node.id));
    const normalizedSelectedNodeIds = normalizeSelectedNodeIds(selectedNodeIds, validIds);
    const parts = ["nodes"];
    currentNodes.forEach((node) => {
      parts.push(nodeContextSignature(node));
    });
    parts.push(`selected:${normalizedSelectedNodeIds.join(",")}`);
    parts.push(pointPathContextSignature(pointPathData));
    return parts.join("||");
  }

  function collectionFeatures(collection) {
    if (!collection || !Array.isArray(collection.features)) {
      return [];
    }
    return collection.features;
  }

  function createSvgElement(tagName) {
    return document.createElementNS(SVG_NS, tagName);
  }

  function ensureProjectedOverlay(state) {
    const root = document.getElementById("native-map-path-overlay");
    if (!root) {
      state.projectedOverlay = null;
      state.projectedOverlayFeatures = [];
      return null;
    }

    if (state.projectedOverlay && state.projectedOverlay.root === root) {
      return state.projectedOverlay;
    }

    root.innerHTML = "";
    const svg = createSvgElement("svg");
    svg.setAttribute("class", "native-map-path-overlay-svg");
    svg.setAttribute("aria-hidden", "true");
    root.appendChild(svg);
    state.projectedOverlay = {root: root, svg: svg};
    state.projectedOverlayFeatures = [];
    return state.projectedOverlay;
  }

  function syncProjectedOverlayViewport(overlay) {
    if (!overlay) {
      return {width: 0, height: 0};
    }

    const width = Math.max(0, Math.round(overlay.root.clientWidth || 0));
    const height = Math.max(0, Math.round(overlay.root.clientHeight || 0));
    overlay.svg.setAttribute("width", String(width));
    overlay.svg.setAttribute("height", String(height));
    overlay.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    return {width: width, height: height};
  }

  function clearProjectedOverlay(overlay) {
    if (!overlay) {
      return;
    }
    overlay.svg.replaceChildren();
  }

  function projectedPoint(map, coordinate) {
    if (!map || !Array.isArray(coordinate) || coordinate.length < 2) {
      return null;
    }
    const point = map.project([Number(coordinate[0]), Number(coordinate[1])]);
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return null;
    }
    return point;
  }

  function drawProjectedPathOverlay(state) {
    const map = state.map;
    const spec = state.pendingSpec;
    const overlay = ensureProjectedOverlay(state);
    if (!overlay) {
      return;
    }
    if (!map || !spec) {
      clearProjectedOverlay(overlay);
      state.projectedOverlayFeatures = [];
      return;
    }

    const viewport = syncProjectedOverlayViewport(overlay);
    clearProjectedOverlay(overlay);
    state.projectedOverlayFeatures = [];
    if (!viewport.width || !viewport.height) {
      return;
    }

    collectionFeatures(spec.path_line).forEach((feature) => {
      const coordinates = (((feature || {}).geometry || {}).coordinates || []);
      if (!Array.isArray(coordinates) || coordinates.length < 2) {
        return;
      }

      const projectedCoordinates = coordinates
        .map((coordinate) => projectedPoint(map, coordinate))
        .filter((point) => point !== null);
      if (projectedCoordinates.length < 2) {
        return;
      }

      const line = createSvgElement("polyline");
      line.setAttribute(
        "points",
        projectedCoordinates.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ")
      );
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", String((((feature || {}).properties || {}).color) || "#38bdf8"));
      line.setAttribute("stroke-width", "4");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("stroke-linejoin", "round");
      line.setAttribute("opacity", "1");
      if ((((feature || {}).properties || {}).dashed)) {
        line.setAttribute("stroke-dasharray", "8 8");
      }
      overlay.svg.appendChild(line);
    });

    collectionFeatures(spec.attenuation_points).forEach((feature) => {
      const point = projectedPoint(map, (((feature || {}).geometry || {}).coordinates || []));
      if (!point) {
        return;
      }

      const properties = (feature || {}).properties || {};
      const radius = Math.max(3, Number(properties.radius || 5));
      const circle = createSvgElement("circle");
      circle.setAttribute("cx", point.x.toFixed(2));
      circle.setAttribute("cy", point.y.toFixed(2));
      circle.setAttribute("r", radius.toFixed(2));
      circle.setAttribute("fill", String(properties.color || "#f97316"));
      circle.setAttribute("fill-opacity", "0.88");
      circle.setAttribute("stroke", "#7c2d12");
      circle.setAttribute("stroke-width", "1");
      overlay.svg.appendChild(circle);
      state.projectedOverlayFeatures.push({feature: feature, point: point, radius: radius + 3});
    });

    collectionFeatures(spec.terrain_block_points).forEach((feature) => {
      const point = projectedPoint(map, (((feature || {}).geometry || {}).coordinates || []));
      if (!point) {
        return;
      }

      const properties = (feature || {}).properties || {};
      const radius = Math.max(4, Number(properties.radius || 6.5));
      const diamond = createSvgElement("rect");
      diamond.setAttribute("x", (point.x - radius).toFixed(2));
      diamond.setAttribute("y", (point.y - radius).toFixed(2));
      diamond.setAttribute("width", (radius * 2).toFixed(2));
      diamond.setAttribute("height", (radius * 2).toFixed(2));
      diamond.setAttribute("fill", String(properties.color || "#a855f7"));
      diamond.setAttribute("fill-opacity", "0.95");
      diamond.setAttribute("transform", `rotate(45 ${point.x.toFixed(2)} ${point.y.toFixed(2)})`);
      overlay.svg.appendChild(diamond);
      state.projectedOverlayFeatures.push({feature: feature, point: point, radius: radius + 3});
    });
  }

  function queryProjectedOverlayFeature(state, point) {
    const projectedOverlayFeatures = state.projectedOverlayFeatures || [];
    if (!projectedOverlayFeatures.length || !point) {
      return null;
    }

    let bestMatch = null;
    let bestDistanceSquared = Infinity;
    projectedOverlayFeatures.forEach((entry) => {
      const dx = Number(point.x) - Number(entry.point.x);
      const dy = Number(point.y) - Number(entry.point.y);
      const distanceSquared = dx * dx + dy * dy;
      const radiusSquared = Number(entry.radius) * Number(entry.radius);
      if (distanceSquared > radiusSquared || distanceSquared >= bestDistanceSquared) {
        return;
      }
      bestMatch = entry.feature;
      bestDistanceSquared = distanceSquared;
    });
    return bestMatch;
  }

  function normalizeFeature(feature) {
    if (!feature || !feature.properties) {
      return null;
    }
    const properties = feature.properties;
    const normalized = {};
    ["kind", "node_id", "name"].forEach((key) => {
      if (properties[key] !== undefined && properties[key] !== null) {
        normalized[key] = String(properties[key]);
      }
    });
    ["blocked_fraction", "added_loss_db", "radius"].forEach((key) => {
      const parsed = parseMaybeNumber(properties[key]);
      if (parsed !== null) {
        normalized[key] = parsed;
      }
    });
    return normalized;
  }

  function viewportBounds(map) {
    const bounds = map.getBounds();
    return {
      min_lon: Number(bounds.getWest().toFixed(6)),
      min_lat: Number(bounds.getSouth().toFixed(6)),
      max_lon: Number(bounds.getEast().toFixed(6)),
      max_lat: Number(bounds.getNorth().toFixed(6)),
    };
  }

  function ensureMap(state, spec, cameraBounds) {
    const container = document.getElementById("native-map");
    if (!container || !window.maplibregl) {
      return null;
    }

    if (state.map) {
      return state.map;
    }

    const fallbackBounds = cameraBounds || {
      min_lon: -113.9,
      min_lat: 39.8667,
      max_lon: -111.15,
      max_lat: 42.65,
    };
    const initialCenter = [
      (Number(fallbackBounds.min_lon) + Number(fallbackBounds.max_lon)) / 2,
      (Number(fallbackBounds.min_lat) + Number(fallbackBounds.max_lat)) / 2,
    ];

    const map = new window.maplibregl.Map({
      container: container,
      style: spec && spec.base_style ? spec.base_style : {version: 8, sources: {}, layers: []},
      center: initialCenter,
      zoom: 8,
      attributionControl: true,
      maxPitch: 85,
      canvasContextAttributes: {antialias: true},
    });

    map.addControl(new window.maplibregl.NavigationControl({showCompass: false}), "top-right");
    bindMapEvents(state, map);
    state.map = map;
    state.activeStyleKey = spec && spec.base_style_key ? String(spec.base_style_key) : null;
    return map;
  }

  function queryNodeFeature(map, point) {
    if (!map.getLayer(NODE_LAYER_ID)) {
      return null;
    }
    const features = map.queryRenderedFeatures(point, {layers: [NODE_LAYER_ID]});
    return features && features.length ? features[0] : null;
  }

  function queryHoverFeature(state, map, point) {
    const projectedOverlayFeature = queryProjectedOverlayFeature(state, point);
    if (projectedOverlayFeature) {
      return projectedOverlayFeature;
    }
    const layers = [NODE_LAYER_ID].filter((layerId) => map.getLayer(layerId));
    if (!layers.length) {
      return null;
    }
    const features = map.queryRenderedFeatures(point, {layers: layers});
    return features && features.length ? features[0] : null;
  }

  function updateCursor(map, runtime, point) {
    const mode = ((runtime || {}).clickMode || {}).mode || "none";
    if (COORDINATE_CAPTURE_MODES.has(mode)) {
      map.getCanvas().style.cursor = "crosshair";
      return;
    }
    map.getCanvas().style.cursor = queryNodeFeature(map, point) ? "pointer" : "";
  }

  function commitInteractionResult(state, result) {
    if (!result) {
      return;
    }

    if (result.nodes !== undefined) {
      state.runtime.nodes = result.nodes;
      setProp("nodes-store", "data", result.nodes);
    }
    if (result.counter !== undefined) {
      state.runtime.counter = result.counter;
      setProp("node-counter-store", "data", result.counter);
    }
    if (result.selectedNodeIds !== undefined) {
      state.runtime.selectedNodeIds = result.selectedNodeIds;
      setProp("selected-node-ids-store", "data", result.selectedNodeIds);
    }
    if (result.clickMode !== undefined) {
      state.runtime.clickMode = result.clickMode;
      setProp("map-click-mode-store", "data", result.clickMode);
    }
    if (result.pointPathData !== undefined) {
      setProp("point-path-store", "data", result.pointPathData);
    }
    if (result.message !== undefined) {
      setProp("node-action-message", "children", result.message);
    }
  }

  function applyMapInteraction(runtime, payload) {
    const mode = ((runtime || {}).clickMode || {}).mode || "none";
    const longitude = Number(payload.longitude);
    const latitude = Number(payload.latitude);
    const nodes = (runtime && runtime.nodes ? runtime.nodes : []).slice();
    const counter = Number((runtime || {}).counter || 0);
    const selectedNodeIds = runtime && runtime.selectedNodeIds ? runtime.selectedNodeIds : [];
    const validIds = nodes.map((node) => String(node.id));

    if (payload.clickedNodeId && mode === "none") {
      if (!validIds.includes(String(payload.clickedNodeId))) {
        return null;
      }
      return {
        selectedNodeIds: selectPrimaryNode(selectedNodeIds, payload.clickedNodeId),
      };
    }

    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) {
      return {
        message: "Native map click did not resolve to a valid coordinate.",
      };
    }

    if (mode === "point-path") {
      return {
        clickMode: {mode: "none", node_id: null},
        pointPathData: {
          source_node_id: String(((runtime || {}).clickMode || {}).node_id || ""),
          target_longitude: longitude,
          target_latitude: latitude,
        },
        message: `Path endpoint selected at (${longitude.toFixed(5)}, ${latitude.toFixed(5)}).`,
      };
    }

    if (mode === "move-node") {
      const nodeId = String((((runtime || {}).clickMode || {}).node_id) || "");
      return {
        nodes: nodes.map((node) => {
          if (String(node.id) !== nodeId) {
            return node;
          }
          return Object.assign({}, node, {
            longitude: longitude,
            latitude: latitude,
          });
        }),
        clickMode: {mode: "none", node_id: null},
        message: `Moved node to (${longitude.toFixed(5)}, ${latitude.toFixed(5)}).`,
      };
    }

    if (mode === "add-node") {
      const promptFn = payload.prompt || window.prompt;
      const rawName = promptFn(`Node name for (${longitude.toFixed(5)}, ${latitude.toFixed(5)})`, "");
      if (rawName === null) {
        return {
          clickMode: {mode: "add-node", node_id: null},
          message: "Click-to-add canceled. Click another map location or disable Click-To-Add.",
        };
      }
      const name = String(rawName).trim();
      if (!name) {
        return {
          clickMode: {mode: "add-node", node_id: null},
          message: "Node name cannot be empty. Click another map location or disable Click-To-Add.",
        };
      }

      const nextCounter = counter + 1;
      const nextNodeId = `node-${nextCounter}`;
      const nextNodes = nodes.slice();
      nextNodes.push({
        id: nextNodeId,
        name: name,
        longitude: longitude,
        latitude: latitude,
        height_agl_m: 8.0,
        antenna_gain_dbi: 6.0,
        tx_power_dbm: 30.0,
      });
      return {
        nodes: nextNodes,
        counter: nextCounter,
        selectedNodeIds: selectPrimaryNode(selectedNodeIds, nextNodeId),
        clickMode: {mode: "add-node", node_id: null},
        message: `Added node ${name} from native map click. Click again to add another node.`,
      };
    }

    return null;
  }

  function removeLayerIfPresent(map, layerId) {
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId);
    }
  }

  function removeSourceIfPresent(map, sourceId) {
    if (map.getSource(sourceId)) {
      map.removeSource(sourceId);
    }
  }

  function firstPresentLayerId(map, layerIds) {
    for (let index = 0; index < layerIds.length; index += 1) {
      if (map.getLayer(layerIds[index])) {
        return layerIds[index];
      }
    }
    return undefined;
  }

  function syncImageLayer(map, sourceId, layerId, layerSpec) {
    if (!layerSpec || !layerSpec.image || !Array.isArray(layerSpec.coordinates)) {
      removeLayerIfPresent(map, layerId);
      removeSourceIfPresent(map, sourceId);
      return;
    }

    const existingSource = map.getSource(sourceId);
    if (!existingSource) {
      map.addSource(sourceId, {
        type: "image",
        url: layerSpec.image,
        coordinates: layerSpec.coordinates,
      });
    } else if (typeof existingSource.updateImage === "function") {
      existingSource.updateImage({
        url: layerSpec.image,
        coordinates: layerSpec.coordinates,
      });
    } else {
      removeLayerIfPresent(map, layerId);
      removeSourceIfPresent(map, sourceId);
      map.addSource(sourceId, {
        type: "image",
        url: layerSpec.image,
        coordinates: layerSpec.coordinates,
      });
    }

    ensureLayer(map, {
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: {
        "raster-opacity": Number(layerSpec.opacity || 1.0),
        "raster-fade-duration": 0,
      },
    });
  }

  function setGeoJsonSourceData(map, sourceId, data) {
    const nextData = data || emptyFeatureCollection();
    const existingSource = map.getSource(sourceId);
    if (existingSource && typeof existingSource.setData === "function") {
      existingSource.setData(nextData);
      return;
    }
    if (existingSource) {
      removeSourceIfPresent(map, sourceId);
    }
    map.addSource(sourceId, {
      type: "geojson",
      data: nextData,
    });
  }

  function ensureLayer(map, layerSpec, beforeId) {
    if (!map.getLayer(layerSpec.id)) {
      map.addLayer(layerSpec, beforeId);
      return;
    }
    if (beforeId && layerSpec.id !== beforeId) {
      try {
        map.moveLayer(layerSpec.id, beforeId);
      } catch (_error) {
      }
    }
    if (layerSpec.filter) {
      map.setFilter(layerSpec.id, layerSpec.filter);
    }
    Object.entries(layerSpec.layout || {}).forEach(([key, value]) => {
      map.setLayoutProperty(layerSpec.id, key, value);
    });
    Object.entries(layerSpec.paint || {}).forEach(([key, value]) => {
      map.setPaintProperty(layerSpec.id, key, value);
    });
  }

  function applyTerrainState(state, spec) {
    const map = state.map;
    if (!map) {
      return;
    }

    if (!spec.terrain_dem) {
      try {
        map.setTerrain(null);
      } catch (_error) {
      }
      if (map.getLayer(TERRAIN_COLOR_RELIEF_LAYER_ID)) {
        map.removeLayer(TERRAIN_COLOR_RELIEF_LAYER_ID);
      }
      if (map.getSource(TERRAIN_DEM_SOURCE_ID)) {
        map.removeSource(TERRAIN_DEM_SOURCE_ID);
      }
      state.lastTerrainToken = null;
      return;
    }

    const terrainDem = spec.terrain_dem;
    const terrainChanged = state.lastTerrainToken !== terrainDem.token || !map.getSource(TERRAIN_DEM_SOURCE_ID);

    if (terrainChanged) {
      try {
        map.setTerrain(null);
      } catch (_error) {
      }
      if (map.getLayer(TERRAIN_COLOR_RELIEF_LAYER_ID)) {
        map.removeLayer(TERRAIN_COLOR_RELIEF_LAYER_ID);
      }
      if (map.getSource(TERRAIN_DEM_SOURCE_ID)) {
        map.removeSource(TERRAIN_DEM_SOURCE_ID);
      }

      map.addSource(TERRAIN_DEM_SOURCE_ID, {
        type: "raster-dem",
        tiles: terrainDem.tiles,
        bounds: terrainDem.bounds,
        tileSize: Number(terrainDem.tile_size || 256),
        maxzoom: Number(terrainDem.maxzoom || 15),
        encoding: terrainDem.encoding || "terrarium",
      });

      map.setTerrain({
        source: TERRAIN_DEM_SOURCE_ID,
        exaggeration: Number(terrainDem.exaggeration || 1.0),
      });
    }

    if (terrainDem.color_relief && Number(terrainDem.color_relief.opacity || 0) > 0) {
      const colorReliefBeforeId = firstPresentLayerId(map, [
        LOADED_BBOX_LAYER_ID,
        NODE_LAYER_ID,
        NODE_LABEL_LAYER_ID,
      ]);
      if (map.getLayer(TERRAIN_COLOR_RELIEF_LAYER_ID)) {
        map.removeLayer(TERRAIN_COLOR_RELIEF_LAYER_ID);
      }
      map.addLayer({
        id: TERRAIN_COLOR_RELIEF_LAYER_ID,
        type: "color-relief",
        source: TERRAIN_DEM_SOURCE_ID,
        paint: {
          "color-relief-color": terrainDem.color_relief.expression,
          "color-relief-opacity": Number(terrainDem.color_relief.opacity || 0),
        },
      }, colorReliefBeforeId);
    } else if (map.getLayer(TERRAIN_COLOR_RELIEF_LAYER_ID)) {
      map.removeLayer(TERRAIN_COLOR_RELIEF_LAYER_ID);
    }

    if (terrainChanged && map.getPitch() < 35) {
      map.easeTo({
        pitch: Number(terrainDem.pitch || 60),
        bearing: Number(terrainDem.bearing || 20),
        duration: 0,
      });
    }
    state.lastTerrainToken = terrainDem.token;
  }

  function applyPendingSpec(state) {
    if (!state.map || !state.pendingSpec || !state.map.isStyleLoaded()) {
      return;
    }
    if (!state.needsStyleRebuild && state.appliedSpecRevision === state.pendingSpecRevision) {
      return;
    }

    const map = state.map;
    const spec = state.pendingSpec;
    applyTerrainState(state, spec);

    syncImageLayer(map, TERRAIN_SOURCE_ID, TERRAIN_LAYER_ID, spec.terrain_layer);
    syncImageLayer(map, RSSI_SOURCE_ID, RSSI_LAYER_ID, spec.rssi_layer);

    setGeoJsonSourceData(map, LOADED_BBOX_SOURCE_ID, spec.loaded_bbox);
    ensureLayer(map, {
      id: LOADED_BBOX_LAYER_ID,
      type: "line",
      source: LOADED_BBOX_SOURCE_ID,
      paint: {
        "line-color": "#38bdf8",
        "line-width": 2,
        "line-dasharray": [2, 2],
      },
    });

    setGeoJsonSourceData(map, NODE_SOURCE_ID, spec.nodes);
    ensureLayer(map, {
      id: NODE_LAYER_ID,
      type: "circle",
      source: NODE_SOURCE_ID,
      layout: {
        "visibility": "visible",
      },
      paint: {
        "circle-radius": 8,
        "circle-color": ["coalesce", ["get", "color"], "#38bdf8"],
        "circle-stroke-color": ["coalesce", ["get", "outline_color"], "#ffffff"],
        "circle-stroke-width": ["coalesce", ["to-number", ["get", "outline_width"]], 1.5],
        "circle-opacity": 1.0,
        "circle-stroke-opacity": 1.0,
      },
    });
    ensureLayer(map, {
      id: NODE_LABEL_LAYER_ID,
      type: "symbol",
      source: NODE_SOURCE_ID,
      layout: {
        "text-field": ["get", "name"],
        "text-size": 12,
        "text-anchor": "top",
        "text-offset": [0, 1.1],
        "text-font": ["Open Sans Regular", "Noto Sans Regular"],
        "visibility": "visible",
      },
      paint: {
        "text-color": "#f9fafb",
        "text-halo-color": "rgba(3, 7, 18, 0.88)",
        "text-halo-width": 1.2,
      },
    });

    drawProjectedPathOverlay(state);
    map.triggerRepaint();
    state.appliedSpecRevision = state.pendingSpecRevision;
    state.needsStyleRebuild = false;
  }

  function syncStyleAndSpec(state, spec) {
    const map = state.map;
    if (!map || !spec) {
      return;
    }

    const nextStyleKey = spec.base_style_key ? String(spec.base_style_key) : "satellite";
    state.pendingSpec = spec;
    state.pendingSpecRevision += 1;
    if (state.activeStyleKey !== nextStyleKey) {
      state.activeStyleKey = nextStyleKey;
      state.needsStyleRebuild = true;
      map.setStyle(spec.base_style);
      return;
    }

    applyPendingSpec(state);
  }

  function maybeFitCamera(state, cameraBounds, cameraRevision) {
    const map = state.map;
    const revision = Number(cameraRevision || 0);
    if (!map || !cameraBounds || revision === state.appliedCameraRevision) {
      return;
    }
    state.appliedCameraRevision = revision;
    map.fitBounds(
      [
        [Number(cameraBounds.min_lon), Number(cameraBounds.min_lat)],
        [Number(cameraBounds.max_lon), Number(cameraBounds.max_lat)],
      ],
      {
        padding: 40,
        duration: 0,
        essential: true,
      }
    );
    if (state.pendingSpec && state.pendingSpec.terrain_dem && map.getPitch() < 35) {
      map.easeTo({
        pitch: Number(state.pendingSpec.terrain_dem.pitch || 60),
        bearing: Number(state.pendingSpec.terrain_dem.bearing || 20),
        duration: 0,
      });
    }
  }

  function bindMapEvents(state, map) {
    map.on("load", function () {
      applyPendingSpec(state);
      maybeFitCamera(state, state.lastCameraBounds, state.lastCameraRevision);
    });

    map.on("styledata", function () {
      if (map.isStyleLoaded()) {
        applyPendingSpec(state);
      }
    });

    map.on("render", function () {
      drawProjectedPathOverlay(state);
    });

    map.on("moveend", function () {
      const bbox = viewportBounds(map);
      setProp("map-view-store", "data", bbox);
      if ((((state.runtime || {}).clickMode || {}).mode || "none") === "terrain-bbox") {
        setProp("min_lon", "value", bbox.min_lon);
        setProp("min_lat", "value", bbox.min_lat);
        setProp("max_lon", "value", bbox.max_lon);
        setProp("max_lat", "value", bbox.max_lat);
        setProp("map-click-mode-store", "data", {mode: "none", node_id: null});
        state.runtime.clickMode = {mode: "none", node_id: null};
        setProp("node-action-message", "children", "Terrain load bounds copied from the current native map viewport.");
      }
    });

    map.on("mousemove", function (event) {
      updateCursor(map, state.runtime, event.point);
      const feature = normalizeFeature(queryHoverFeature(state, map, event.point));
      const hoverKey = `${event.lngLat.lng.toFixed(4)}:${event.lngLat.lat.toFixed(4)}:${feature ? feature.kind || "" : ""}`;
      const now = Date.now();
      if (hoverKey === state.lastHoverKey && now - state.hoverPushMs < 180) {
        return;
      }
      if (now - state.hoverPushMs < 180) {
        return;
      }
      state.hoverPushMs = now;
      state.lastHoverKey = hoverKey;
      setProp("native-map-hover-store", "data", {
        longitude: Number(event.lngLat.lng),
        latitude: Number(event.lngLat.lat),
        feature: feature,
      });
    });

    map.on("mouseout", function () {
      map.getCanvas().style.cursor = "";
      state.lastHoverKey = null;
      setProp("native-map-hover-store", "data", null);
    });

    map.on("click", function (event) {
      const nodeFeature = queryNodeFeature(map, event.point);
      const result = applyMapInteraction(state.runtime, {
        clickedNodeId: nodeFeature && nodeFeature.properties ? String(nodeFeature.properties.node_id) : null,
        longitude: event.lngLat.lng,
        latitude: event.lngLat.lat,
        prompt: window.prompt ? window.prompt.bind(window) : null,
      });
      commitInteractionResult(state, result);
    });
  }

  function renderNativeMap(spec, cameraBounds, cameraRevision, nodes, counter, selectedNodeIds, pointPathData, clickMode) {
    const state = getControllerState();
    state.runtime = {
      nodes: (nodes || []).slice(),
      counter: Number(counter || 0),
      selectedNodeIds: (selectedNodeIds || []).slice(),
      clickMode: clickMode || {mode: "none", node_id: null},
    };
    state.lastCameraBounds = cameraBounds || null;
    state.lastCameraRevision = Number(cameraRevision || 0);

    if (!window.maplibregl) {
      return `maplibre-missing:${Date.now()}`;
    }

    const map = ensureMap(state, spec || {}, cameraBounds || null);
    if (!map) {
      return `map-container-missing:${Date.now()}`;
    }

    const currentOverlayContextKey = overlayContextKey(nodes, selectedNodeIds, pointPathData);
    if (spec && spec.overlay_context_key && spec.overlay_context_key !== currentOverlayContextKey) {
      maybeFitCamera(state, cameraBounds, cameraRevision);
      updateCursor(map, state.runtime, map.project(map.getCenter()));
      return `native-map-stale-spec:${Date.now()}`;
    }

    syncStyleAndSpec(state, spec || {
      base_style_key: "satellite",
      base_style: {version: 8, sources: {}, layers: []},
    });
    drawProjectedPathOverlay(state);
    maybeFitCamera(state, cameraBounds, cameraRevision);
    updateCursor(map, state.runtime, map.project(map.getCenter()));
    return `native-map-rendered:${Date.now()}`;
  }

  function startRssiOverlay(nClicks, nodes, includeGroundLoss) {
    const noUpdate = window.dash_clientside.no_update;
    if (!nClicks) {
      return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate];
    }
    const nodeCount = (nodes || []).length;
    if (!nodeCount) {
      return [noUpdate, true, noUpdate, noUpdate, false, "Updating map..."];
    }
    const expensive = (includeGroundLoss || []).includes("enabled");
    const eta = expensive ? Math.max(10, nodeCount * 18) : Math.max(4, nodeCount * 3);
    const requestId = Number(nClicks);
    return [
      {
        request_id: requestId,
        start_ms: Date.now(),
        eta_sec: eta,
      },
      false,
      null,
      {
        request_id: requestId,
      },
      true,
      "Performing RSSI and LOS Calculations",
    ];
  }

  window.__meshTerrainNativeMapTest = {
    applyMapInteraction: applyMapInteraction,
    normalizeSelectedNodeIds: normalizeSelectedNodeIds,
    selectPrimaryNode: selectPrimaryNode,
  };

  window.dash_clientside = Object.assign({}, window.dash_clientside);
  window.dash_clientside.clientside = Object.assign({}, window.dash_clientside.clientside, {
    renderNativeMap: renderNativeMap,
    startRssiOverlay: startRssiOverlay,
  });
})();
