(function () {
  const NODE_SOURCE_ID = "mesh-terrain-nodes";
  const LOADED_BBOX_SOURCE_ID = "mesh-terrain-loaded-bbox";
  const TERRAIN_SOURCE_ID = "mesh-terrain-terrain";
  const TERRAIN_DEM_SOURCE_ID = "mesh-terrain-dem";
  const WORLDCOVER_SOURCE_ID = "mesh-terrain-worldcover";
  const VIEWSHED_SOURCE_ID = "mesh-terrain-viewshed";
  const RSSI_SOURCE_ID = "mesh-terrain-rssi";
  const VIEWSHED_POINT_SOURCE_ID = "mesh-terrain-viewshed-point";
  const VIEWSHED_RADIUS_SOURCE_ID = "mesh-terrain-viewshed-radius";
  const VIEWSHED_SAMPLES_SOURCE_ID = "mesh-terrain-viewshed-samples";
  const BASEMAP_SOURCE_ID = "basemap";

  const NODE_LAYER_ID = "mesh-terrain-node-circles";
  const NODE_LABEL_LAYER_ID = "mesh-terrain-node-labels";
  const LOADED_BBOX_LAYER_ID = "mesh-terrain-loaded-bbox-line";
  const TERRAIN_LAYER_ID = "mesh-terrain-terrain-layer";
  const TERRAIN_COLOR_RELIEF_LAYER_ID = "mesh-terrain-color-relief";
  const WORLDCOVER_LAYER_ID = "mesh-terrain-worldcover-layer";
  const VIEWSHED_LAYER_ID = "mesh-terrain-viewshed-layer";
  const RSSI_LAYER_ID = "mesh-terrain-rssi-layer";
  const VIEWSHED_POINT_LAYER_ID = "mesh-terrain-viewshed-point-outer";
  const VIEWSHED_POINT_INNER_LAYER_ID = "mesh-terrain-viewshed-point-inner";
  const VIEWSHED_RADIUS_LAYER_ID = "mesh-terrain-viewshed-radius-line";
  const VIEWSHED_SAMPLES_LAYER_ID = "mesh-terrain-viewshed-samples-circle";
  const BASEMAP_LAYER_ID = "basemap";
  const DEFAULT_NODE_HEIGHT_M = 2.0;
  const DEFAULT_TX_GAIN_DBI = 5.0;
  const DEFAULT_TX_POWER_DBM = 22.0;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const COORDINATE_CAPTURE_MODES = new Set(["add-node", "move-node", "point-path", "terrain-bbox", "viewshed-point"]);

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
        imageLayerState: {},
        projectedOverlay: null,
        projectedOverlayFeatures: [],
        bboxSelection: {
          active: false,
          startPoint: null,
          currentPoint: null,
          moveHandler: null,
          upHandler: null,
        },
        suppressClickUntilMs: 0,
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

  function buildAddNodeInteractionResult(runtime, longitude, latitude, rawName, sourceLabel) {
    const nodes = (runtime && runtime.nodes ? runtime.nodes : []).slice();
    const counter = Number((runtime || {}).counter || 0);
    const selectedNodeIds = runtime && runtime.selectedNodeIds ? runtime.selectedNodeIds : [];
    const validIds = nodes.map((node) => String(node.id));
    const name = String(rawName || "").trim();
    if (!name) {
      return {
        clickMode: {mode: "add-node", node_id: null},
        message: sourceLabel === "manual entry"
          ? "Node name cannot be empty."
          : "Node name cannot be empty. Click another map location or disable Click-To-Add.",
      };
    }

    const nextCounter = counter + 1;
    const nextNodeId = `node-${nextCounter}`;
    const nextNodes = nodes.slice();
    const normalizedSelectedNodeIds = normalizeSelectedNodeIds(selectedNodeIds, validIds);
    nextNodes.push({
      id: nextNodeId,
      name: name,
      longitude: Number(longitude),
      latitude: Number(latitude),
      height_agl_m: DEFAULT_NODE_HEIGHT_M,
      antenna_gain_dbi: DEFAULT_TX_GAIN_DBI,
      tx_power_dbm: DEFAULT_TX_POWER_DBM,
    });
    return {
      nodes: nextNodes,
      counter: nextCounter,
      selectedNodeIds: selectPrimaryNode(normalizedSelectedNodeIds, nextNodeId),
      clickMode: {mode: "none", node_id: null},
      message: sourceLabel === "manual entry"
        ? `Added node ${name}.`
        : `Added node ${name} from native map click.`,
    };
  }

  function parseMaybeNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function parseCoordinateInputValue(value) {
    if (value === null || value === undefined) {
      return null;
    }
    if (typeof value === "string" && value.trim() === "") {
      return null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function readInputElementValue(componentId, fallbackValue) {
    const element = document.getElementById(String(componentId));
    if (element && "value" in element) {
      return element.value;
    }
    return fallbackValue;
  }

  function clearInputElementValue(componentId) {
    const element = document.getElementById(String(componentId));
    if (element && "value" in element) {
      element.value = "";
    }
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

  function viewshedPointContextSignature(pointData) {
    if (!pointData) {
      return "viewshed:none";
    }
    const longitude = parseMaybeNumber(pointData.longitude);
    const latitude = parseMaybeNumber(pointData.latitude);
    if (longitude === null || latitude === null) {
      return "viewshed:invalid";
    }
    return `viewshed:${longitude.toFixed(6)}:${latitude.toFixed(6)}`;
  }

  function viewshedAssessmentContextSignature(assessmentStore) {
    if (!assessmentStore) {
      return "viewshed-assessment:none";
    }
    const cacheKey = String(assessmentStore.cache_key || "");
    if (!cacheKey) {
      return "viewshed-assessment:pending";
    }
    return `viewshed-assessment:${cacheKey}`;
  }

  function overlayContextKey(nodes, selectedNodeIds, pointPathData, viewshedPointData, viewshedAssessmentStore) {
    const currentNodes = nodes || [];
    const validIds = currentNodes.map((node) => String(node.id));
    const normalizedSelectedNodeIds = normalizeSelectedNodeIds(selectedNodeIds, validIds);
    const parts = ["nodes"];
    currentNodes.forEach((node) => {
      parts.push(nodeContextSignature(node));
    });
    parts.push(`selected:${normalizedSelectedNodeIds.join(",")}`);
    parts.push(pointPathContextSignature(pointPathData));
    parts.push(viewshedPointContextSignature(viewshedPointData));
    parts.push(viewshedAssessmentContextSignature(viewshedAssessmentStore));
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

  function terrainBboxSelectionElement() {
    return document.getElementById("native-map-bbox-selection");
  }

  function setTerrainBboxSelectionBox(startPoint, endPoint) {
    const box = terrainBboxSelectionElement();
    if (!box || !startPoint || !endPoint) {
      return;
    }
    const left = Math.min(Number(startPoint.x), Number(endPoint.x));
    const top = Math.min(Number(startPoint.y), Number(endPoint.y));
    const width = Math.abs(Number(endPoint.x) - Number(startPoint.x));
    const height = Math.abs(Number(endPoint.y) - Number(startPoint.y));
    box.style.display = "block";
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
  }

  function hideTerrainBboxSelectionBox() {
    const box = terrainBboxSelectionElement();
    if (!box) {
      return;
    }
    box.style.display = "none";
    box.style.left = "0";
    box.style.top = "0";
    box.style.width = "0";
    box.style.height = "0";
  }

  function terrainBboxPointFromMouseEvent(map, mouseEvent) {
    if (!map || !mouseEvent) {
      return null;
    }
    const rect = map.getContainer().getBoundingClientRect();
    const x = Math.min(Math.max(Number(mouseEvent.clientX) - rect.left, 0), rect.width);
    const y = Math.min(Math.max(Number(mouseEvent.clientY) - rect.top, 0), rect.height);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return null;
    }
    return {x: x, y: y};
  }

  function clearTerrainBboxSelection(state, map) {
    const selection = state && state.bboxSelection ? state.bboxSelection : null;
    if (selection && selection.moveHandler) {
      window.removeEventListener("mousemove", selection.moveHandler);
    }
    if (selection && selection.upHandler) {
      window.removeEventListener("mouseup", selection.upHandler);
    }
    if (
      map &&
      map.dragPan &&
      typeof map.dragPan.enable === "function" &&
      typeof map.dragPan.isEnabled === "function" &&
      !map.dragPan.isEnabled()
    ) {
      map.dragPan.enable();
    }
    if (state) {
      state.bboxSelection = {
        active: false,
        startPoint: null,
        currentPoint: null,
        moveHandler: null,
        upHandler: null,
      };
    }
    hideTerrainBboxSelectionBox();
  }

  function finishTerrainBboxSelection(state, map, mouseEvent) {
    if (!state || !state.bboxSelection || !state.bboxSelection.active) {
      return;
    }
    const startPoint = state.bboxSelection.startPoint;
    const endPoint = terrainBboxPointFromMouseEvent(map, mouseEvent) || state.bboxSelection.currentPoint || startPoint;
    const dx = Math.abs(Number(endPoint.x) - Number(startPoint.x));
    const dy = Math.abs(Number(endPoint.y) - Number(startPoint.y));

    clearTerrainBboxSelection(state, map);
    state.suppressClickUntilMs = Date.now() + 250;

    if (dx < 4 || dy < 4) {
      setProp("node-action-message", "children", "Drag a box on the map to set the terrain bounding box.");
      return;
    }

    const left = Math.min(Number(startPoint.x), Number(endPoint.x));
    const right = Math.max(Number(startPoint.x), Number(endPoint.x));
    const top = Math.min(Number(startPoint.y), Number(endPoint.y));
    const bottom = Math.max(Number(startPoint.y), Number(endPoint.y));
    const northWest = map.unproject([left, top]);
    const southEast = map.unproject([right, bottom]);
    const minLon = Math.min(Number(northWest.lng), Number(southEast.lng));
    const maxLon = Math.max(Number(northWest.lng), Number(southEast.lng));
    const minLat = Math.min(Number(northWest.lat), Number(southEast.lat));
    const maxLat = Math.max(Number(northWest.lat), Number(southEast.lat));

    setProp("min_lon", "value", Number(minLon.toFixed(6)));
    setProp("min_lat", "value", Number(minLat.toFixed(6)));
    setProp("max_lon", "value", Number(maxLon.toFixed(6)));
    setProp("max_lat", "value", Number(maxLat.toFixed(6)));
    setProp("map-click-mode-store", "data", {mode: "none", node_id: null});
    setProp("node-action-message", "children", "Terrain bounding box selected from drag box.");
    state.runtime.clickMode = {mode: "none", node_id: null};
  }

  function beginTerrainBboxSelection(state, map, mapEvent) {
    const originalEvent = mapEvent && mapEvent.originalEvent ? mapEvent.originalEvent : null;
    if (originalEvent && Number(originalEvent.button) !== 0) {
      return;
    }
    clearTerrainBboxSelection(state, map);

    const startPoint = mapEvent && mapEvent.point ? {x: Number(mapEvent.point.x), y: Number(mapEvent.point.y)} : null;
    if (!startPoint) {
      return;
    }

    if (originalEvent && typeof originalEvent.preventDefault === "function") {
      originalEvent.preventDefault();
    }
    if (originalEvent && typeof originalEvent.stopPropagation === "function") {
      originalEvent.stopPropagation();
    }

    if (
      map.dragPan &&
      typeof map.dragPan.disable === "function" &&
      typeof map.dragPan.isEnabled === "function" &&
      map.dragPan.isEnabled()
    ) {
      map.dragPan.disable();
    }

    const moveHandler = function (nextEvent) {
      const point = terrainBboxPointFromMouseEvent(map, nextEvent);
      if (!point) {
        return;
      }
      state.bboxSelection.currentPoint = point;
      setTerrainBboxSelectionBox(startPoint, point);
    };
    const upHandler = function (nextEvent) {
      finishTerrainBboxSelection(state, map, nextEvent);
    };

    state.bboxSelection = {
      active: true,
      startPoint: startPoint,
      currentPoint: startPoint,
      moveHandler: moveHandler,
      upHandler: upHandler,
    };
    setTerrainBboxSelectionBox(startPoint, startPoint);
    window.addEventListener("mousemove", moveHandler);
    window.addEventListener("mouseup", upHandler, {once: false});
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

  function interpolateLineCoordinate(start, end, fraction) {
    return [
      Number(start[0]) + (Number(end[0]) - Number(start[0])) * fraction,
      Number(start[1]) + (Number(end[1]) - Number(start[1])) * fraction,
    ];
  }

  function densifyLineCoordinates(coordinates) {
    if (!Array.isArray(coordinates) || coordinates.length < 2) {
      return [];
    }
    const segmentSampleCount = coordinates.length === 2 ? 96 : 16;
    const densified = [];
    for (let index = 0; index < coordinates.length - 1; index += 1) {
      const start = coordinates[index];
      const end = coordinates[index + 1];
      if (!Array.isArray(start) || start.length < 2 || !Array.isArray(end) || end.length < 2) {
        continue;
      }
      if (densified.length === 0) {
        densified.push([Number(start[0]), Number(start[1])]);
      }
      for (let sampleIndex = 1; sampleIndex < segmentSampleCount; sampleIndex += 1) {
        densified.push(interpolateLineCoordinate(start, end, sampleIndex / segmentSampleCount));
      }
      densified.push([Number(end[0]), Number(end[1])]);
    }
    return densified;
  }

  function pointDistance(pointA, pointB) {
    const dx = Number(pointA.x) - Number(pointB.x);
    const dy = Number(pointA.y) - Number(pointB.y);
    return Math.sqrt(dx * dx + dy * dy);
  }

  function pointInsideExpandedViewport(point, viewport, margin) {
    return (
      Number(point.x) >= -margin &&
      Number(point.x) <= Number(viewport.width) + margin &&
      Number(point.y) >= -margin &&
      Number(point.y) <= Number(viewport.height) + margin
    );
  }

  function clippedRayEndpoint(origin, direction, viewport) {
    const dx = Number(direction.x);
    const dy = Number(direction.y);
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6)) {
      return null;
    }

    const margin = Math.max(80, Math.max(Number(viewport.width), Number(viewport.height)) * 0.2);
    const bounds = {
      minX: -margin,
      maxX: Number(viewport.width) + margin,
      minY: -margin,
      maxY: Number(viewport.height) + margin,
    };
    const candidates = [];

    if (Math.abs(dx) >= 1e-6) {
      candidates.push((bounds.minX - Number(origin.x)) / dx);
      candidates.push((bounds.maxX - Number(origin.x)) / dx);
    }
    if (Math.abs(dy) >= 1e-6) {
      candidates.push((bounds.minY - Number(origin.y)) / dy);
      candidates.push((bounds.maxY - Number(origin.y)) / dy);
    }

    let best = null;
    candidates.forEach((t) => {
      if (!Number.isFinite(t) || t <= 0) {
        return;
      }
      const x = Number(origin.x) + dx * t;
      const y = Number(origin.y) + dy * t;
      if (x < bounds.minX - 1 || x > bounds.maxX + 1 || y < bounds.minY - 1 || y > bounds.maxY + 1) {
        return;
      }
      if (!best || t < best.t) {
        best = {x: x, y: y, t: t};
      }
    });
    return best ? {x: best.x, y: best.y} : null;
  }

  function traceProjectedLineFromEndpoint(map, densifiedCoordinates, fromStart, viewport) {
    if (densifiedCoordinates.length < 2) {
      return null;
    }

    const jumpThreshold = Math.max(140, Math.max(Number(viewport.width), Number(viewport.height)) * 0.45);
    const anchorIndex = fromStart ? 0 : densifiedCoordinates.length - 1;
    const anchorPoint = projectedPoint(map, densifiedCoordinates[anchorIndex]);
    if (!anchorPoint) {
      return null;
    }

    let previousPoint = anchorPoint;
    let directionPoint = null;
    let lastStablePoint = anchorPoint;
    let stableCount = 1;
    let reachedOppositeEndpoint = true;

    for (
      let index = anchorIndex + (fromStart ? 1 : -1);
      index >= 0 && index < densifiedCoordinates.length;
      index += fromStart ? 1 : -1
    ) {
      const point = projectedPoint(map, densifiedCoordinates[index]);
      if (!point || pointDistance(point, previousPoint) > jumpThreshold) {
        reachedOppositeEndpoint = false;
        break;
      }
      if (!directionPoint && pointDistance(anchorPoint, point) >= 0.5) {
        directionPoint = point;
      }
      lastStablePoint = point;
      previousPoint = point;
      stableCount += 1;
    }

    return {
      anchorPoint: anchorPoint,
      directionPoint: directionPoint || (stableCount >= 2 ? lastStablePoint : null),
      lastStablePoint: lastStablePoint,
      stableCount: stableCount,
      reachedOppositeEndpoint: reachedOppositeEndpoint,
      anchorVisible: pointInsideExpandedViewport(anchorPoint, viewport, 32),
    };
  }

  function candidateLineSegmentFromTrace(trace, viewport) {
    if (!trace || !trace.anchorPoint) {
      return null;
    }
    if (trace.reachedOppositeEndpoint && trace.stableCount >= 2) {
      return {
        points: [trace.anchorPoint, trace.lastStablePoint],
        anchorVisible: trace.anchorVisible,
        stableCount: trace.stableCount,
        fullSpan: true,
      };
    }
    if (!trace.directionPoint || pointDistance(trace.anchorPoint, trace.directionPoint) < 0.5) {
      return null;
    }
    const clippedEnd = clippedRayEndpoint(
      trace.anchorPoint,
      {
        x: Number(trace.directionPoint.x) - Number(trace.anchorPoint.x),
        y: Number(trace.directionPoint.y) - Number(trace.anchorPoint.y),
      },
      viewport
    );
    if (!clippedEnd) {
      return null;
    }
    return {
      points: [trace.anchorPoint, clippedEnd],
      anchorVisible: trace.anchorVisible,
      stableCount: trace.stableCount,
      fullSpan: false,
    };
  }

  function stableProjectedLineSegment(map, coordinates, viewport) {
    const densifiedCoordinates = densifyLineCoordinates(coordinates);
    if (densifiedCoordinates.length < 2) {
      return null;
    }

    const startTrace = traceProjectedLineFromEndpoint(map, densifiedCoordinates, true, viewport);
    const endTrace = traceProjectedLineFromEndpoint(map, densifiedCoordinates, false, viewport);
    const startCandidate = candidateLineSegmentFromTrace(startTrace, viewport);
    const endCandidate = candidateLineSegmentFromTrace(endTrace, viewport);

    if (startCandidate && startCandidate.fullSpan) {
      return startCandidate.points;
    }
    if (endCandidate && endCandidate.fullSpan) {
      return [endCandidate.points[1], endCandidate.points[0]];
    }
    if (startCandidate && startCandidate.anchorVisible && (!endCandidate || !endCandidate.anchorVisible)) {
      return startCandidate.points;
    }
    if (endCandidate && endCandidate.anchorVisible && (!startCandidate || !startCandidate.anchorVisible)) {
      return endCandidate.points;
    }
    if (startCandidate && endCandidate) {
      if (startCandidate.anchorVisible !== endCandidate.anchorVisible) {
        return startCandidate.anchorVisible ? startCandidate.points : endCandidate.points;
      }
      if (startCandidate.stableCount !== endCandidate.stableCount) {
        return startCandidate.stableCount > endCandidate.stableCount ? startCandidate.points : endCandidate.points;
      }
      return startCandidate.points;
    }
    if (startCandidate) {
      return startCandidate.points;
    }
    if (endCandidate) {
      return endCandidate.points;
    }
    return null;
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

      const projectedSegment = stableProjectedLineSegment(map, coordinates, viewport);
      if (!projectedSegment) {
        return;
      }

      const line = createSvgElement("line");
      line.setAttribute("x1", projectedSegment[0].x.toFixed(2));
      line.setAttribute("y1", projectedSegment[0].y.toFixed(2));
      line.setAttribute("x2", projectedSegment[1].x.toFixed(2));
      line.setAttribute("y2", projectedSegment[1].y.toFixed(2));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", String((((feature || {}).properties || {}).color) || "#38bdf8"));
      line.setAttribute("stroke-width", "4");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("opacity", "1");
      if ((((feature || {}).properties || {}).dashed)) {
        line.setAttribute("stroke-dasharray", "8 8");
      }
      overlay.svg.appendChild(line);
    });

    collectionFeatures(spec.attenuation_points).forEach((feature) => {
      const properties = (feature || {}).properties || {};
      const point = projectedPoint(map, (((feature || {}).geometry || {}).coordinates || []));
      if (!point) {
        return;
      }

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
      const properties = (feature || {}).properties || {};
      const point = projectedPoint(map, (((feature || {}).geometry || {}).coordinates || []));
      if (!point) {
        return;
      }

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
      maxPitch: 60,
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

  function queryViewshedPointFeature(map, point) {
    if (!map.getLayer(VIEWSHED_POINT_LAYER_ID)) {
      return null;
    }
    const features = map.queryRenderedFeatures(point, {layers: [VIEWSHED_POINT_LAYER_ID]});
    return features && features.length ? features[0] : null;
  }

  function queryHoverFeature(state, map, point) {
    const projectedOverlayFeature = queryProjectedOverlayFeature(state, point);
    if (projectedOverlayFeature) {
      return projectedOverlayFeature;
    }
    const layers = [VIEWSHED_POINT_LAYER_ID, NODE_LAYER_ID].filter((layerId) => map.getLayer(layerId));
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
    map.getCanvas().style.cursor = (queryViewshedPointFeature(map, point) || queryNodeFeature(map, point)) ? "pointer" : "";
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
    if (result.viewshedPointData !== undefined) {
      setProp("viewshed-point-store", "data", result.viewshedPointData);
    }
    if (result.viewshedAssessmentData !== undefined) {
      setProp("viewshed-assessment-store", "data", result.viewshedAssessmentData);
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

    if (mode === "viewshed-point") {
      return {
        clickMode: {mode: "none", node_id: null},
        viewshedPointData: {
          longitude: longitude,
          latitude: latitude,
        },
        viewshedAssessmentData: null,
        message: `Viewshed assessment point selected at (${longitude.toFixed(5)}, ${latitude.toFixed(5)}).`,
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
      return buildAddNodeInteractionResult(runtime, longitude, latitude, rawName, "native map click");
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

  function syncBaseMapStyle(map, spec) {
    if (!map || !spec || !spec.base_style) {
      return;
    }

    const styleSpec = spec.base_style || {};
    const sourceSpec = ((styleSpec.sources || {})[BASEMAP_SOURCE_ID]) || null;
    const layerSpec = Array.isArray(styleSpec.layers)
      ? styleSpec.layers.find((candidate) => candidate && candidate.id === BASEMAP_LAYER_ID) || null
      : null;
    if (!sourceSpec || !layerSpec) {
      return;
    }

    const beforeId = firstPresentLayerId(map, [
      TERRAIN_COLOR_RELIEF_LAYER_ID,
      TERRAIN_LAYER_ID,
      WORLDCOVER_LAYER_ID,
      VIEWSHED_LAYER_ID,
      RSSI_LAYER_ID,
      LOADED_BBOX_LAYER_ID,
      VIEWSHED_POINT_LAYER_ID,
      NODE_LAYER_ID,
      NODE_LABEL_LAYER_ID,
    ]);

    removeLayerIfPresent(map, BASEMAP_LAYER_ID);
    removeSourceIfPresent(map, BASEMAP_SOURCE_ID);
    map.addSource(BASEMAP_SOURCE_ID, Object.assign({}, sourceSpec, {
      tiles: Array.isArray(sourceSpec.tiles) ? sourceSpec.tiles.slice() : sourceSpec.tiles,
    }));
    map.addLayer(Object.assign({}, layerSpec, {
      id: BASEMAP_LAYER_ID,
      source: BASEMAP_SOURCE_ID,
      layout: Object.assign({}, layerSpec.layout || {}),
      paint: Object.assign({}, layerSpec.paint || {}),
    }), beforeId);
  }

  function syncImageLayer(state, map, sourceId, layerId, layerSpec, beforeId) {
    if (!layerSpec || !layerSpec.image || !Array.isArray(layerSpec.coordinates)) {
      removeLayerIfPresent(map, layerId);
      removeSourceIfPresent(map, sourceId);
      if (state && state.imageLayerState) {
        delete state.imageLayerState[sourceId];
      }
      return;
    }
    const rasterOpacity = layerSpec.opacity === undefined || layerSpec.opacity === null ? 1.0 : Number(layerSpec.opacity);
    const rasterResampling = layerSpec.resampling === "nearest" ? "nearest" : "linear";
    const sourceSignature = JSON.stringify({
      image: String(layerSpec.image),
      coordinates: layerSpec.coordinates,
    });
    const previousSignature = state && state.imageLayerState ? state.imageLayerState[sourceId] : null;

    const existingSource = map.getSource(sourceId);
    if (!existingSource) {
      map.addSource(sourceId, {
        type: "image",
        url: layerSpec.image,
        coordinates: layerSpec.coordinates,
      });
    } else if (previousSignature !== sourceSignature && typeof existingSource.updateImage === "function") {
      existingSource.updateImage({
        url: layerSpec.image,
        coordinates: layerSpec.coordinates,
      });
    } else if (previousSignature !== sourceSignature) {
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
        "raster-opacity": rasterOpacity,
        "raster-fade-duration": 0,
        "raster-resampling": rasterResampling,
      },
    }, beforeId);
    if (state && state.imageLayerState) {
      state.imageLayerState[sourceId] = sourceSignature;
    }
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

  function forceMapRedraw(map) {
    if (!map) {
      return;
    }
    if (typeof map.redraw === "function") {
      map.redraw();
      return;
    }
    if (typeof map.triggerRepaint === "function") {
      map.triggerRepaint();
    }
  }

  function resetStyleDependentState(state) {
    if (!state) {
      return;
    }
    state.lastTerrainToken = null;
    state.imageLayerState = {};
    state.appliedSpecRevision = -1;
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

    if (terrainDem.color_relief) {
      const colorReliefBeforeId = firstPresentLayerId(map, [
        LOADED_BBOX_LAYER_ID,
        NODE_LAYER_ID,
        NODE_LABEL_LAYER_ID,
      ]);
      ensureLayer(map, {
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

    const rasterBeforeId = firstPresentLayerId(map, [
      LOADED_BBOX_LAYER_ID,
      NODE_LAYER_ID,
      NODE_LABEL_LAYER_ID,
    ]);
    syncImageLayer(state, map, TERRAIN_SOURCE_ID, TERRAIN_LAYER_ID, spec.terrain_layer, rasterBeforeId);
    syncImageLayer(state, map, WORLDCOVER_SOURCE_ID, WORLDCOVER_LAYER_ID, spec.worldcover_layer, rasterBeforeId);
    syncImageLayer(state, map, VIEWSHED_SOURCE_ID, VIEWSHED_LAYER_ID, spec.viewshed_layer, rasterBeforeId);
    syncImageLayer(state, map, RSSI_SOURCE_ID, RSSI_LAYER_ID, spec.rssi_layer, rasterBeforeId);

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

    setGeoJsonSourceData(map, VIEWSHED_POINT_SOURCE_ID, spec.viewshed_point);
    setGeoJsonSourceData(map, VIEWSHED_RADIUS_SOURCE_ID, spec.viewshed_radius_outline);
    setGeoJsonSourceData(map, VIEWSHED_SAMPLES_SOURCE_ID, spec.viewshed_samples);
    ensureLayer(map, {
      id: VIEWSHED_RADIUS_LAYER_ID,
      type: "line",
      source: VIEWSHED_RADIUS_SOURCE_ID,
      paint: {
        "line-color": ["coalesce", ["get", "color"], "#ef4444"],
        "line-width": 2,
        "line-opacity": 0.9,
        "line-dasharray": [2, 2],
      },
    });
    ensureLayer(map, {
      id: VIEWSHED_SAMPLES_LAYER_ID,
      type: "circle",
      source: VIEWSHED_SAMPLES_SOURCE_ID,
      paint: {
        "circle-radius": 3,
        "circle-color": ["coalesce", ["get", "color"], "#ef4444"],
        "circle-stroke-color": ["coalesce", ["get", "outline_color"], "#ffffff"],
        "circle-stroke-width": 1.25,
        "circle-opacity": 0.85,
        "circle-stroke-opacity": 0.85,
      },
    });
    ensureLayer(map, {
      id: VIEWSHED_POINT_LAYER_ID,
      type: "circle",
      source: VIEWSHED_POINT_SOURCE_ID,
      paint: {
        "circle-radius": 10,
        "circle-color": ["coalesce", ["get", "color"], "#ef4444"],
        "circle-stroke-color": ["coalesce", ["get", "outline_color"], "#ffffff"],
        "circle-stroke-width": 2.5,
        "circle-opacity": 0.95,
        "circle-stroke-opacity": 0.95,
      },
    });
    ensureLayer(map, {
      id: VIEWSHED_POINT_INNER_LAYER_ID,
      type: "circle",
      source: VIEWSHED_POINT_SOURCE_ID,
      paint: {
        "circle-radius": 3.5,
        "circle-color": "#111827",
        "circle-stroke-color": ["coalesce", ["get", "color"], "#ef4444"],
        "circle-stroke-width": 2.0,
        "circle-opacity": 0.95,
        "circle-stroke-opacity": 0.95,
      },
    });

    drawProjectedPathOverlay(state);
    forceMapRedraw(map);
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
      if (map.isStyleLoaded()) {
        syncBaseMapStyle(map, spec);
        forceMapRedraw(map);
      }
    }

    state.needsStyleRebuild = false;
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
      if (!state.needsStyleRebuild && map.isStyleLoaded()) {
        applyPendingSpec(state);
      }
    });

    map.on("style.load", function () {
      syncBaseMapStyle(map, state.pendingSpec);
      resetStyleDependentState(state);
      applyPendingSpec(state);
      maybeFitCamera(state, state.lastCameraBounds, state.lastCameraRevision);
      forceMapRedraw(map);
    });

    map.on("render", function () {
      drawProjectedPathOverlay(state);
    });

    map.on("moveend", function () {
      const bbox = viewportBounds(map);
      setProp("map-view-store", "data", bbox);
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

    map.on("mousedown", function (event) {
      if ((((state.runtime || {}).clickMode || {}).mode || "none") !== "terrain-bbox") {
        return;
      }
      beginTerrainBboxSelection(state, map, event);
    });

    map.on("click", function (event) {
      if (Date.now() < Number(state.suppressClickUntilMs || 0)) {
        return;
      }
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

  function renderNativeMap(
    spec,
    cameraBounds,
    cameraRevision,
    nodes,
    counter,
    selectedNodeIds,
    pointPathData,
    clickMode,
    viewshedPointData,
    viewshedAssessmentStore
  ) {
    const state = getControllerState();
    state.runtime = {
      nodes: (nodes || []).slice(),
      counter: Number(counter || 0),
      selectedNodeIds: (selectedNodeIds || []).slice(),
      clickMode: clickMode || {mode: "none", node_id: null},
    };
    if (((state.runtime || {}).clickMode || {}).mode !== "terrain-bbox" && state.bboxSelection && state.bboxSelection.active) {
      clearTerrainBboxSelection(state, state.map);
    }
    state.lastCameraBounds = cameraBounds || null;
    state.lastCameraRevision = Number(cameraRevision || 0);

    if (!window.maplibregl) {
      return `maplibre-missing:${Date.now()}`;
    }

    const map = ensureMap(state, spec || {}, cameraBounds || null);
    if (!map) {
      return `map-container-missing:${Date.now()}`;
    }

    const currentOverlayContextKey = overlayContextKey(
      nodes,
      selectedNodeIds,
      pointPathData,
      viewshedPointData,
      viewshedAssessmentStore
    );
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

  function applyManualNodeEntry(nClicks, manualName, manualLon, manualLat, nodes, counter, selectedNodeIds) {
    const noUpdate = window.dash_clientside.no_update;
    if (!nClicks) {
      return [noUpdate, noUpdate, noUpdate, noUpdate];
    }

    const resolvedName = String(readInputElementValue("manual-node-name", manualName) || "").trim();
    const longitude = parseCoordinateInputValue(readInputElementValue("manual-node-lon", manualLon));
    const latitude = parseCoordinateInputValue(readInputElementValue("manual-node-lat", manualLat));
    if (!resolvedName || longitude === null || latitude === null) {
      setProp("node-action-message", "children", "Manual node requires name, longitude, and latitude.");
      return [noUpdate, noUpdate, noUpdate, `manual-node-invalid:${Date.now()}`];
    }

    const state = getControllerState();
    state.runtime = {
      nodes: (nodes || []).slice(),
      counter: Number(counter || 0),
      selectedNodeIds: (selectedNodeIds || []).slice(),
      clickMode: {mode: "add-node", node_id: null},
    };
    const result = applyMapInteraction(state.runtime, {
      longitude: longitude,
      latitude: latitude,
      prompt: function () {
        return resolvedName;
      },
    });
    commitInteractionResult(state, result);
    clearInputElementValue("manual-node-name");
    clearInputElementValue("manual-node-lon");
    clearInputElementValue("manual-node-lat");
    setProp("manual-node-name", "value", "");
    setProp("manual-node-lon", "value", null);
    setProp("manual-node-lat", "value", null);
    return [noUpdate, noUpdate, noUpdate, `manual-node-added:${Date.now()}`];
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
    applyManualNodeEntry: applyManualNodeEntry,
  });
})();
