(function () {
  function selectPrimaryNode(currentSelected, nodeId) {
    const normalized = [];
    (currentSelected || []).forEach((value) => {
      const id = String(value);
      const existingIndex = normalized.indexOf(id);
      if (existingIndex !== -1) {
        normalized.splice(existingIndex, 1);
      }
      normalized.push(id);
    });
    const trimmed = normalized.slice(-2);
    const previousPrimary = trimmed.length ? trimmed[trimmed.length - 1] : null;
    const nextId = String(nodeId);
    if (previousPrimary === nextId) {
      return [nextId];
    }
    if (previousPrimary === null) {
      return [nextId];
    }
    return [previousPrimary, nextId];
  }

  function handleMapClick(clickData, clickMode, nodes, counter, selectedNodeIds) {
    const noUpdate = window.dash_clientside.no_update;

    if (window.__meshTerrainIgnoreClickUntil && Date.now() < window.__meshTerrainIgnoreClickUntil) {
      return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate];
    }
    if (!clickData || !clickData.points || !clickData.points.length) {
      return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate];
    }

    const mode = (clickMode || {}).mode || "none";
    if (mode === "none") {
      return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate];
    }

    const nodePoint = (clickData.points || []).find(
      (item) => item && Array.isArray(item.customdata) && item.customdata[0] === "node"
    );
    if (nodePoint) {
      return [noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate, noUpdate];
    }
    const point = clickData.points[0];
    const lon = Number(point.x);
    const lat = Number(point.y);

    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      return [
        noUpdate,
        noUpdate,
        noUpdate,
        {mode: "none", node_id: null},
        noUpdate,
        noUpdate,
        "Map click ended without a valid coordinate.",
        false,
      ];
    }

    if (mode === "point-path") {
      window.__meshTerrainIgnoreClickUntil = Date.now() + 800;
      return [
        noUpdate,
        noUpdate,
        noUpdate,
        {mode: "none", node_id: null},
        {
          source_node_id: String((clickMode || {}).node_id || ""),
          target_longitude: lon,
          target_latitude: lat,
        },
        noUpdate,
        `Path endpoint selected at (${lon.toFixed(5)}, ${lat.toFixed(5)}).`,
        false,
      ];
    }

    if (mode === "move-node") {
      const nodeId = String((clickMode || {}).node_id || "");
      const nextNodes = (nodes || []).map((node) => {
        if (String(node.id) !== nodeId) {
          return node;
        }
        return Object.assign({}, node, {
          longitude: lon,
          latitude: lat,
        });
      });
      window.__meshTerrainIgnoreClickUntil = Date.now() + 800;
      return [
        nextNodes,
        noUpdate,
        noUpdate,
        {mode: "none", node_id: null},
        noUpdate,
        noUpdate,
        `Moved node to (${lon.toFixed(5)}, ${lat.toFixed(5)}).`,
        true,
      ];
    }

    const rawName = window.prompt(`Node name for (${lon.toFixed(5)}, ${lat.toFixed(5)})`, "");

    if (rawName === null) {
      return [
        noUpdate,
        noUpdate,
        noUpdate,
        {mode: "none", node_id: null},
        noUpdate,
        noUpdate,
        "Click-to-add canceled.",
        false,
      ];
    }

    const name = rawName.trim();
    if (!name) {
      return [
        noUpdate,
        noUpdate,
        noUpdate,
        {mode: "none", node_id: null},
        noUpdate,
        noUpdate,
        "Node name cannot be empty.",
        false,
      ];
    }

    const nextCounter = Number(counter || 0) + 1;
    const nextNodeId = `node-${nextCounter}`;
    const nextNodes = (nodes || []).slice();
    nextNodes.push({
      id: nextNodeId,
      name: name,
      longitude: lon,
      latitude: lat,
      height_agl_m: 8.0,
      antenna_gain_dbi: 6.0,
      tx_power_dbm: 30.0,
    });
    window.__meshTerrainIgnoreClickUntil = Date.now() + 800;
    return [
      nextNodes,
      nextCounter,
      noUpdate,
      {mode: "none", node_id: null},
      noUpdate,
      selectPrimaryNode(selectedNodeIds, nextNodeId),
      `Added node ${name} from map click.`,
      true,
    ];
  }

  function startRssiOverlay(nClicks, nodes, includeGroundLoss) {
    const noUpdate = window.dash_clientside.no_update;
    if (!nClicks) {
      return [noUpdate, noUpdate, noUpdate, noUpdate];
    }
    const nodeCount = (nodes || []).length;
    if (!nodeCount) {
      return [noUpdate, true, noUpdate, noUpdate];
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
    ];
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside);
  window.dash_clientside.clientside = Object.assign({}, window.dash_clientside.clientside, {
    handleMapClick: handleMapClick,
    startRssiOverlay: startRssiOverlay,
  });
})();
