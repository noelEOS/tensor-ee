"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const elements = Object.fromEntries([
    "draw-button", "preset-button", "draw-help", "area-summary", "area-name", "area-size", "area-coordinates", "clear-area", "area-step", "scene-step", "run-step", "start-date", "end-date", "search-button", "scene-message", "scene-error", "scene-list", "grid-summary", "run-button", "run-help", "job-card", "job-status", "job-time", "job-message", "job-progress", "job-detached", "restore-job", "job-result-error", "job-details", "job-asset", "example-section", "example-button", "example-error", "global-error", "global-error-text", "retry-config", "fit-button", "map-kicker", "map-title", "map-subtitle", "drawing-banner", "drawing-instruction", "map-feedback", "layers-panel", "layer-origin", "layer-opacity", "opacity-value", "legend-title", "legend-gradient", "legend-labels", "legend-water", "map-empty-note"
  ].map((id) => [id, $(id)]));
  const state = {
    config: null, token: null, aoi: null, areaName: "Selected area", scenes: [], scene: null, grid: null,
    drawing: false, firstCorner: null, rectangle: null, chipRectangle: null, draft: null, marker: null,
    queryRevision: 0, querySequence: 0, actionRevision: 0, previewSequence: 0,
    scenesLoading: false, exampleLoading: false, submitting: false, sceneSearchVerified: false,
    scenesController: null, previewController: null, job: null, pollTimer: null, pollInFlight: false,
    layers: {}, activeLayer: null, layerBounds: null, tileLayer: null, resultKind: null,
    fitBounds: null, lastTileError: 0
  };
  let map;

  function setText(id, value) { elements[id].textContent = value; }
  function show(id, visible = true) { elements[id].hidden = !visible; }
  function errorMessage(error) { return error instanceof Error ? error.message : String(error); }
  function formatDate(value) {
    if (!value) return "Sentinel-1 scene";
    const date = new Date(`${String(value).slice(0, 10)}T12:00:00Z`);
    return Number.isNaN(date.valueOf()) ? String(value).slice(0, 10) : date.toLocaleDateString("en-GB", {day:"2-digit", month:"short", year:"numeric", timeZone:"UTC"});
  }
  function sceneDate(scene) {
    if (scene?.date) return String(scene.date).slice(0, 10);
    const match = /_(\d{4})(\d{2})(\d{2})T/.exec(scene?.id || "");
    return match ? `${match[1]}-${match[2]}-${match[3]}` : null;
  }
  function currentSignature() { return JSON.stringify([state.aoi, elements["start-date"].value, elements["end-date"].value, state.scene?.id || null]); }
  function jobIsAttached() { return Boolean(state.job && state.job.signature === currentSignature()); }
  function isActiveJob(job = state.job) { return Boolean(job && !["COMPLETED", "FAILED", "CANCELLED"].includes(String(job.data.state).toUpperCase())); }
  function totalTiles() {
    const grid = state.grid;
    if (!grid) return null;
    return grid.tile_count ?? grid.tiles ?? null;
  }
  function exampleTitle() { return state.config?.example_title || "Completed example"; }
  function covered(scene = state.scene) { return scene?.coverage_fraction == null || Number(scene.coverage_fraction) >= 0.999; }

  async function api(path, body, signal) {
    const options = {signal, headers:{Accept:"application/json"}};
    if (body !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.headers["X-Demo-Token"] = state.token || "";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The demo server returned an unreadable response (${response.status}).`); }
    if (!response.ok || data.error) throw new Error(data.error || `The request failed (${response.status}).`);
    return data;
  }

  function feedback(message = "", error = false) {
    setText("map-feedback", message);
    elements["map-feedback"].classList.toggle("error", error);
    show("map-feedback", Boolean(message));
    // Keep messages clear of the layer controls.
    elements["map-feedback"].style.bottom = Object.keys(state.layers).length ? "228px" : "30px";
  }

  function geometryBounds(geometry) {
    const points = geometry?.coordinates?.flat(geometry.type === "MultiPolygon" ? 2 : 1) || [];
    if (!points.length) return null;
    return L.latLngBounds(points.map(([lon, lat]) => [lat, lon]));
  }
  function boundsGeometry(bounds) {
    const south = bounds.getSouth(), west = bounds.getWest(), north = bounds.getNorth(), east = bounds.getEast();
    return {type:"Polygon", coordinates:[[[west,south],[east,south],[east,north],[west,north],[west,south]]]};
  }
  function fit(bounds) { if (bounds?.isValid()) map.fitBounds(bounds, {padding:[35,35], maxZoom:15, animate:!window.matchMedia("(prefers-reduced-motion: reduce)").matches}); }
  function areaKm2(bounds) {
    const latitude = (bounds.getSouth() + bounds.getNorth()) / 2 * Math.PI / 180;
    return Math.abs((bounds.getNorth() - bounds.getSouth()) * 111.32 * (bounds.getEast() - bounds.getWest()) * 111.32 * Math.cos(latitude));
  }

  function clearLayers() {
    state.previewSequence++;
    state.previewController?.abort();
    state.previewController = null;
    if (state.tileLayer) map.removeLayer(state.tileLayer);
    state.tileLayer = null;
    state.layers = {};
    state.activeLayer = null;
    state.layerBounds = null;
    state.resultKind = null;
    state.fitBounds = state.aoi ? geometryBounds(state.aoi) : null;
    show("layers-panel", false);
    show("map-empty-note", true);
    setText("map-empty-note", "Choose a scene to preview the radar image.");
    feedback();
    renderMapTitle();
  }

  function invalidateScenes() {
    if (state.chipRectangle) map.removeLayer(state.chipRectangle);
    state.chipRectangle = null;
    state.queryRevision++;
    state.actionRevision++;
    state.querySequence++;
    state.scenesController?.abort();
    state.scenesController = null;
    state.scenesLoading = false;
    state.scenes = [];
    state.sceneSearchVerified = false;
    state.scene = null;
    state.grid = null;
    elements["scene-list"].replaceChildren();
    show("scene-error", false);
    show("grid-summary", false);
    clearLayers();
    setText("scene-message", state.aoi ? "Search for scenes covering your selected area." : "Select an area to search Sentinel-1.");
    renderControls();
    renderJob();
  }

  function setArea(geometry, name = "Selected area", shouldFit = true) {
    stopDrawing();
    if (state.rectangle) map.removeLayer(state.rectangle);
    state.rectangle = null;
    state.aoi = geometry;
    state.areaName = name;
    if (geometry) {
      const bounds = geometryBounds(geometry);
      state.rectangle = L.geoJSON(geometry, {pane:"aoiPane", interactive:false, style:{color:"#137a70", weight:2, fillColor:"#5aab8d", fillOpacity:0.07, dashArray:"6 4"}}).addTo(map);
      state.fitBounds = bounds;
      const area = areaKm2(bounds);
      setText("area-name", name);
      setText("area-size", `${area < 1 ? area.toFixed(2) : area.toFixed(1)} km²`);
      setText("area-coordinates", `${bounds.getSouth().toFixed(4)}° N, ${bounds.getWest().toFixed(4)}° E → ${bounds.getNorth().toFixed(4)}° N, ${bounds.getEast().toFixed(4)}° E`);
      if (shouldFit) fit(bounds);
    }
    show("area-summary", Boolean(geometry));
    invalidateScenes();
    renderMapTitle();
  }

  function stopDrawing() {
    state.drawing = false;
    state.firstCorner = null;
    if (state.draft) map.removeLayer(state.draft);
    if (state.marker) map.removeLayer(state.marker);
    state.draft = null;
    state.marker = null;
    $("map").classList.remove("drawing");
    show("drawing-banner", false);
    if (map) map.doubleClickZoom.enable();
    renderControls();
  }
  function startDrawing() {
    if (state.drawing) { stopDrawing(); return; }
    state.actionRevision++;
    state.querySequence++;
    state.scenesController?.abort();
    state.scenesLoading = false;
    state.drawing = true;
    state.firstCorner = null;
    $("map").classList.add("drawing");
    map.doubleClickZoom.disable();
    setText("drawing-instruction", "Click the first corner");
    show("drawing-banner");
    feedback();
    renderControls();
  }
  function onMapClick(event) {
    if (!state.drawing) return;
    const corner = L.latLng(Math.max(-85, Math.min(85, event.latlng.lat)), event.latlng.wrap().lng);
    if (!state.firstCorner) {
      state.firstCorner = corner;
      state.marker = L.circleMarker(corner, {pane:"aoiPane", radius:4, weight:2, color:"#fffef7", fillColor:"#0e716b", fillOpacity:1, interactive:false}).addTo(map);
      setText("drawing-instruction", "Click the opposite corner");
      return;
    }
    const bounds = L.latLngBounds(state.firstCorner, corner);
    if (bounds.getEast() - bounds.getWest() > 180) { feedback("Choose an area that does not cross the antimeridian.", true); return; }
    if (Math.abs(bounds.getNorth() - bounds.getSouth()) < 0.00001 || Math.abs(bounds.getEast() - bounds.getWest()) < 0.00001) { feedback("Move farther from the first corner to draw a rectangle.", true); return; }
    setArea(boundsGeometry(bounds), "Drawn area", false);
  }
  function onMapMove(event) {
    if (!state.drawing || !state.firstCorner) return;
    const bounds = L.latLngBounds(state.firstCorner, event.latlng);
    if (state.draft) state.draft.setBounds(bounds);
    else state.draft = L.rectangle(bounds, {pane:"aoiPane", interactive:false, color:"#0e716b", fillColor:"#6bb391", fillOpacity:0.11, weight:1.5, dashArray:"5 4"}).addTo(map);
  }

  function renderMapTitle() {
    if (state.resultKind === "example") {
      setText("map-kicker", "COMPLETED EXAMPLE"); setText("map-title", exampleTitle()); setText("map-subtitle", "A previous Earth Engine result, not a new run.");
    } else if (state.resultKind === "result") {
      setText("map-kicker", "INFERENCE COMPLETE"); setText("map-title", state.areaName); setText("map-subtitle", `${formatDate(sceneDate(state.scene))} · SegWater prediction`);
    } else if (state.layers.source && state.scene) {
      setText("map-kicker", "SENTINEL-1 · VV"); setText("map-title", state.areaName); setText("map-subtitle", `${formatDate(sceneDate(state.scene))} · Selected radar scene`);
    } else {
      setText("map-kicker", "AREA EXPLORER"); setText("map-title", state.aoi ? state.areaName : "Sun Moon Lake, Taiwan"); setText("map-subtitle", state.aoi ? "Your area is ready. Find an available radar scene." : "Draw a small area or use the lake preset.");
    }
  }

  function renderControls() {
    const configured = Boolean(state.config);
    const tileCount = totalTiles();
    const tooLarge = tileCount != null && Number(tileCount) > (state.config?.max_tiles || 4);
    elements["draw-button"].disabled = !configured;
    elements["draw-button"].querySelector("span").textContent = state.drawing ? "Cancel drawing" : state.aoi ? "Draw new area" : "Draw an area";
    elements["draw-button"].classList.toggle("button-soft", state.drawing);
    elements["preset-button"].disabled = !configured;
    elements["start-date"].disabled = !configured;
    elements["end-date"].disabled = !configured;
    elements["search-button"].disabled = !configured || !state.aoi || state.drawing || state.scenesLoading;
    elements["search-button"].querySelector("span").textContent = state.scenesLoading ? "Finding scenes…" : "Find available scenes";
    elements["run-button"].disabled = !configured || !state.aoi || !state.scene || !state.sceneSearchVerified || state.drawing || state.scenesLoading || state.submitting || isActiveJob() || !covered() || tooLarge;
    elements["run-button"].querySelector("span").textContent = state.submitting ? "Starting inference…" : isActiveJob() ? "Inference in progress" : "Run native inference";
    elements["fit-button"].disabled = !state.fitBounds;
    elements["area-step"].classList.toggle("done", Boolean(state.aoi));
    elements["scene-step"].classList.toggle("done", Boolean(state.scene));
    elements["run-step"].classList.toggle("done", Boolean(state.job?.data.state === "COMPLETED" && jobIsAttached()));
    let help = state.config?.model === "swin-t" ? "Three EE exports run in sequence. Allow several minutes." : "Batch inference can take several minutes.";
    if (state.scene && !covered()) help = "This scene only partly covers the area. Choose full coverage to run inference.";
    else if (tooLarge) help = `This area needs ${tileCount} chips. Draw a smaller area (maximum ${state.config?.max_tiles || 4}).`;
    else if (isActiveJob() && !jobIsAttached()) help = "The previous run is still in progress. You can prepare your next selection.";
    else if (state.scene && !state.sceneSearchVerified && !isActiveJob()) help = "Find available scenes to run this area again.";
    setText("run-help", help);
    elements["example-button"].disabled = !configured || state.exampleLoading;
    elements["example-button"].querySelector("strong").textContent = state.exampleLoading ? "Opening example…" : exampleTitle();
  }

  function renderScenes() {
    elements["scene-list"].replaceChildren();
    state.scenes.forEach((scene, index) => {
      const label = document.createElement("label");
      label.className = `scene-option${scene.id === state.scene?.id ? " selected" : ""}${covered(scene) ? "" : " partial"}`;
      label.title = scene.id;
      const input = document.createElement("input");
      input.type = "radio"; input.name = "scene"; input.value = String(index); input.checked = scene.id === state.scene?.id;
      input.addEventListener("change", () => selectScene(scene));
      const text = document.createElement("span"); text.className = "scene-text";
      const main = document.createElement("span"); main.className = "scene-main";
      const date = document.createElement("span"); date.className = "scene-date"; date.textContent = formatDate(sceneDate(scene));
      const orbit = document.createElement("span"); orbit.className = "scene-orbit";
      orbit.textContent = String(scene.orbit || "RADAR").toUpperCase().replace("ASCENDING", "ASCENDING ↗").replace("DESCENDING", "DESCENDING ↘");
      const info = document.createElement("span"); info.className = "scene-info";
      const details = [];
      if (scene.relative_orbit != null) details.push(`Orbit ${scene.relative_orbit}`);
      if (scene.coverage_fraction != null) {
        const percent = Number(scene.coverage_fraction) * 100;
        details.push(`${covered(scene) ? "100" : percent.toFixed(1)}% ${state.config?.model === "swin-t" ? "chip" : "area"} coverage`);
      } else details.push("Sentinel-1 GRD");
      info.textContent = details.join(" · ");
      main.append(date, orbit); text.append(main, info); label.append(input, text); elements["scene-list"].append(label);
    });
    if (state.grid) {
      const count = totalTiles();
      const parts = [];
      if (count != null) parts.push(`${count} model chip${Number(count) === 1 ? "" : "s"}`);
      if (state.grid.width != null && state.grid.height != null) parts.push(`${state.grid.width} × ${state.grid.height} px`);
      setText("grid-summary", parts.join(" · "));
      elements["grid-summary"].classList.toggle("too-large", Number(count) > (state.config?.max_tiles || 4));
      show("grid-summary", Boolean(parts.length));
    }
  }

  async function findScenes() {
    if (!state.aoi || state.drawing || state.scenesLoading) return;
    const start = elements["start-date"].value, end = elements["end-date"].value;
    if (!start || !end || end < start) { setText("scene-error", "Choose an end date on or after the start date."); show("scene-error"); return; }
    const inclusiveDays = Math.round((Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86400000) + 1;
    if (inclusiveDays > 31) { setText("scene-error", "Choose a date range of up to 31 days."); show("scene-error"); return; }
    const revision = state.queryRevision;
    const sequence = ++state.querySequence;
    state.scenesController?.abort();
    const controller = new AbortController(); state.scenesController = controller;
    state.scenesLoading = true;
    state.scenes = []; state.scene = null; state.grid = null; state.sceneSearchVerified = false;
    state.actionRevision++;
    clearLayers(); renderScenes(); renderControls(); renderJob();
    show("scene-error", false); show("grid-summary", false);
    setText("scene-message", "Looking for Sentinel-1 scenes covering your area…");
    try {
      const data = await api("/api/scenes", {aoi:state.aoi, start, end}, controller.signal);
      if (revision !== state.queryRevision || sequence !== state.querySequence) return;
      state.scenes = Array.isArray(data.scenes) ? data.scenes : [];
      state.sceneSearchVerified = true;
      state.grid = data.grid || null;
      if (state.grid?.chip_aoi) {
        state.chipRectangle = L.geoJSON(state.grid.chip_aoi, {pane:"aoiPane", interactive:false,
          style:{color:"#137a70", weight:2, fillOpacity:0}}).addTo(map);
        fit(geometryBounds(state.grid.chip_aoi));
      }
      state.scenesLoading = false;
      const count = state.scenes.length;
      setText("scene-message", count ? `${count} available scene${count === 1 ? "" : "s"}${data.truncated ? " · showing the first matches" : ""}. Select one below.` : "No scenes found. Try a wider date range or a different area.");
      renderScenes();
      if (count) selectScene(state.scenes.find(covered) || state.scenes[0]);
    } catch (error) {
      if (revision !== state.queryRevision || sequence !== state.querySequence || error.name === "AbortError") return;
      state.scenesLoading = false;
      setText("scene-error", errorMessage(error)); show("scene-error");
      setText("scene-message", "Update your selection and try again.");
    } finally {
      if (revision === state.queryRevision && sequence === state.querySequence) { state.scenesLoading = false; renderControls(); }
    }
  }

  function selectScene(scene, preview = true) {
    if (scene.id !== state.scene?.id) { state.actionRevision++; clearLayers(); }
    state.scene = scene;
    renderScenes(); renderControls(); renderJob();
    if (preview) previewScene();
  }
  async function previewScene() {
    if (!state.aoi || !state.scene) return;
    const signature = currentSignature();
    const sequence = ++state.previewSequence;
    state.previewController?.abort();
    const controller = new AbortController(); state.previewController = controller;
    feedback("Loading the selected radar scene…");
    try {
      const data = await api("/api/preview", {aoi:state.aoi, scene_id:state.scene.id}, controller.signal);
      if (sequence !== state.previewSequence || signature !== currentSignature()) return;
      if (!data.layers?.source) throw new Error("The selected scene has no radar preview.");
      state.layers = {source:data.layers.source};
      state.layerBounds = data.bounds || geometryBounds(state.aoi);
      state.fitBounds = L.latLngBounds(state.layerBounds);
      state.resultKind = "preview";
      feedback();
      renderLayers("source"); renderMapTitle(); renderControls();
    } catch (error) {
      if (sequence !== state.previewSequence || signature !== currentSignature() || error.name === "AbortError") return;
      feedback(`Radar preview unavailable: ${errorMessage(error)}`, true);
    }
  }

  function renderLayers(preferred = state.activeLayer) {
    const available = Object.keys(state.layers).filter((key) => state.layers[key]);
    show("layers-panel", Boolean(available.length)); show("map-empty-note", !available.length);
    document.querySelectorAll(".layer-option").forEach((button) => { button.disabled = !state.layers[button.dataset.layer]; });
    setText("layer-origin", state.resultKind === "example" ? "Completed example" : state.resultKind === "result" ? "Inference result" : "Source preview");
    const selected = state.layers[preferred] ? preferred : available.includes("probability") ? "probability" : available[0];
    if (selected) chooseLayer(selected);
    if (!elements["map-feedback"].hidden) elements["map-feedback"].style.bottom = available.length ? "228px" : "30px";
  }
  function chooseLayer(key) {
    if (!state.layers[key]) return;
    if (state.tileLayer) map.removeLayer(state.tileLayer);
    state.activeLayer = key;
    state.lastTileError = 0;
    const entry = state.layers[key];
    const url = typeof entry === "string" ? entry : entry.url;
    state.tileLayer = L.tileLayer(url, {pane:"resultPane", maxZoom:19, maxNativeZoom:19, opacity:Number(elements["layer-opacity"].value)/100, bounds:state.layerBounds || undefined, noWrap:true, attribution:"Google Earth Engine · Copernicus Sentinel-1"});
    state.tileLayer.on("tileerror", () => {
      state.lastTileError++;
      if (state.lastTileError === 2) {
        const message = "Some map tiles could not load. Select another layer or refresh the result.";
        feedback(message, true);
        if (state.job?.data.state === "COMPLETED" && jobIsAttached()) { setText("job-result-error", message); show("job-result-error"); }
      }
    });
    state.tileLayer.addTo(map);
    document.querySelectorAll(".layer-option").forEach((button) => button.setAttribute("aria-checked", String(button.dataset.layer === key)));
    show("legend-gradient", key !== "water"); show("legend-labels", key !== "water"); show("legend-water", key === "water");
    if (key === "source") {
      setText("legend-title", "Radar backscatter · VV");
      elements["legend-gradient"].style.background = "linear-gradient(90deg, #000, #fff)";
      elements["legend-labels"].firstElementChild.textContent = "−25 dB";
      elements["legend-labels"].lastElementChild.textContent = "0 dB";
    } else if (key === "probability") {
      setText("legend-title", "Water probability");
      elements["legend-gradient"].style.background = "linear-gradient(90deg, #f7fcff, #c6dbef, #6baed6, #2171b5, #08306b)";
      elements["legend-labels"].firstElementChild.textContent = "0 · low";
      elements["legend-labels"].lastElementChild.textContent = "1 · high";
    } else setText("legend-title", "Predicted water");
  }

  function elapsedLabel(seconds) {
    if (seconds == null || !Number.isFinite(Number(seconds))) return "";
    const total = Math.max(0, Math.floor(Number(seconds)));
    return total < 60 ? `${total}s` : `${Math.floor(total/60)}m ${String(total%60).padStart(2,"0")}s`;
  }
  function renderJob() {
    const job = state.job;
    show("job-card", Boolean(job));
    if (!job) return;
    const data = job.data;
    const status = String(data.state || "BUILDING").toUpperCase();
    const labels = {BUILDING:"Building the inference", READY:"Queued in Earth Engine", QUEUED:"Queued in Earth Engine", RECONNECTING:"Reconnecting to Earth Engine", RUNNING:"Inference is running", COMPLETED:job.example ? "Example completed" : "Inference completed", FAILED:"Inference failed", CANCELLED:"Inference cancelled"};
    setText("job-status", data.stage_index != null && status !== "COMPLETED" && status !== "FAILED" && status !== "CANCELLED"
      ? `Stage ${data.stage_index + 1}/3 · ${status === "BUILDING" ? "Preparing" : status === "READY" ? "Queued" : status === "RUNNING" ? "Running" : "Reconnecting"}`
      : labels[status] || status);
    setText("job-time", elapsedLabel(data.elapsed_seconds));
    const fallbacks = {BUILDING:"Preparing the native model and selected radar input.", READY:"Earth Engine will start the batch job when capacity is available.", QUEUED:"Earth Engine will start the batch job when capacity is available.", RECONNECTING:"Waiting to reconnect. The submitted Earth Engine job continues running.", RUNNING:"Earth Engine is computing your water prediction. This can take several minutes.", COMPLETED:"Water probability, the water mask, and the source scene are ready.", FAILED:"The job did not complete. Review the message and try again."};
    setText("job-message", data.message || fallbacks[status] || "Checking the Earth Engine job.");
    elements["job-card"].classList.toggle("completed", status === "COMPLETED");
    elements["job-card"].classList.toggle("failed", status === "FAILED" || status === "CANCELLED");
    const step = status === "COMPLETED" ? 4 : data.stage_index != null ? data.stage_index + 1
      : ({BUILDING:0, READY:1, QUEUED:1, RUNNING:2}[status] ?? 0);
    [...elements["job-progress"].children].forEach((segment,index) => { segment.classList.toggle("filled", index < step); segment.classList.toggle("current", index === step && step < 4); });
    show("job-progress", status !== "FAILED" && status !== "CANCELLED");
    show("job-detached", !jobIsAttached());
    show("job-details", Boolean(data.asset_id));
    setText("job-asset", data.asset_id || ""); elements["job-asset"].title = data.asset_id || "";
    renderControls();
  }
  function snapshot() { return {aoi:structuredClone(state.aoi), areaName:state.areaName, scene:{...state.scene}, scenes:state.scenes.map((scene) => ({...scene})), grid:state.grid, sceneSearchVerified:state.sceneSearchVerified, start:elements["start-date"].value, end:elements["end-date"].value}; }

  async function runInference() {
    if (elements["run-button"].disabled) return;
    state.submitting = true;
    show("job-result-error", false);
    const job = {data:{state:"BUILDING"}, signature:currentSignature(), snapshot:snapshot(), example:false, applied:false};
    state.job = job;
    renderJob(); renderControls();
    try {
      const data = await api("/api/run", {aoi:job.snapshot.aoi, scene_id:job.snapshot.scene.id});
      if (state.job !== job) return;
      if (!data.job_id) throw new Error("The server did not return a job identifier.");
      job.data = data;
      state.submitting = false;
      renderJob();
      if (String(data.state).toUpperCase() === "COMPLETED") applyJobLayers(job);
      else pollJob(job, true);
    } catch (error) {
      if (state.job !== job) return;
      job.data = {state:"FAILED", message:errorMessage(error)};
      state.submitting = false;
      renderJob(); renderControls();
    }
  }

  async function pollJob(job, immediate = false) {
    clearTimeout(state.pollTimer);
    if (state.job !== job || !job.data.job_id) return;
    if (!immediate) { state.pollTimer = setTimeout(() => pollJob(job, true), 5000); return; }
    if (state.pollInFlight) { state.pollTimer = setTimeout(() => pollJob(job, true), 5000); return; }
    state.pollInFlight = true;
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(job.data.job_id)}`);
      if (state.job !== job) return;
      job.data = data;
      renderJob();
      const status = String(data.state).toUpperCase();
      if (status === "COMPLETED") applyJobLayers(job);
      else if (!["FAILED", "CANCELLED"].includes(status)) pollJob(job);
    } catch (error) {
      if (state.job !== job) return;
      setText("job-message", `Status check interrupted: ${errorMessage(error)} Retrying shortly.`);
      pollJob(job);
    } finally { state.pollInFlight = false; }
  }

  function applyJobLayers(job) {
    if (!jobIsAttached() || job.applied) return;
    const layers = job.data.layers || {};
    if (!Object.keys(layers).length) {
      setText("job-result-error", job.data.layer_error ? `The job completed, but map display failed: ${job.data.layer_error}` : "The job completed, but its map layers are unavailable. Refresh the page to load the result again."); show("job-result-error"); return;
    }
    state.previewSequence++; state.previewController?.abort();
    state.layers = layers;
    state.layerBounds = job.data.bounds || geometryBounds(job.snapshot.aoi);
    state.fitBounds = L.latLngBounds(state.layerBounds);
    state.resultKind = job.example ? "example" : "result";
    job.applied = true;
    show("job-result-error", false); feedback();
    renderLayers("probability"); renderMapTitle(); renderControls();
    fit(state.fitBounds);
  }

  function restoreJob() {
    const job = state.job;
    if (!job?.snapshot?.aoi) return;
    elements["start-date"].value = job.snapshot.start;
    elements["end-date"].value = job.snapshot.end;
    setArea(job.snapshot.aoi, job.snapshot.areaName, false);
    state.scenes = job.snapshot.scenes.length ? job.snapshot.scenes : [job.snapshot.scene];
    state.scene = job.snapshot.scene; state.grid = job.snapshot.grid; state.sceneSearchVerified = Boolean(job.snapshot.sceneSearchVerified);
    job.signature = currentSignature(); job.applied = false;
    renderScenes(); renderJob(); renderControls();
    setText("scene-message", job.example ? "Scene used by the completed example." : "Scene used by this inference run.");
    if (String(job.data.state).toUpperCase() === "COMPLETED") applyJobLayers(job);
    else { fit(geometryBounds(state.aoi)); previewScene(); }
  }

  function installExistingJob(data, example = false) {
    const scene = data.scene || {id:data.scene_id || "Saved Sentinel-1 scene", date:data.scene_date || null, orbit:data.orbit || "RADAR"};
    const geometry = data.aoi || (data.bounds ? boundsGeometry(L.latLngBounds(data.bounds)) : state.config.default_aoi);
    setArea(geometry, example ? exampleTitle() : "Saved inference area", false);
    state.scenes = [scene]; state.scene = scene; state.grid = data.grid || null;
    const job = {data, signature:currentSignature(), snapshot:snapshot(), example, applied:false};
    state.job = job;
    state.submitting = false;
    show("job-result-error", false);
    renderScenes(); renderJob(); renderControls();
    setText("scene-message", example ? "Scene used by the completed example." : "Restored your most recent inference run.");
    if (String(data.state).toUpperCase() === "COMPLETED") applyJobLayers(job);
    else if (isActiveJob(job)) { fit(geometryBounds(geometry)); pollJob(job, true); }
  }

  async function openExample() {
    if (state.exampleLoading || !state.config) return;
    const revision = ++state.actionRevision;
    state.exampleLoading = true; renderControls(); show("example-error", false);
    try {
      const data = await api("/api/example");
      if (revision !== state.actionRevision) return;
      if (String(data.state).toUpperCase() !== "COMPLETED") throw new Error("The example has not completed yet.");
      if (isActiveJob()) {
        // Keep tracking a running job while displaying the example.
        state.previewSequence++; state.previewController?.abort();
        state.layers = data.layers || {}; state.layerBounds = data.bounds || geometryBounds(state.config.default_aoi);
        state.fitBounds = L.latLngBounds(state.layerBounds); state.resultKind = "example";
        renderLayers("probability"); renderMapTitle(); renderControls(); fit(state.fitBounds); feedback();
      } else installExistingJob(data, true);
    } catch (error) {
      if (revision !== state.actionRevision) return;
      setText("example-error", errorMessage(error)); show("example-error");
    } finally { state.exampleLoading = false; renderControls(); }
  }

  function readToken() {
    // The server printed the page URL with the session token in its fragment; the browser never
    // sends the fragment, the page sends the token as X-Demo-Token on every POST.
    const token = new URLSearchParams(location.hash.replace(/^#/, "")).get("token");
    state.token = token || null;
    if (!state.token) {
      setText("global-error-text", "Open the demo from the URL the server printed (it ends with #token=…); " +
              "without it the page can show results but cannot start a search or a run.");
      show("global-error");
    }
  }

  async function loadConfig() {
    show("global-error", false);
    readToken();
    const revision = state.actionRevision;
    try {
      const config = await api("/api/config");
      state.config = config;
      if (config.model === "swin-t") {
        document.title = "SegWater | Swin-T + UPerNet in Earth Engine";
        $("model-label").textContent = "Swin-T + UPerNet";
        $("intro-title").textContent = "Map one radar chip.";
        $("intro-copy").textContent = "Run the trained Swin-T + UPerNet in Earth Engine. One 224×224 chip, from radar to water.";
        $("limit-note").textContent = "One 224×224 chip at 10 m. Use the lake preset or draw a small area in Taiwan’s UTM zone 51N. The full snapped chip is mapped.";
        $("footer-note").textContent = "Trained step-6000 model. All 3 inference stages run in Earth Engine.";
        elements["preset-button"].textContent = "Lake chip";
      }
      if (config.offline) {
        // Offline fixture: say so wherever a result could be mistaken for a real one.
        $("local-badge").classList.add("offline");
        $("local-badge-text").textContent = "Offline fixture";
        $("footer-note").textContent = "Offline fixture: no Earth Engine requests. Scenes, map tiles and task progress are simulated; graphs are built locally.";
      }
      if (!state.aoi && revision === state.actionRevision) {
        elements["start-date"].value = config.default_dates?.start || "2024-01-01";
        elements["end-date"].value = config.default_dates?.end || "2024-01-07";
        if (config.default_aoi) fit(geometryBounds(config.default_aoi));
      }
      show("example-section", Boolean(config.example_available));
      renderControls();
      const jobId = config.active_job_id || config.last_job_id;
      if (jobId && revision === state.actionRevision) {
        try {
          const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
          if (revision === state.actionRevision && !state.job) installExistingJob(data);
        } catch (error) { feedback(`The saved run could not be restored: ${errorMessage(error)}`, true); }
      }
    } catch (error) {
      setText("global-error-text", `Could not connect to the local demo. ${errorMessage(error)}`); show("global-error");
    }
  }

  if (!window.L) {
    setText("global-error-text", "The map library could not load. Check your internet connection and reload this page."); show("global-error");
    elements["retry-config"].textContent = "Reload page";
    elements["retry-config"].addEventListener("click", () => window.location.reload());
    return;
  }
  map = L.map("map", {zoomControl:false, minZoom:3, maxZoom:19}).setView([23.862, 120.914], 13);
  map.createPane("resultPane"); map.getPane("resultPane").style.zIndex = 300; map.getPane("resultPane").classList.add("result-pane");
  map.createPane("aoiPane"); map.getPane("aoiPane").style.zIndex = 450;
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom:19, attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
  L.control.zoom({position:"bottomright"}).addTo(map);
  map.on("click", onMapClick); map.on("mousemove", onMapMove);
  elements["draw-button"].addEventListener("click", startDrawing);
  elements["preset-button"].addEventListener("click", () => setArea(state.config.default_aoi, "Sun Moon Lake", true));
  elements["clear-area"].addEventListener("click", () => setArea(null));
  elements["fit-button"].addEventListener("click", () => fit(state.fitBounds));
  elements["search-button"].addEventListener("click", findScenes);
  elements["run-button"].addEventListener("click", runInference);
  elements["restore-job"].addEventListener("click", restoreJob);
  elements["example-button"].addEventListener("click", openExample);
  elements["retry-config"].addEventListener("click", loadConfig);
  ["start-date", "end-date"].forEach((id) => elements[id].addEventListener("change", invalidateScenes));
  document.querySelectorAll(".layer-option").forEach((button) => button.addEventListener("click", () => { feedback(); chooseLayer(button.dataset.layer); }));
  elements["layer-opacity"].addEventListener("input", (event) => { setText("opacity-value", `${event.target.value}%`); state.tileLayer?.setOpacity(Number(event.target.value)/100); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && state.drawing) { stopDrawing(); event.preventDefault(); } });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && isActiveJob() && state.job.data.job_id) pollJob(state.job, true); });
  window.addEventListener("resize", () => map.invalidateSize());
  loadConfig();
})();
