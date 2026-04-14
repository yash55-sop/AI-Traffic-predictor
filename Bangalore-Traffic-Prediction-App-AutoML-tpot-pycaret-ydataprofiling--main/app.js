/* ============================================================
   RouteVista — Application Logic
   Lightweight Google Maps clone using OSM + Leaflet + OSRM
   ============================================================ */

// ─── Configuration ───────────────────────────────────────────
const CONFIG = {
    // Nominatim geocoding API (free, no key)
    NOMINATIM_URL: 'https://nominatim.openstreetmap.org',
    // OSRM public demo routing API (free, no key)
    OSRM_URL: 'https://router.project-osrm.org',
    // Default center (world center) if geolocation fails
    DEFAULT_CENTER: [20, 0],
    DEFAULT_ZOOM: 3,
    LOCATED_ZOOM: 14,
    ROUTE_ZOOM_PAD: 60,
    // Debounce delay for autocomplete (ms)
    DEBOUNCE_MS: 350,
    // User-Agent for Nominatim (required by usage policy)
    APP_NAME: 'RouteVista/1.0',
};

// ─── State ───────────────────────────────────────────────────
const state = {
    map: null,
    sourceLatLng: null,  // { lat, lng }
    destLatLng: null,
    sourceMarker: null,
    destMarker: null,
    routeLayer: null,     // Leaflet polyline layer group
    airLine: null,        // dashed straight line
    travelMode: 'driving', // driving | walking | cycling
    clickTarget: 'source', // which input gets the next map-click
    isCalculating: false,
};

// ─── DOM References ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const dom = {
    map: $('#map'),
    searchPanel: $('#searchPanel'),
    panelToggle: $('#panelToggle'),
    panelBody: $('#panelBody'),
    sourceInput: $('#sourceInput'),
    destInput: $('#destInput'),
    sourceSuggestions: $('#sourceSuggestions'),
    destSuggestions: $('#destSuggestions'),
    btnMyLocation: $('#btnMyLocation'),
    btnCalculate: $('#btnCalculate'),
    btnReset: $('#btnReset'),
    travelModes: $('#travelModes'),
    resultsCard: $('#resultsCard'),
    btnCloseResults: $('#btnCloseResults'),
    airDistance: $('#airDistance'),
    routeDistance: $('#routeDistance'),
    routeDistLabel: $('#routeDistLabel'),
    travelTime: $('#travelTime'),
    routeStepsWrapper: $('#routeStepsWrapper'),
    btnToggleSteps: $('#btnToggleSteps'),
    routeSteps: $('#routeSteps'),
    clickHint: $('#clickHint'),
    clickTarget: $('#clickTarget'),
    toastContainer: $('#toastContainer'),
    loaderOverlay: $('#loaderOverlay'),
    // AI Dashboard
    aiDashboard: $('#aiDashboard'),
    btnCloseAI: $('#btnCloseAI'),
    aiTodayDist: $('#aiTodayDist'),
    aiTodayTime: $('#aiTodayTime'),
    aiTodayCongestion: $('#aiTodayCongestion'),
    aiTodayRisk: $('#aiTodayRisk'),
    riskArcToday: $('#riskArcToday'),
    todayWeatherIcon: $('#todayWeatherIcon'),
    todayWeatherLabel: $('#todayWeatherLabel'),
    todayWeatherEffect: $('#todayWeatherEffect'),
    aiTodayFactors: $('#aiTodayFactors'),
    aiTodayIncidents: $('#aiTodayIncidents'),
    aiTodayIncidentList: $('#aiTodayIncidentList'),
    confFillToday: $('#confFillToday'),
    confPctToday: $('#confPctToday'),
    // Tomorrow
    tmrwDate: $('#tmrwDate'),
    aiTmrwDist: $('#aiTmrwDist'),
    aiTmrwTime: $('#aiTmrwTime'),
    aiTmrwCongestion: $('#aiTmrwCongestion'),
    aiTmrwRisk: $('#aiTmrwRisk'),
    riskArcTmrw: $('#riskArcTmrw'),
    tmrwWeatherIcon: $('#tmrwWeatherIcon'),
    tmrwWeatherLabel: $('#tmrwWeatherLabel'),
    tmrwWeatherEffect: $('#tmrwWeatherEffect'),
    aiTmrwFactors: $('#aiTmrwFactors'),
    aiComparison: $('#aiComparison'),
    aiComparisonText: $('#aiComparisonText'),
    confFillTmrw: $('#confFillTmrw'),
    confPctTmrw: $('#confPctTmrw'),
    // Insights
    aiBestTime: $('#aiBestTime'),
    aiBestTimeDesc: $('#aiBestTimeDesc'),
    aiStrategyList: $('#aiStrategyList'),
    aiWarningsCard: $('#aiWarningsCard'),
    aiWarningsList: $('#aiWarningsList'),
    confFillOverall: $('#confFillOverall'),
    confPctOverall: $('#confPctOverall'),
};

// ============================================================
//  1. MAP INITIALIZATION
// ============================================================

/**
 * Initializes the Leaflet map with a dark-themed tile layer.
 */
function initMap() {
    state.map = L.map('map', {
        zoomControl: true,
        attributionControl: true,
    }).setView(CONFIG.DEFAULT_CENTER, CONFIG.DEFAULT_ZOOM);

    // Dark tile layer (CartoDB Dark Matter) — free, no key
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    }).addTo(state.map);

    // Move zoom control to bottom-right
    state.map.zoomControl.setPosition('bottomright');

    // Map click handler
    state.map.on('click', handleMapClick);
}

/**
 * Attempts to center the map on the user's current location.
 */
function geolocateUser() {
    if (!navigator.geolocation) {
        showToast('Geolocation is not supported by your browser.', 'error');
        return;
    }

    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const { latitude, longitude } = pos.coords;
            state.map.setView([latitude, longitude], CONFIG.LOCATED_ZOOM);
            showToast('Map centered on your location.', 'success');
        },
        (err) => {
            const messages = {
                1: 'Location permission denied. You can still search manually.',
                2: 'Position unavailable — using default view.',
                3: 'Location request timed out.',
            };
            showToast(messages[err.code] || 'Could not get location.', 'info');
        },
        { enableHighAccuracy: true, timeout: 8000 }
    );
}

// ============================================================
//  2. CUSTOM MARKERS
// ============================================================

/**
 * Creates a custom Leaflet divIcon marker.
 * @param {'source'|'dest'} type
 * @returns {L.DivIcon}
 */
function createIcon(type) {
    return L.divIcon({
        className: 'custom-marker',
        html: `
      <div class="marker-pulse ${type}"></div>
      <div class="marker-pin ${type}"></div>
    `,
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -34],
    });
}

/**
 * Places or moves a marker on the map.
 * @param {'source'|'dest'} type
 * @param {number} lat
 * @param {number} lng
 * @param {string} [label]
 */
function setMarker(type, lat, lng, label) {
    const latlng = L.latLng(lat, lng);

    if (type === 'source') {
        if (state.sourceMarker) state.map.removeLayer(state.sourceMarker);
        state.sourceMarker = L.marker(latlng, { icon: createIcon('source') })
            .addTo(state.map)
            .bindPopup(`<b>Source</b><br>${label || `${lat.toFixed(5)}, ${lng.toFixed(5)}`}`);
        state.sourceLatLng = { lat, lng };
    } else {
        if (state.destMarker) state.map.removeLayer(state.destMarker);
        state.destMarker = L.marker(latlng, { icon: createIcon('dest') })
            .addTo(state.map)
            .bindPopup(`<b>Destination</b><br>${label || `${lat.toFixed(5)}, ${lng.toFixed(5)}`}`);
        state.destLatLng = { lat, lng };
    }
}

// ============================================================
//  3. GEOCODING (Nominatim)
// ============================================================

/**
 * Searches for places using Nominatim.
 * @param {string} query
 * @returns {Promise<Array>}
 */
async function geocodeSearch(query) {
    if (!query || query.trim().length < 2) return [];

    const url = `${CONFIG.NOMINATIM_URL}/search?` + new URLSearchParams({
        q: query,
        format: 'json',
        addressdetails: 1,
        limit: 6,
    });

    try {
        const res = await fetch(url, {
            headers: { 'Accept-Language': 'en', 'User-Agent': CONFIG.APP_NAME },
        });
        if (!res.ok) throw new Error(`Nominatim error: ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error('Geocoding failed:', err);
        showToast('Geocoding service unavailable. Try again.', 'error');
        return [];
    }
}

/**
 * Reverse-geocodes a lat/lng to a human-readable name.
 * @param {number} lat
 * @param {number} lng
 * @returns {Promise<string>}
 */
async function reverseGeocode(lat, lng) {
    const url = `${CONFIG.NOMINATIM_URL}/reverse?` + new URLSearchParams({
        lat, lon: lng, format: 'json',
    });

    try {
        const res = await fetch(url, {
            headers: { 'User-Agent': CONFIG.APP_NAME },
        });
        if (!res.ok) return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
        const data = await res.json();
        return data.display_name || `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    } catch {
        return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    }
}

// ============================================================
//  4. AUTOCOMPLETE
// ============================================================

let debounceTimers = { source: null, dest: null };

/**
 * Sets up autocomplete on an input element.
 * @param {HTMLInputElement} input
 * @param {HTMLUListElement} listEl
 * @param {'source'|'dest'} type
 */
function setupAutocomplete(input, listEl, type) {
    input.addEventListener('input', () => {
        clearTimeout(debounceTimers[type]);
        const query = input.value.trim();
        if (query.length < 2) {
            hideSuggestions(listEl);
            return;
        }

        debounceTimers[type] = setTimeout(async () => {
            const results = await geocodeSearch(query);
            renderSuggestions(results, listEl, input, type);
        }, CONFIG.DEBOUNCE_MS);
    });

    // Close suggestions on outside click
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !listEl.contains(e.target)) {
            hideSuggestions(listEl);
        }
    });

    // Keyboard navigation
    input.addEventListener('keydown', (e) => {
        handleSuggestionKeyboard(e, listEl, input, type);
    });
}

/**
 * Renders suggestion items in the dropdown.
 */
function renderSuggestions(results, listEl, input, type) {
    listEl.innerHTML = '';
    if (!results.length) {
        hideSuggestions(listEl);
        return;
    }

    results.forEach((place) => {
        const li = document.createElement('li');
        li.innerHTML = `
      <span class="sug-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C8 2 4 6 4 10c0 6 8 12 8 12s8-6 8-12c0-4-4-8-8-8z"/><circle cx="12" cy="10" r="3"/></svg>
      </span>
      <span class="sug-text">${place.display_name}</span>
    `;
        li.addEventListener('click', () => {
            selectSuggestion(place, input, listEl, type);
        });
        listEl.appendChild(li);
    });

    listEl.classList.add('visible');
}

function hideSuggestions(listEl) {
    listEl.classList.remove('visible');
    listEl.innerHTML = '';
}

/**
 * Handles selecting a suggestion.
 */
function selectSuggestion(place, input, listEl, type) {
    const lat = parseFloat(place.lat);
    const lng = parseFloat(place.lon);
    const label = place.display_name;

    input.value = label;
    hideSuggestions(listEl);
    setMarker(type, lat, lng, label);
    state.map.setView([lat, lng], CONFIG.LOCATED_ZOOM);
}

/**
 * Keyboard navigation for suggestion list.
 */
function handleSuggestionKeyboard(e, listEl, input, type) {
    const items = listEl.querySelectorAll('li');
    if (!items.length) return;

    let activeIdx = Array.from(items).findIndex((li) => li.classList.contains('active'));

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (activeIdx >= 0) items[activeIdx].classList.remove('active');
        activeIdx = (activeIdx + 1) % items.length;
        items[activeIdx].classList.add('active');
        items[activeIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (activeIdx >= 0) items[activeIdx].classList.remove('active');
        activeIdx = (activeIdx - 1 + items.length) % items.length;
        items[activeIdx].classList.add('active');
        items[activeIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && activeIdx >= 0) {
        e.preventDefault();
        items[activeIdx].click();
    } else if (e.key === 'Escape') {
        hideSuggestions(listEl);
    }
}

// ============================================================
//  5. MAP CLICK HANDLER
// ============================================================

/**
 * Handles click on the map to set source or destination.
 */
async function handleMapClick(e) {
    const { lat, lng } = e.latlng;
    const type = state.clickTarget;
    const input = type === 'source' ? dom.sourceInput : dom.destInput;

    // Place marker immediately
    setMarker(type, lat, lng);

    // Reverse-geocode for a name
    const name = await reverseGeocode(lat, lng);
    input.value = name;

    // Update marker popup with name
    const marker = type === 'source' ? state.sourceMarker : state.destMarker;
    if (marker) {
        marker.setPopupContent(`<b>${type === 'source' ? 'Source' : 'Destination'}</b><br>${name}`);
    }

    // Toggle click target for next click
    state.clickTarget = state.clickTarget === 'source' ? 'dest' : 'source';
    dom.clickTarget.textContent = state.clickTarget === 'source' ? 'source' : 'destination';
}

// ============================================================
//  6. DISTANCE CALCULATIONS
// ============================================================

/**
 * Haversine formula — calculates straight-line distance between two points.
 * @returns {number} Distance in kilometers
 */
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const toRad = (deg) => (deg * Math.PI) / 180;

    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
}

/**
 * Formats a distance value for display.
 */
function formatDistance(km) {
    if (km < 1) return `${(km * 1000).toFixed(0)} m`;
    if (km < 100) return `${km.toFixed(2)} km`;
    return `${km.toFixed(1)} km`;
}

/**
 * Formats seconds into human-readable travel time.
 */
function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)} sec`;
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.round((seconds % 3600) / 60);
    if (hrs === 0) return `${mins} min`;
    return `${hrs} hr ${mins} min`;
}

// ============================================================
//  7. ROUTING (OSRM)
// ============================================================

/**
 * Map internal travel mode names to OSRM profile names.
 */
const OSRM_PROFILES = {
    driving: 'car',
    walking: 'foot',
    cycling: 'bike',
};

/**
 * Fetches the driving/walking/cycling route from OSRM.
 * @returns {Promise<{distance: number, duration: number, geometry: Array, steps: Array}>}
 */
async function fetchRoute(srcLat, srcLng, dstLat, dstLng, mode = 'driving') {
    const profile = OSRM_PROFILES[mode] || 'car';
    const coords = `${srcLng},${srcLat};${dstLng},${dstLat}`;
    const url = `${CONFIG.OSRM_URL}/route/v1/${profile}/${coords}?overview=full&geometries=geojson&steps=true`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`OSRM error: ${res.status}`);

    const data = await res.json();
    if (data.code !== 'Ok' || !data.routes.length) {
        throw new Error(data.message || 'No route found.');
    }

    const route = data.routes[0];
    const geometry = route.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    const steps = route.legs[0].steps.map((s) => ({
        instruction: s.maneuver.type.replace(/_/g, ' '),
        name: s.name || '—',
        distance: s.distance,
        duration: s.duration,
    }));

    return {
        distance: route.distance / 1000, // meters → km
        duration: route.duration,          // seconds
        geometry,
        steps,
    };
}

// ============================================================
//  8. ROUTE DRAWING
// ============================================================

/**
 * Clears any existing route lines from the map.
 */
function clearRouteLines() {
    if (state.routeLayer) {
        state.map.removeLayer(state.routeLayer);
        state.routeLayer = null;
    }
    if (state.airLine) {
        state.map.removeLayer(state.airLine);
        state.airLine = null;
    }
}

/**
 * Draws the driving route and a dashed air-distance line.
 */
function drawRoute(geometry, srcLatLng, dstLatLng) {
    clearRouteLines();

    // Animated gradient route line
    state.routeLayer = L.polyline(geometry, {
        color: '#4f8cff',
        weight: 5,
        opacity: 0.85,
        lineCap: 'round',
        lineJoin: 'round',
        className: 'animated-route',
    }).addTo(state.map);

    // Dashed air-distance line
    state.airLine = L.polyline(
        [[srcLatLng.lat, srcLatLng.lng], [dstLatLng.lat, dstLatLng.lng]],
        {
            color: '#a855f7',
            weight: 2,
            opacity: 0.5,
            dashArray: '8, 10',
        }
    ).addTo(state.map);

    // Fit map to route bounds
    const bounds = state.routeLayer.getBounds().pad(0.15);
    state.map.fitBounds(bounds, { paddingTopLeft: [400, 20] });
}

// ============================================================
//  9. MAIN CALCULATE HANDLER
// ============================================================

async function calculateRoute() {
    // Validate
    if (!state.sourceLatLng || !state.destLatLng) {
        showToast('Please set both a source and destination.', 'error');
        return;
    }
    if (state.isCalculating) return;

    state.isCalculating = true;
    dom.loaderOverlay.classList.remove('hidden');

    const { lat: sLat, lng: sLng } = state.sourceLatLng;
    const { lat: dLat, lng: dLng } = state.destLatLng;

    // 1) Air distance (instant)
    const airKm = haversineDistance(sLat, sLng, dLat, dLng);
    dom.airDistance.textContent = formatDistance(airKm);

    // Store actual OSRM values for AI engine
    let osrmDistanceKm = airKm;
    let osrmDurationSec = airKm * 60; // rough fallback

    // 2) Route via OSRM
    try {
        const route = await fetchRoute(sLat, sLng, dLat, dLng, state.travelMode);

        osrmDistanceKm = route.distance;
        osrmDurationSec = route.duration;

        dom.routeDistance.textContent = formatDistance(route.distance);
        dom.travelTime.textContent = formatDuration(route.duration);

        // Update label based on travel mode
        const modeLabels = { driving: 'Driving Distance', walking: 'Walking Distance', cycling: 'Cycling Distance' };
        dom.routeDistLabel.textContent = modeLabels[state.travelMode] || 'Route Distance';

        // Draw the route
        drawRoute(route.geometry, state.sourceLatLng, state.destLatLng);

        // Populate turn-by-turn steps
        if (route.steps.length > 0) {
            dom.routeStepsWrapper.classList.remove('hidden');
            dom.routeSteps.innerHTML = route.steps
                .filter((s) => s.instruction !== 'arrive')
                .map((s) => `<li><strong>${capitalize(s.instruction)}</strong> on ${s.name} — ${formatDistance(s.distance / 1000)}</li>`)
                .join('');
        }

        showToast('Route calculated successfully!', 'success');
    } catch (err) {
        console.error('Route error:', err);
        dom.routeDistance.textContent = 'N/A';
        dom.travelTime.textContent = 'N/A';
        // Still draw the air line
        clearRouteLines();
        state.airLine = L.polyline(
            [[sLat, sLng], [dLat, dLng]],
            { color: '#a855f7', weight: 2, opacity: 0.5, dashArray: '8, 10' }
        ).addTo(state.map);
        state.map.fitBounds(state.airLine.getBounds().pad(0.2));
        showToast(`Routing failed: ${err.message}`, 'error');
    }

    // Show results card
    dom.resultsCard.classList.remove('hidden');
    // Force re-animation
    dom.resultsCard.style.animation = 'none';
    // Trigger reflow
    void dom.resultsCard.offsetHeight;
    dom.resultsCard.style.animation = '';

    // ── Run AI Traffic Intelligence Engine ──
    try {
        const prediction = await TrafficIntelligenceEngine.predict({
            source: state.sourceLatLng,
            destination: state.destLatLng,
            routeDistanceKm: osrmDistanceKm,
            routeDurationSec: osrmDurationSec,
            airDistanceKm: airKm,
            travelMode: state.travelMode,
            dateTime: new Date(),
        });
        renderAIDashboard(prediction);
    } catch (aiErr) {
        console.warn('AI prediction failed:', aiErr);
    }

    state.isCalculating = false;
    dom.loaderOverlay.classList.add('hidden');
}

// ============================================================
//  10. RESET
// ============================================================

function resetAll() {
    // Clear markers
    if (state.sourceMarker) { state.map.removeLayer(state.sourceMarker); state.sourceMarker = null; }
    if (state.destMarker) { state.map.removeLayer(state.destMarker); state.destMarker = null; }
    state.sourceLatLng = null;
    state.destLatLng = null;

    // Clear route
    clearRouteLines();

    // Clear inputs
    dom.sourceInput.value = '';
    dom.destInput.value = '';

    // Hide results
    dom.resultsCard.classList.add('hidden');
    dom.routeStepsWrapper.classList.add('hidden');
    dom.routeSteps.classList.add('hidden');
    dom.btnToggleSteps.classList.remove('open');
    dom.airDistance.textContent = '—';
    dom.routeDistance.textContent = '—';
    dom.travelTime.textContent = '—';

    // Hide AI dashboard
    dom.aiDashboard.classList.add('hidden');

    // Reset click target
    state.clickTarget = 'source';
    dom.clickTarget.textContent = 'source';

    showToast('Map cleared.', 'info');
}

// ============================================================
//  11. USE CURRENT LOCATION (for Source)
// ============================================================

function useCurrentLocationAsSource() {
    if (!navigator.geolocation) {
        showToast('Geolocation not supported.', 'error');
        return;
    }

    showToast('Fetching your location…', 'info');

    navigator.geolocation.getCurrentPosition(
        async (pos) => {
            const { latitude: lat, longitude: lng } = pos.coords;
            setMarker('source', lat, lng);
            state.map.setView([lat, lng], CONFIG.LOCATED_ZOOM);

            // Reverse-geocode for the input
            const name = await reverseGeocode(lat, lng);
            dom.sourceInput.value = name;
            if (state.sourceMarker) {
                state.sourceMarker.setPopupContent(`<b>Source</b><br>${name}`);
            }

            // Next click sets destination
            state.clickTarget = 'dest';
            dom.clickTarget.textContent = 'destination';

            showToast('Source set to your location.', 'success');
        },
        (err) => {
            const msgs = { 1: 'Permission denied.', 2: 'Position unavailable.', 3: 'Timed out.' };
            showToast(msgs[err.code] || 'Could not get location.', 'error');
        },
        { enableHighAccuracy: true, timeout: 8000 }
    );
}

// ============================================================
//  12. TOAST NOTIFICATIONS
// ============================================================

/**
 * Shows a small notification toast.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 */
function showToast(message, type = 'info') {
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
    <span class="toast-icon">${icons[type]}</span>
    <span>${message}</span>
  `;
    dom.toastContainer.appendChild(toast);

    // Auto-remove after 4s
    setTimeout(() => {
        toast.classList.add('removing');
        toast.addEventListener('animationend', () => toast.remove());
    }, 4000);
}

// ============================================================
//  13. UTILITY
// ============================================================

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Parses a formatted duration string back to seconds (rough).
 * E.g., "2 hr 29 min" → 8940, "45 min" → 2700
 */
function parseDurationToSeconds(str) {
    let total = 0;
    const hrMatch = str.match(/(\d+)\s*hr/);
    const minMatch = str.match(/(\d+)\s*min/);
    const secMatch = str.match(/(\d+)\s*sec/);
    if (hrMatch) total += parseInt(hrMatch[1]) * 3600;
    if (minMatch) total += parseInt(minMatch[1]) * 60;
    if (secMatch) total += parseInt(secMatch[1]);
    return total || 3600; // fallback 1 hour
}

// ============================================================
//  13b. AI DASHBOARD RENDERER
// ============================================================

/**
 * Renders the AI Traffic Intelligence Dashboard with prediction data.
 * @param {Object} prediction — output from TrafficIntelligenceEngine.predict()
 */
function renderAIDashboard(prediction) {
    const { today, tomorrow, insights, confidence } = prediction;

    // Show the dashboard
    dom.aiDashboard.classList.remove('hidden');
    dom.aiDashboard.style.animation = 'none';
    void dom.aiDashboard.offsetHeight;
    dom.aiDashboard.style.animation = '';

    // ── TODAY ────────────────────────────────────────────────
    dom.aiTodayDist.textContent = formatDistance(today.distance);
    dom.aiTodayTime.textContent = today.travelTimeFormatted;
    dom.aiTodayCongestion.innerHTML = `${today.congestionInfo.emoji} ${today.congestionInfo.level}`;
    dom.aiTodayCongestion.style.color = today.congestionInfo.color;

    // Risk gauge animation
    animateRiskGauge(dom.riskArcToday, dom.aiTodayRisk, today.riskScore);

    // Weather
    dom.todayWeatherIcon.textContent = today.weather.icon;
    dom.todayWeatherLabel.textContent = today.weather.label;
    dom.todayWeatherEffect.textContent = today.weather.speedFactor < 0.9
        ? `Speed −${Math.round((1 - today.weather.speedFactor) * 100)}%`
        : '';

    // Factors
    dom.aiTodayFactors.innerHTML = today.factors
        .map(f => `<li>${f}</li>`).join('');

    // Incidents
    if (today.incidents.length > 0) {
        dom.aiTodayIncidents.classList.remove('hidden');
        dom.aiTodayIncidentList.innerHTML = today.incidents
            .map(i => `<li>🚧 ${i.type} — +${i.delay} min delay</li>`).join('');
    } else {
        dom.aiTodayIncidents.classList.add('hidden');
    }

    // Confidence
    animateConfidence(dom.confFillToday, dom.confPctToday, confidence.today);

    // ── TOMORROW ────────────────────────────────────────────
    dom.tmrwDate.textContent = tomorrow.date;
    dom.aiTmrwDist.textContent = formatDistance(tomorrow.distance);
    dom.aiTmrwTime.textContent = tomorrow.travelTimeFormatted;
    dom.aiTmrwCongestion.innerHTML = `${tomorrow.congestionInfo.emoji} ${tomorrow.congestionInfo.level}`;
    dom.aiTmrwCongestion.style.color = tomorrow.congestionInfo.color;

    animateRiskGauge(dom.riskArcTmrw, dom.aiTmrwRisk, tomorrow.riskScore);

    dom.tmrwWeatherIcon.textContent = tomorrow.weather.icon;
    dom.tmrwWeatherLabel.textContent = tomorrow.weather.label;
    dom.tmrwWeatherEffect.textContent = tomorrow.weather.speedFactor < 0.9
        ? `Speed −${Math.round((1 - tomorrow.weather.speedFactor) * 100)}%`
        : '';

    dom.aiTmrwFactors.innerHTML = tomorrow.factors
        .map(f => `<li>${f}</li>`).join('');

    // Comparison
    dom.aiComparisonText.textContent = insights.comparison.message;

    animateConfidence(dom.confFillTmrw, dom.confPctTmrw, confidence.tomorrow);

    // ── INSIGHTS ────────────────────────────────────────────
    dom.aiBestTime.textContent = insights.bestTimeToTravel.formatted;
    dom.aiBestTimeDesc.textContent = `Lowest predicted congestion window for this route today.`;

    dom.aiStrategyList.innerHTML = insights.routeStrategy
        .map(s => `<li>${s}</li>`).join('');

    // Warnings
    if (insights.warnings.length > 0) {
        dom.aiWarningsCard.classList.remove('hidden');
        dom.aiWarningsList.innerHTML = insights.warnings
            .map(w => `<li class="${w.level}">${w.message}</li>`).join('');
    } else {
        dom.aiWarningsCard.classList.add('hidden');
    }

    animateConfidence(dom.confFillOverall, dom.confPctOverall, confidence.overall);
}

/**
 * Animates the SVG risk arc gauge.
 */
function animateRiskGauge(arcEl, numberEl, score) {
    const maxDash = 157; // approximate arc length
    const targetDash = (score / 100) * maxDash;

    // Color based on score
    let color = '#34d399'; // green
    if (score >= 70) color = '#ef4444'; // red
    else if (score >= 45) color = '#f97316'; // orange
    else if (score >= 25) color = '#fbbf24'; // yellow

    arcEl.style.stroke = color;
    numberEl.style.color = color;

    // Animate
    let current = 0;
    const step = () => {
        current += (targetDash - current) * 0.12;
        if (Math.abs(current - targetDash) < 0.5) current = targetDash;
        arcEl.setAttribute('stroke-dasharray', `${current} ${maxDash}`);
        numberEl.textContent = Math.round((current / maxDash) * 100);
        if (current < targetDash) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

/**
 * Animates a confidence bar fill.
 */
function animateConfidence(fillEl, pctEl, value) {
    setTimeout(() => {
        fillEl.style.width = `${value}%`;
        pctEl.textContent = `${value}%`;
    }, 200);
}

// ============================================================
//  14. EVENT LISTENERS
// ============================================================

function bindEvents() {
    // Panel toggle (mobile)
    dom.panelToggle.addEventListener('click', () => {
        dom.panelBody.classList.toggle('collapsed');
    });

    // Autocomplete
    setupAutocomplete(dom.sourceInput, dom.sourceSuggestions, 'source');
    setupAutocomplete(dom.destInput, dom.destSuggestions, 'dest');

    // My Location
    dom.btnMyLocation.addEventListener('click', useCurrentLocationAsSource);

    // Calculate
    dom.btnCalculate.addEventListener('click', calculateRoute);

    // Reset
    dom.btnReset.addEventListener('click', resetAll);

    // Close results
    dom.btnCloseResults.addEventListener('click', () => {
        dom.resultsCard.classList.add('hidden');
    });

    // Close AI dashboard
    dom.btnCloseAI.addEventListener('click', () => {
        dom.aiDashboard.classList.add('hidden');
    });

    // AI Dashboard tab switching
    document.querySelectorAll('.ai-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.ai-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.ai-tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const tabName = tab.dataset.tab;
            const mapping = { today: 'tabToday', tomorrow: 'tabTomorrow', insights: 'tabInsights' };
            document.getElementById(mapping[tabName]).classList.add('active');
        });
    });

    // Travel mode buttons
    dom.travelModes.querySelectorAll('.mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            dom.travelModes.querySelector('.active').classList.remove('active');
            btn.classList.add('active');
            state.travelMode = btn.dataset.mode;

            // Re-calculate if we already have a route
            if (state.sourceLatLng && state.destLatLng && !dom.resultsCard.classList.contains('hidden')) {
                calculateRoute();
            }
        });
    });

    // Toggle turn-by-turn steps
    dom.btnToggleSteps.addEventListener('click', () => {
        dom.routeSteps.classList.toggle('hidden');
        dom.btnToggleSteps.classList.toggle('open');
    });

    // Show click hint on map hover
    state.map.on('mousemove', () => {
        // Only show hint when at least one point is missing
        if (!state.sourceLatLng || !state.destLatLng) {
            dom.clickHint.classList.add('visible');
        }
    });
    state.map.on('mouseout', () => {
        dom.clickHint.classList.remove('visible');
    });

    // Focus management — when user focuses an input, set click target accordingly
    dom.sourceInput.addEventListener('focus', () => {
        state.clickTarget = 'source';
        dom.clickTarget.textContent = 'source';
    });
    dom.destInput.addEventListener('focus', () => {
        state.clickTarget = 'dest';
        dom.clickTarget.textContent = 'destination';
    });

    // Keyboard shortcut: Enter in dest input triggers calculate
    dom.destInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && dom.destSuggestions.querySelectorAll('li').length === 0) {
            calculateRoute();
        }
    });
}

// ============================================================
//  15. BOOTSTRAP
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    geolocateUser();
    bindEvents();
});
