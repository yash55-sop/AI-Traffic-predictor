/* ============================================================
   RouteVista — AI Traffic Intelligence Engine
   Multi-factor traffic prediction system
   ============================================================ */

/**
 * TrafficIntelligenceEngine
 * 
 * Simulates a real-world AI traffic prediction system by analyzing:
 *  - Time of day & day of week patterns
 *  - Weather conditions (seasonal + regional simulation)
 *  - Public events & festivals (date-aware)
 *  - Historical congestion patterns
 *  - Road incident probability
 *  - Distance-based scaling
 * 
 * All predictions are deterministic for the same inputs,
 * producing realistic, structured dashboard data.
 */
const TrafficIntelligenceEngine = (() => {

    // ─── Constants ───────────────────────────────────────────

    /** Rush-hour congestion multipliers by hour (0-23) */
    const HOURLY_CONGESTION = [
        0.15, 0.10, 0.08, 0.08, 0.10, 0.18, // 00–05: very low (night)
        0.35, 0.65, 0.90, 0.78, 0.55, 0.50, // 06–11: morning rush peaks at 08
        0.55, 0.52, 0.48, 0.50, 0.60, 0.85, // 12–17: afternoon, evening rush starts
        0.92, 0.75, 0.55, 0.40, 0.30, 0.20, // 18–23: evening rush peaks at 18, then drops
    ];

    /** Day-of-week multipliers (0=Sun, 6=Sat) */
    const DAY_MULTIPLIERS = [0.55, 1.0, 0.95, 0.92, 0.95, 1.05, 0.65];

    /** Weather types and their impact on traffic */
    const WEATHER_PROFILES = {
        clear: { label: 'Clear Sky', icon: '☀️', congestionAdd: 0.0, riskAdd: 0, speedFactor: 1.00 },
        partly: { label: 'Partly Cloudy', icon: '⛅', congestionAdd: 0.02, riskAdd: 3, speedFactor: 0.98 },
        cloudy: { label: 'Overcast', icon: '☁️', congestionAdd: 0.05, riskAdd: 5, speedFactor: 0.96 },
        light_rain: { label: 'Light Rain', icon: '🌦️', congestionAdd: 0.12, riskAdd: 15, speedFactor: 0.88 },
        heavy_rain: { label: 'Heavy Rain', icon: '🌧️', congestionAdd: 0.25, riskAdd: 30, speedFactor: 0.72 },
        thunderstorm: { label: 'Thunderstorm', icon: '⛈️', congestionAdd: 0.35, riskAdd: 45, speedFactor: 0.60 },
        fog: { label: 'Dense Fog', icon: '🌫️', congestionAdd: 0.22, riskAdd: 35, speedFactor: 0.65 },
        snow: { label: 'Snow', icon: '❄️', congestionAdd: 0.30, riskAdd: 40, speedFactor: 0.55 },
        haze: { label: 'Haze / Smog', icon: '😶‍🌫️', congestionAdd: 0.10, riskAdd: 12, speedFactor: 0.90 },
        hot: { label: 'Extreme Heat', icon: '🔥', congestionAdd: 0.05, riskAdd: 8, speedFactor: 0.95 },
    };

    /** Major Indian & global festivals / events (month-day) */
    const KNOWN_EVENTS = {
        '01-01': { name: 'New Year\'s Day', impact: 0.30 },
        '01-14': { name: 'Makar Sankranti / Pongal', impact: 0.25 },
        '01-26': { name: 'Republic Day', impact: 0.35 },
        '02-14': { name: 'Valentine\'s Day', impact: 0.10 },
        '03-08': { name: 'Holi (approx.)', impact: 0.35 },
        '03-29': { name: 'Holi (2026)', impact: 0.40 },
        '04-06': { name: 'Ugadi / Gudi Padwa', impact: 0.20 },
        '04-10': { name: 'Good Friday', impact: 0.15 },
        '04-13': { name: 'Baisakhi', impact: 0.25 },
        '04-14': { name: 'Ambedkar Jayanti', impact: 0.20 },
        '05-01': { name: 'Labour Day', impact: 0.15 },
        '06-17': { name: 'Eid al-Adha (approx.)', impact: 0.30 },
        '07-17': { name: 'Muharram (approx.)', impact: 0.20 },
        '08-15': { name: 'Independence Day', impact: 0.40 },
        '08-16': { name: 'Janmashtami (approx.)', impact: 0.25 },
        '09-05': { name: 'Teachers\' Day', impact: 0.10 },
        '10-02': { name: 'Gandhi Jayanti', impact: 0.30 },
        '10-12': { name: 'Dussehra (approx.)', impact: 0.35 },
        '10-20': { name: 'Diwali (approx.)', impact: 0.45 },
        '11-01': { name: 'Diwali Holiday Period', impact: 0.35 },
        '11-14': { name: 'Children\'s Day', impact: 0.10 },
        '12-25': { name: 'Christmas', impact: 0.30 },
        '12-31': { name: 'New Year\'s Eve', impact: 0.35 },
    };

    // ─── Helper Functions ────────────────────────────────────

    /**
     * Simple seeded pseudo-random number generator (for deterministic results).
     * Uses a hash of the input string as the seed.
     */
    function seededRandom(seed) {
        let hash = 0;
        for (let i = 0; i < seed.length; i++) {
            const char = seed.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        // Normalize to 0-1
        return Math.abs(Math.sin(hash) * 10000) % 1;
    }

    /**
     * Determines the simulated weather based on month, hour, and location seed.
     */
    function predictWeather(date, lat, lng) {
        const month = date.getMonth(); // 0-11
        const hour = date.getHours();
        const seed = `weather-${lat.toFixed(1)}-${lng.toFixed(1)}-${date.toDateString()}`;
        const rand = seededRandom(seed);

        // Seasonal weather probabilities (Northern Hemisphere / India bias)
        // Summer (Apr-Jun): hot, haze, thunderstorm
        // Monsoon (Jul-Sep): heavy rain, thunderstorm
        // Winter (Nov-Feb): fog, clear, cold
        // Spring/Autumn: pleasant

        let weatherKey = 'clear';

        if (month >= 6 && month <= 8) {
            // Monsoon season
            if (rand < 0.30) weatherKey = 'heavy_rain';
            else if (rand < 0.55) weatherKey = 'light_rain';
            else if (rand < 0.70) weatherKey = 'thunderstorm';
            else if (rand < 0.85) weatherKey = 'cloudy';
            else weatherKey = 'partly';
        } else if (month >= 3 && month <= 5) {
            // Summer
            if (rand < 0.25) weatherKey = 'hot';
            else if (rand < 0.40) weatherKey = 'haze';
            else if (rand < 0.55) weatherKey = 'partly';
            else if (rand < 0.70) weatherKey = 'clear';
            else if (rand < 0.85) weatherKey = 'light_rain';
            else weatherKey = 'thunderstorm';
        } else if (month >= 10 || month <= 1) {
            // Winter
            if (hour >= 4 && hour <= 9 && rand < 0.35) weatherKey = 'fog';
            else if (rand < 0.30) weatherKey = 'clear';
            else if (rand < 0.55) weatherKey = 'partly';
            else if (rand < 0.70) weatherKey = 'haze';
            else if (rand < 0.80) weatherKey = 'cloudy';
            else weatherKey = 'light_rain';

            // High latitude → snow possibility
            if (Math.abs(lat) > 30 && rand < 0.15) weatherKey = 'snow';
        } else {
            // Autumn / Spring
            if (rand < 0.35) weatherKey = 'clear';
            else if (rand < 0.55) weatherKey = 'partly';
            else if (rand < 0.70) weatherKey = 'cloudy';
            else if (rand < 0.85) weatherKey = 'light_rain';
            else weatherKey = 'haze';
        }

        return { key: weatherKey, ...WEATHER_PROFILES[weatherKey] };
    }

    /**
     * Checks if any known events fall on the given date (±1 day window).
     */
    function detectEvents(date) {
        const events = [];
        const pad = (n) => String(n).padStart(2, '0');

        for (let offset = -1; offset <= 1; offset++) {
            const d = new Date(date);
            d.setDate(d.getDate() + offset);
            const key = `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

            if (KNOWN_EVENTS[key]) {
                const impactMod = offset === 0 ? 1.0 : 0.5; // adjacent days have half impact
                events.push({
                    ...KNOWN_EVENTS[key],
                    date: d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
                    adjustedImpact: KNOWN_EVENTS[key].impact * impactMod,
                    isToday: offset === 0,
                });
            }
        }

        return events;
    }

    /**
     * Simulates road incidents based on congestion level, weather, and time.
     */
    function simulateIncidents(congestionBase, weather, date) {
        const seed = `incident-${date.toISOString().slice(0, 13)}`;
        const rand = seededRandom(seed);

        const incidents = [];
        const incidentProb = congestionBase * 0.4 + (weather.riskAdd / 100) * 0.3 + rand * 0.1;

        if (incidentProb > 0.5) {
            incidents.push({ type: 'Minor Accident', delay: 8, severity: 'medium' });
        }
        if (incidentProb > 0.7) {
            incidents.push({ type: 'Road Construction', delay: 12, severity: 'low' });
        }
        if (incidentProb > 0.85) {
            incidents.push({ type: 'Major Accident', delay: 25, severity: 'high' });
        }
        if (weather.key === 'heavy_rain' || weather.key === 'thunderstorm') {
            incidents.push({ type: 'Waterlogging', delay: 15, severity: 'medium' });
        }
        if (weather.key === 'fog') {
            incidents.push({ type: 'Low Visibility Zone', delay: 10, severity: 'medium' });
        }

        return incidents;
    }

    /**
     * Computes the congestion level label from a 0-1 score.
     */
    function congestionLabel(score) {
        if (score < 0.30) return { level: 'Low', color: '#34d399', emoji: '🟢' };
        if (score < 0.60) return { level: 'Medium', color: '#fbbf24', emoji: '🟡' };
        if (score < 0.80) return { level: 'High', color: '#f97316', emoji: '🟠' };
        return { level: 'Severe', color: '#ef4444', emoji: '🔴' };
    }

    /**
     * Finds the best travel window for the day (lowest congestion hour).
     */
    function findBestTravelTime(date, lat, lng, events) {
        let bestHour = 0;
        let bestScore = Infinity;

        for (let h = 5; h <= 23; h++) {
            const testDate = new Date(date);
            testDate.setHours(h, 0, 0, 0);
            const hourCongestion = HOURLY_CONGESTION[h];
            const dayMul = DAY_MULTIPLIERS[testDate.getDay()];
            const weather = predictWeather(testDate, lat, lng);
            const eventImpact = events.reduce((sum, e) => sum + e.adjustedImpact, 0);
            const score = hourCongestion * dayMul + weather.congestionAdd + eventImpact * 0.3;

            if (score < bestScore) {
                bestScore = score;
                bestHour = h;
            }
        }

        return {
            hour: bestHour,
            label: `${bestHour}:00`,
            formatted: `${bestHour > 12 ? bestHour - 12 : bestHour}:00 ${bestHour >= 12 ? 'PM' : 'AM'}`,
            score: bestScore,
        };
    }

    // ─── Main Prediction Function ────────────────────────────

    /**
     * Generates a full traffic intelligence prediction.
     * 
     * @param {Object} params
     * @param {Object} params.source      — { lat, lng }
     * @param {Object} params.destination  — { lat, lng }
     * @param {number} params.routeDistanceKm — driving distance from OSRM
     * @param {number} params.routeDurationSec — driving duration from OSRM
     * @param {number} params.airDistanceKm — Haversine distance
     * @param {string} params.travelMode — 'driving' | 'walking' | 'cycling'
     * @param {Date}   [params.dateTime]  — defaults to now
     * @returns {Object} Structured prediction data
     */
    function predict(params) {
        const {
            source,
            destination,
            routeDistanceKm,
            routeDurationSec,
            airDistanceKm,
            travelMode = 'driving',
            dateTime = new Date(),
        } = params;

        const now = new Date(dateTime);
        const tomorrow = new Date(now);
        tomorrow.setDate(tomorrow.getDate() + 1);

        const midLat = (source.lat + destination.lat) / 2;
        const midLng = (source.lng + destination.lng) / 2;

        // ── TODAY'S PREDICTION ────────────────────────────────
        const todayWeather = predictWeather(now, midLat, midLng);
        const todayEvents = detectEvents(now);
        const todayHour = now.getHours();
        const todayDay = now.getDay();

        // Base congestion from time + day
        const todayBaseCongestion = HOURLY_CONGESTION[todayHour] * DAY_MULTIPLIERS[todayDay];

        // Event impact
        const todayEventImpact = todayEvents.reduce((sum, e) => sum + e.adjustedImpact, 0);

        // Total congestion score (0-1)
        let todayCongestion = Math.min(1,
            todayBaseCongestion + todayWeather.congestionAdd + todayEventImpact * 0.5
        );

        // Incidents
        const todayIncidents = simulateIncidents(todayCongestion, todayWeather, now);
        const incidentDelay = todayIncidents.reduce((sum, i) => sum + i.delay, 0);

        // Adjusted travel time (seconds)
        const todaySpeedAdjust = todayWeather.speedFactor * (1 - todayCongestion * 0.35);
        const todayTravelTimeSec = (routeDurationSec / todaySpeedAdjust) + (incidentDelay * 60);

        // Risk score (0-100)
        const todayRisk = Math.min(100, Math.round(
            todayCongestion * 40 +
            todayWeather.riskAdd +
            todayEventImpact * 15 +
            todayIncidents.length * 8
        ));

        // Key factors
        const todayFactors = [];
        if (todayHour >= 7 && todayHour <= 10) todayFactors.push('🏢 Morning Rush Hour');
        if (todayHour >= 17 && todayHour <= 20) todayFactors.push('🏢 Evening Rush Hour');
        if (todayWeather.key !== 'clear' && todayWeather.key !== 'partly') {
            todayFactors.push(`${todayWeather.icon} ${todayWeather.label}`);
        }
        todayEvents.forEach(e => todayFactors.push(`🎉 ${e.name}`));
        todayIncidents.forEach(i => todayFactors.push(`⚠️ ${i.type}`));
        if (todayDay === 0 || todayDay === 6) todayFactors.push('📅 Weekend');
        if (todayFactors.length === 0) todayFactors.push('✅ Normal Conditions');

        const todayPrediction = {
            distance: routeDistanceKm,
            airDistance: airDistanceKm,
            travelTime: todayTravelTimeSec,
            travelTimeFormatted: formatDurationFull(todayTravelTimeSec),
            congestion: todayCongestion,
            congestionInfo: congestionLabel(todayCongestion),
            riskScore: todayRisk,
            weather: todayWeather,
            events: todayEvents,
            incidents: todayIncidents,
            factors: todayFactors,
        };

        // ── TOMORROW'S PREDICTION ─────────────────────────────
        const tmrwWeather = predictWeather(tomorrow, midLat, midLng);
        const tmrwEvents = detectEvents(tomorrow);
        const tmrwDay = tomorrow.getDay();
        const tmrwBaseCongestion = HOURLY_CONGESTION[todayHour] * DAY_MULTIPLIERS[tmrwDay];
        const tmrwEventImpact = tmrwEvents.reduce((sum, e) => sum + e.adjustedImpact, 0);

        let tmrwCongestion = Math.min(1,
            tmrwBaseCongestion + tmrwWeather.congestionAdd + tmrwEventImpact * 0.5
        );

        const tmrwIncidents = simulateIncidents(tmrwCongestion, tmrwWeather, tomorrow);
        const tmrwIncidentDelay = tmrwIncidents.reduce((sum, i) => sum + i.delay, 0);
        const tmrwSpeedAdjust = tmrwWeather.speedFactor * (1 - tmrwCongestion * 0.35);
        const tmrwTravelTimeSec = (routeDurationSec / tmrwSpeedAdjust) + (tmrwIncidentDelay * 60);

        const tmrwRisk = Math.min(100, Math.round(
            tmrwCongestion * 40 +
            tmrwWeather.riskAdd +
            tmrwEventImpact * 15 +
            tmrwIncidents.length * 8
        ));

        const tmrwFactors = [];
        if (todayHour >= 7 && todayHour <= 10) tmrwFactors.push('🏢 Morning Rush Hour');
        if (todayHour >= 17 && todayHour <= 20) tmrwFactors.push('🏢 Evening Rush Hour');
        if (tmrwWeather.key !== 'clear' && tmrwWeather.key !== 'partly') {
            tmrwFactors.push(`${tmrwWeather.icon} ${tmrwWeather.label}`);
        }
        tmrwEvents.forEach(e => tmrwFactors.push(`🎉 ${e.name}`));
        if (tmrwDay === 0 || tmrwDay === 6) tmrwFactors.push('📅 Weekend');
        if (tmrwFactors.length === 0) tmrwFactors.push('✅ Normal Conditions');

        const tomorrowPrediction = {
            date: tomorrow.toLocaleDateString('en-IN', { weekday: 'long', month: 'short', day: 'numeric' }),
            distance: routeDistanceKm,
            travelTime: tmrwTravelTimeSec,
            travelTimeFormatted: formatDurationFull(tmrwTravelTimeSec),
            congestion: tmrwCongestion,
            congestionInfo: congestionLabel(tmrwCongestion),
            riskScore: tmrwRisk,
            weather: tmrwWeather,
            events: tmrwEvents,
            factors: tmrwFactors,
        };

        // ── AI INSIGHTS ───────────────────────────────────────
        const bestTime = findBestTravelTime(now, midLat, midLng, todayEvents);

        const insights = {
            bestTimeToTravel: bestTime,
            routeStrategy: generateRouteStrategy(todayCongestion, todayWeather, todayIncidents, travelMode),
            warnings: generateWarnings(todayPrediction, tomorrowPrediction),
            comparison: generateComparison(todayPrediction, tomorrowPrediction),
        };

        // ── CONFIDENCE SCORE ──────────────────────────────────
        // Higher for today (more data), lower for tomorrow (forecast uncertainty)
        const todayConfidence = Math.min(95, Math.round(82 + seededRandom(`conf-${now.toDateString()}`) * 13));
        const tomorrowConfidence = Math.min(88, Math.round(65 + seededRandom(`conf-${tomorrow.toDateString()}`) * 18));

        return {
            today: todayPrediction,
            tomorrow: tomorrowPrediction,
            insights,
            confidence: {
                today: todayConfidence,
                tomorrow: tomorrowConfidence,
                overall: Math.round((todayConfidence + tomorrowConfidence) / 2),
            },
            meta: {
                generatedAt: now.toISOString(),
                travelMode,
                source,
                destination,
            },
        };
    }

    // ─── Strategy & Warnings ─────────────────────────────────

    function generateRouteStrategy(congestion, weather, incidents, mode) {
        const strategies = [];

        if (congestion > 0.7) {
            strategies.push('Consider alternative routes via smaller roads to avoid highway congestion.');
        }
        if (weather.key === 'heavy_rain' || weather.key === 'thunderstorm') {
            strategies.push('Drive at reduced speed. Keep headlights on and maintain safe following distance.');
        }
        if (weather.key === 'fog') {
            strategies.push('Use fog lights. Avoid overtaking. Drive well below the speed limit.');
        }
        if (incidents.some(i => i.severity === 'high')) {
            strategies.push('Major incident reported. Expect significant delays or consider postponing travel.');
        }
        if (mode === 'walking' && (weather.key === 'heavy_rain' || weather.key === 'hot')) {
            strategies.push('Walking conditions are poor. Consider using public transit or postponing.');
        }
        if (mode === 'cycling' && weather.riskAdd > 20) {
            strategies.push('Weather is unfavorable for cycling. Consider an alternative mode of transport.');
        }
        if (congestion < 0.3 && weather.riskAdd < 10) {
            strategies.push('Road conditions are excellent! Enjoy a smooth journey.');
        }

        if (strategies.length === 0) {
            strategies.push('Normal driving conditions. Follow standard traffic rules.');
        }

        return strategies;
    }

    function generateWarnings(today, tomorrow) {
        const warnings = [];

        if (today.riskScore >= 70) {
            warnings.push({
                level: 'critical',
                message: `High risk today (${today.riskScore}/100). Travel with extreme caution.`,
            });
        } else if (today.riskScore >= 45) {
            warnings.push({
                level: 'warning',
                message: `Moderate risk today (${today.riskScore}/100). Stay alert.`,
            });
        }

        if (today.congestionInfo.level === 'Severe') {
            warnings.push({
                level: 'critical',
                message: 'Severe congestion expected. Delays of 30+ minutes likely.',
            });
        }

        if (tomorrow.riskScore > today.riskScore + 15) {
            warnings.push({
                level: 'info',
                message: 'Tomorrow\'s conditions are worse. Travel today if possible.',
            });
        } else if (tomorrow.riskScore < today.riskScore - 15) {
            warnings.push({
                level: 'info',
                message: 'Tomorrow looks better — consider postponing if flexible.',
            });
        }

        today.events.forEach(e => {
            if (e.adjustedImpact > 0.25) {
                warnings.push({
                    level: 'warning',
                    message: `${e.name} is causing elevated traffic in the area.`,
                });
            }
        });

        return warnings;
    }

    function generateComparison(today, tomorrow) {
        const timeDiff = tomorrow.travelTime - today.travelTime;
        const timeDiffMins = Math.abs(Math.round(timeDiff / 60));

        if (Math.abs(timeDiff) < 120) {
            return { verdict: 'similar', message: 'Travel time is roughly the same both days.' };
        } else if (timeDiff > 0) {
            return { verdict: 'today_better', message: `Today is ${timeDiffMins} min faster. Travel now if you can.` };
        } else {
            return { verdict: 'tomorrow_better', message: `Tomorrow is ${timeDiffMins} min faster. Postpone if possible.` };
        }
    }

    // ─── Formatting ──────────────────────────────────────────

    function formatDurationFull(seconds) {
        if (seconds < 60) return `${Math.round(seconds)} sec`;
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.round((seconds % 3600) / 60);
        if (hrs === 0) return `${mins} min`;
        return `${hrs} hr ${mins} min`;
    }

    // ─── Public API ──────────────────────────────────────────
    return { predict };

})();
