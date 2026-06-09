"""
ui_template.py — Eingebettete Web-UI.

- Upload: MIDI (.mid/.midi) ODER Audio (.wav/.mp3/.m4a/.ogg/.flac).
  Audio wird IM BROWSER mit Spotify Basic Pitch (TF.js) zu Note-Events
  transkribiert; der Server bleibt schlank.
- Harmonie-Kontext optional: keiner / Standard bzw. eigene Changes / Tonart.
- Piano-Roll der erkannten Noten (mit Taktraster + Akkord-Lane) zum
  Nachpruefen, ob die Erkennung stimmt.
Rote Card-Optik, keine abcjs-Notenschrift.
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback — Improvisations-Analyse</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-red-50 via-white to-white min-h-screen">
    <div class="max-w-4xl mx-auto p-6">
        <div class="text-center mb-8 pt-8">
            <div class="inline-flex items-center justify-center w-16 h-16 bg-red-600 rounded-full mb-4">
                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
                </svg>
            </div>
            <h1 class="text-4xl font-bold text-gray-900 mb-2">Jazz-Improvisation Feedback</h1>
            <p class="text-gray-600">Solo-Klavier als <strong>MIDI oder Audio</strong> hochladen. Harmonie-Kontext ist optional.</p>
        </div>

        <div id="aiStatus" class="mb-6"></div>

        <div class="bg-white rounded-2xl shadow-lg p-8 mb-6 space-y-6">
            <input type="file" id="fileInput" accept=".mid,.midi,.wav,.mp3,.m4a,.ogg,.flac,audio/*" class="hidden">

            <div id="dropzone" class="border-2 border-dashed border-red-300 rounded-xl p-10 text-center cursor-pointer hover:border-red-500 hover:bg-red-50 transition-all">
                <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg font-semibold text-gray-700 mb-2" id="fileName">Datei hochladen</p>
                <p class="text-sm text-gray-500">MIDI (.mid/.midi) oder Audio (.wav/.mp3/.m4a/.ogg/.flac)</p>
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-2">Harmonie-Kontext <span class="text-gray-400 font-normal">(optional)</span></label>
                <select id="ctxKind" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200 text-lg">
                    <option value="none">Keiner — nur Time-Feel, Dynamik & Kontur</option>
                    <option value="tune">Standard / eigene Changes — volle Harmonie-Analyse</option>
                    <option value="key">Tonart / Modus — Analyse gegen eine Tonleiter</option>
                </select>
            </div>

            <div id="ctxTune" class="hidden space-y-3 border-l-2 border-red-200 pl-4">
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Standard / Tune</label>
                    <select id="tuneSelect" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200">
                        <option value="">— Standard waehlen —</option>
                    </select>
                </div>
                <details class="border border-gray-200 rounded-xl p-3">
                    <summary class="cursor-pointer text-sm font-semibold text-gray-700">Oder eigene Changes</summary>
                    <p class="text-xs text-gray-500 mt-2 mb-2">Takte mit <code>|</code> trennen, Akkorde pro Takt mit Leerzeichen. Z.B. <code>Dm7 | G7 | Cmaj7 | Cmaj7</code>. Hat Vorrang vor dem Standard.</p>
                    <textarea id="manualChanges" rows="2" placeholder="Dm7 | G7 | Cmaj7 | Cmaj7" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 font-mono text-sm"></textarea>
                </details>
            </div>

            <div id="ctxKey" class="hidden grid grid-cols-2 gap-4 border-l-2 border-red-200 pl-4">
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Tonika</label>
                    <select id="keyTonic" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500">
                        <option value="">— aus dem Spiel erkennen —</option>
                        <option>C</option><option>Db</option><option>D</option><option>Eb</option>
                        <option>E</option><option>F</option><option>Gb</option><option>G</option>
                        <option>Ab</option><option>A</option><option>Bb</option><option>B</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Modus</label>
                    <select id="keyMode" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500"></select>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Taktart (Beats/Takt)</label>
                    <input id="beatsPerBar" type="number" min="2" max="12" value="4" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Tempo (BPM)<span id="bpmHint" class="text-gray-400 font-normal"> — optional</span></label>
                    <input id="bpm" type="number" min="30" max="400" placeholder="auto aus MIDI" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500">
                </div>
            </div>

            <button id="analyzeBtn" disabled class="w-full bg-red-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
                Analyse starten
            </button>
            <p class="text-xs text-gray-400 text-center">Audio wird im Browser transkribiert (Basic Pitch). Mit Changes/Tonart sollte die Aufnahme auf Beat 1 beginnen.</p>
        </div>

        <div id="loading" class="bg-white rounded-2xl shadow-lg p-6 mb-6 hidden">
            <div class="flex items-center gap-3 mb-3">
                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600"></div>
                <span class="text-gray-700 font-medium" id="loadingText">Analysiere...</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
                <div id="progressBar" class="bg-red-600 h-2 rounded-full transition-all" style="width: 0%"></div>
            </div>
        </div>

        <div id="results" class="space-y-6"></div>
    </div>

    <script>
        let selectedFile = null;
        let isAudio = false;
        const MIDI_RE = /\\.(mid|midi)$/i;
        const AUDIO_RE = /\\.(wav|mp3|m4a|ogg|flac|aac)$/i;

        function refreshBtn() { document.getElementById('analyzeBtn').disabled = !selectedFile; }
        function setProgress(p, txt) {
            document.getElementById('progressBar').style.width = p + '%';
            if (txt != null) document.getElementById('loadingText').textContent = txt;
        }

        fetch('/ai-status').then(r => r.json()).then(d => {
            document.getElementById('aiStatus').innerHTML = d.ai_enabled
                ? '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="text-sm font-medium text-red-900">Apertus AI aktiv — Feedback aus der regelbasierten Analyse.</div></div>'
                : '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">AI deaktiviert — einfaches regelbasiertes Feedback.</span></div>';
        });
        fetch('/standards').then(r => r.json()).then(d => {
            const sel = document.getElementById('tuneSelect');
            (d.standards || []).forEach(n => { const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); });
        });
        fetch('/modes').then(r => r.json()).then(d => {
            const sel = document.getElementById('keyMode');
            (d.modes || []).forEach(m => { const o = document.createElement('option'); o.value = m.value; o.textContent = m.label; sel.appendChild(o); });
        });

        document.getElementById('ctxKind').addEventListener('change', (e) => {
            document.getElementById('ctxTune').classList.toggle('hidden', e.target.value !== 'tune');
            document.getElementById('ctxKey').classList.toggle('hidden', e.target.value !== 'key');
        });

        document.getElementById('dropzone').addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            if (MIDI_RE.test(file.name)) { isAudio = false; }
            else if (AUDIO_RE.test(file.name) || (file.type || '').startsWith('audio/')) { isAudio = true; }
            else { alert('Bitte MIDI (.mid/.midi) oder Audio (.wav/.mp3/.m4a/.ogg/.flac) waehlen.'); return; }
            selectedFile = file;
            document.getElementById('fileName').textContent = (isAudio ? '🎵 ' : '🎹 ') + file.name;
            const dz = document.getElementById('dropzone');
            dz.classList.add('border-green-400', 'bg-green-50'); dz.classList.remove('border-red-300');
            document.getElementById('bpmHint').textContent = isAudio ? ' — empfohlen bei Audio' : ' — optional';
            refreshBtn();
        });

        // --- Audio -> Note-Events (Spotify Basic Pitch, im Browser) ---------
        async function transcribeAudio(file, onProgress) {
            const mod = await import('https://esm.sh/@spotify/basic-pitch@1.0.1');
            const { BasicPitch, noteFramesToTime, addPitchBendsToNoteEvents, outputToNotesPoly } = mod;
            const arrayBuf = await file.arrayBuffer();
            const ac = new (window.AudioContext || window.webkitAudioContext)();
            const decoded = await ac.decodeAudioData(arrayBuf);
            // Resample auf 22050 Hz mono (Basic Pitch erwartet das).
            const SR = 22050;
            const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
            const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
            const mono = await off.startRendering();

            const bp = new BasicPitch('https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json');
            const frames = [], onsets = [], contours = [];
            await bp.evaluateModel(mono,
                (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); },
                (p) => onProgress(p));
            const notes = noteFramesToTime(addPitchBendsToNoteEvents(contours,
                outputToNotesPoly(frames, onsets, 0.5, 0.3, 5)));
            // -> [start_s, end_s, pitch_midi, amplitude]
            return notes.map(n => [n.startTimeSeconds, n.startTimeSeconds + n.durationSeconds, n.pitchMidi, n.amplitude]);
        }

        function contextFields() {
            const kind = document.getElementById('ctxKind').value;
            const o = {
                beats_per_bar: document.getElementById('beatsPerBar').value || '4',
                bpm: document.getElementById('bpm').value || ''
            };
            if (kind === 'tune') {
                o.tune = document.getElementById('tuneSelect').value;
                o.manual_changes = document.getElementById('manualChanges').value;
            } else if (kind === 'key') {
                o.key_tonic = document.getElementById('keyTonic').value;
                o.key_mode = document.getElementById('keyMode').value;
            }
            return o;
        }

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            setProgress(8, 'Vorbereiten...');
            try {
                let id;
                if (isAudio) {
                    setProgress(10, 'Transkribiere Audio im Browser (Basic Pitch)...');
                    const notes = await transcribeAudio(selectedFile, (p) =>
                        setProgress(10 + Math.round(p * 40), 'Transkribiere Audio... ' + Math.round(p * 100) + '%'));
                    if (!notes.length) throw new Error('Keine Noten erkannt — anderes/laengeres Audio probieren.');
                    setProgress(55, 'Sende ' + notes.length + ' Noten zur Analyse...');
                    const body = Object.assign({ notes }, contextFields());
                    const res = await fetch('/analyze-notes', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Analyse-Upload fehlgeschlagen');
                    id = (await res.json()).analysis_id;
                } else {
                    setProgress(15, 'Upload MIDI...');
                    const fd = new FormData();
                    fd.append('file', selectedFile);
                    const cf = contextFields();
                    Object.keys(cf).forEach(k => fd.append(k, cf[k]));
                    const res = await fetch('/analyze-jazz', { method: 'POST', body: fd });
                    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Upload fehlgeschlagen');
                    id = (await res.json()).analysis_id;
                }
                setProgress(60, 'Analysiere...');
                poll(id);
            } catch (e) {
                console.error(e);
                alert('Fehler: ' + e.message);
                document.getElementById('loading').classList.add('hidden');
            }
        });

        function poll(id) {
            const maxAttempts = 90;
            let attempts = 0;
            const iv = setInterval(async () => {
                attempts++;
                try {
                    const data = await (await fetch('/result/' + id)).json();
                    if (data.status === 'completed') {
                        clearInterval(iv); setProgress(100);
                        setTimeout(() => { document.getElementById('loading').classList.add('hidden'); render(data.result); }, 300);
                    } else if (data.status === 'error') {
                        clearInterval(iv); alert('Fehler: ' + data.error);
                        document.getElementById('loading').classList.add('hidden');
                    } else {
                        setProgress(60 + Math.min(35, attempts), data.stage === 'ai' ? 'Apertus AI Feedback...' : 'Noten & Theorie...');
                    }
                    if (attempts >= maxAttempts) { clearInterval(iv); alert('Timeout'); document.getElementById('loading').classList.add('hidden'); }
                } catch (e) { console.error('poll', e); }
            }, 2000);
        }

        // --- Rendering ------------------------------------------------------
        const pct = x => (x == null ? '–' : Math.round(x * 100) + '%');
        const scoreColor = s => s >= 8 ? 'green' : s >= 6 ? 'yellow' : 'orange';
        function metric(label, value) {
            return '<div class="bg-white/70 rounded-lg p-3"><div class="text-2xl font-bold text-gray-800">' + value
                + '</div><div class="text-xs text-gray-500">' + label + '</div></div>';
        }
        function bar(label, ratio, colorClass) {
            const w = Math.round((ratio || 0) * 100);
            return '<div class="mb-2"><div class="flex justify-between text-xs text-gray-600 mb-1"><span>' + label
                + '</span><span>' + w + '%</span></div><div class="w-full bg-gray-200 rounded-full h-2">'
                + '<div class="' + colorClass + ' h-2 rounded-full" style="width:' + w + '%"></div></div></div>';
        }

        function drawPianoRoll(report) {
            const cv = document.getElementById('pianoRoll');
            if (!cv) return;
            const notes = report.notes_view || [];
            if (!notes.length) return;
            const grid = report.grid || {};
            const bpm = grid.bpm || 120, spb = 60 / bpm;
            const downbeat = grid.downbeat || 0, bpbar = grid.beats_per_bar || 4;
            const t0 = grid.t_start != null ? grid.t_start : Math.min.apply(null, notes.map(n => n.on));
            const t1 = grid.t_end != null ? grid.t_end : Math.max.apply(null, notes.map(n => n.off));
            const dur = Math.max(0.5, t1 - t0);
            const ps = notes.map(n => n.p);
            const pmin = Math.min.apply(null, ps) - 2, pmax = Math.max.apply(null, ps) + 2;
            const rowH = 7, padTop = 16, padBot = 6;
            const W = Math.max(700, Math.min(5000, Math.round(dur * 110)));
            const H = (pmax - pmin + 1) * rowH + padTop + padBot;
            cv.width = W; cv.height = H;
            const g = cv.getContext('2d');
            g.clearRect(0, 0, W, H);
            const X = t => ((t - t0) / dur) * (W - 10) + 5;
            const Y = p => H - padBot - (p - pmin) * rowH;

            // Oktav-Hilfslinien (jedes C)
            g.strokeStyle = '#f1f5f9'; g.lineWidth = 1;
            for (let p = pmin; p <= pmax; p++) { if (p % 12 === 0) { g.beginPath(); g.moveTo(0, Y(p)); g.lineTo(W, Y(p)); g.stroke(); } }

            // Beat-/Taktlinien
            const firstBeat = t0 - ((t0 - downbeat) % spb || 0);
            g.font = '10px sans-serif';
            for (let t = firstBeat; t <= t1 + 1e-6; t += spb) {
                if (t < t0 - 1e-6) continue;
                const beatsFromDb = Math.round((t - downbeat) / spb);
                const isBar = (((beatsFromDb % bpbar) + bpbar) % bpbar) === 0;
                g.strokeStyle = isBar ? '#cbd5e1' : '#eef2f7'; g.lineWidth = isBar ? 1.5 : 1;
                g.beginPath(); g.moveTo(X(t), padTop - 4); g.lineTo(X(t), H); g.stroke();
            }

            // Akkord-Lane (changes), ueber alle Form-Wiederholungen
            const cvw = report.changes_view || [];
            if (cvw.length) {
                const formLen = Math.max.apply(null, cvw.map(s => s.bar)) + 1;
                g.fillStyle = '#be123c';
                for (let k = 0; k < 200; k++) {
                    let any = false;
                    for (const s of cvw) {
                        const t = downbeat + ((s.bar + k * formLen) * bpbar + s.beat) * spb;
                        if (t > t1 + 1e-6) continue;
                        if (t < t0 - 1e-6) continue;
                        any = true;
                        g.fillText(s.symbol, X(t) + 2, 11);
                    }
                    const tNext = downbeat + (k * formLen * bpbar) * spb;
                    if (tNext > t1 + 1e-6 && k > 0) break;
                    if (!any && downbeat + (k * formLen * bpbar) * spb > t1) break;
                }
            }

            // Noten
            notes.forEach(n => {
                const nx = X(n.on), nw = Math.max(2, X(n.off) - X(n.on));
                g.fillStyle = n.voice === 'line' ? '#ef4444' : '#3b82f6';
                g.fillRect(nx, Y(n.p) - rowH + 1, nw, rowH - 1.5);
            });
        }

        function render(d) {
            const r = d.report || {};
            const ctx = r.context || { kind: 'none', label: 'Ohne Harmonie-Kontext' };
            const line = r.line || {};
            const dist = line.distribution;
            const voic = r.voicings || {};
            const vl = r.voice_leading || {};
            const tf = r.time_feel || {};
            const cont = r.contour || {};
            const used = d.used || {};
            let html = '';

            let ctxLine;
            if (ctx.kind === 'changes') ctxLine = 'Voller Changes-Kontext';
            else if (ctx.kind === 'key') ctxLine = 'Tonart-Kontext' + (ctx.tonic_known ? '' : ' (Tonika geschaetzt)');
            else ctxLine = 'Kein Harmonie-Kontext — nur Time-Feel, Dynamik & Kontur';
            const srcLabel = used.source === 'audio' ? 'Audio (Basic Pitch)' : 'MIDI';
            html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-5">'
                + '<div class="text-sm opacity-90">' + ctxLine + '</div>'
                + '<div class="text-2xl font-bold">' + (ctx.label || d.tune || '—') + '</div>'
                + '<div class="text-sm opacity-90 mt-1">' + (used.n_notes || 0) + ' Noten · '
                + srcLabel + ' · ' + (used.bpm || '?') + ' BPM · ' + (used.beats_per_bar || 4) + '/4 · '
                + (d.ai_generated ? 'Feedback: Apertus AI' : 'Feedback: regelbasiert') + '</div></div>';

            // Piano-Roll (Erkennung pruefen)
            html += '<div class="bg-white border-2 border-gray-200 rounded-2xl p-5 shadow">'
                + '<div class="flex items-center justify-between mb-2"><h3 class="font-bold text-gray-900">Erkannte Noten (Piano-Roll)</h3>'
                + '<span class="text-xs text-gray-500"><span style="color:#ef4444">■</span> Melodielinie&nbsp; <span style="color:#3b82f6">■</span> Begleit-Voicing</span></div>'
                + '<p class="text-xs text-gray-500 mb-3">Senkrechte Linien = Taktraster (kraeftig = Taktanfang)' + (ctx.kind === 'changes' ? ', rot oben = Akkorde' : '') + '. So kannst du pruefen, ob Erkennung und Taktlage stimmen.</p>'
                + '<div class="overflow-x-auto"><canvas id="pianoRoll" class="block"></canvas></div></div>';

            const c = scoreColor(d.overall_score);
            html += '<div class="bg-white rounded-2xl shadow-lg p-6 text-center">'
                + '<h2 class="text-xl font-bold mb-2">Gesamtbewertung</h2>'
                + '<div class="text-5xl font-bold text-' + c + '-600">' + d.overall_score
                + '<span class="text-2xl text-gray-400">/10</span></div></div>';

            html += '<div class="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6">';
            if (dist) {
                const harmonic = ctx.kind === 'key';
                html += '<h3 class="font-semibold text-purple-900 mb-4">' + (harmonic ? 'Linie in der Tonleiter' : 'Linie in den Changes') + '</h3>';
                html += '<div class="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">';
                html += metric(harmonic ? 'Stabile Toene auf betonten Zeiten' : 'Akkordtoene auf betonten Zeiten', pct(line.chord_tones_on_strong_beats));
                html += metric('Linien-Noten', line.n_notes != null ? line.n_notes : '–');
                html += metric(harmonic ? 'Tonleiterfremd auf betonten Zeiten' : 'Avoid auf betonten Zeiten', (line.avoid_notes_on_strong_beats || []).length);
                html += '</div>';
                html += bar(harmonic ? 'Stabile Toene (1/3/5/7)' : 'Akkordtoene', dist.chord_tone, 'bg-green-500');
                html += bar(harmonic ? 'Tonleiter-Farbtoene' : 'Tensions', dist.tension, 'bg-blue-500');
                html += bar(harmonic ? 'Chromatisch (ausserhalb)' : 'Chromatik', dist.chromatic, 'bg-yellow-500');
                if (!harmonic) html += bar('Avoid', dist.avoid, 'bg-red-500');
                const av = line.avoid_notes_on_strong_beats || [];
                if (av.length) html += '<div class="mt-3 text-sm text-red-700">' + (harmonic ? 'Tonleiterfremd: ' : 'Avoid-Noten: ')
                    + av.slice(0, 6).map(a => a.note + (a.chord && !harmonic ? ' auf ' + a.chord : '') + ' (T' + a.bar + ')').join(', ') + (av.length > 6 ? ' …' : '') + '</div>';
                if (ctx.kind === 'key' && voic.comp_in_scale_ratio != null)
                    html += '<p class="text-sm text-gray-700 mt-3">Begleit-Voicings: ' + pct(voic.comp_in_scale_ratio) + ' der Toene in der Tonleiter.</p>';
            } else {
                html += '<h3 class="font-semibold text-purple-900 mb-4">Linienform (ohne Harmonie-Kontext)</h3>';
                html += '<div class="grid grid-cols-2 sm:grid-cols-3 gap-3">';
                html += metric('Linien-Noten', line.n_notes != null ? line.n_notes : '–');
                html += metric('Schrittweise gefuehrt', pct(line.stepwise_ratio));
                html += metric('Ø Intervall', line.avg_interval_semitones != null ? line.avg_interval_semitones + ' HT' : '–');
                html += '</div><p class="text-xs text-gray-500 mt-3">Fuer Aussagen zu Akkordtoenen/Tensions oben einen Standard oder eine Tonart waehlen.</p>';
            }
            html += '</div>';

            if (voic.n_voicings) {
                html += '<div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6">';
                html += '<h3 class="font-semibold text-blue-900 mb-4">Voicings & Stimmfuehrung</h3>';
                html += '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">';
                html += metric('Voicings', voic.n_voicings);
                html += metric('Rootless', pct(voic.rootless_ratio));
                html += metric('Leitton-Abdeckung', pct(voic.guide_tone_coverage));
                html += metric('Ø Tensions', voic.avg_tensions_per_voicing);
                html += '</div></div>';
            }

            html += '<div class="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6">';
            html += '<h3 class="font-semibold text-red-900 mb-4">Time-Feel & Kontur</h3>';
            html += '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">';
            html += metric('Swing-Ratio', tf.swing_ratio != null ? tf.swing_ratio : '–');
            html += metric('Lage zum Beat', tf.timing_comment || '–');
            html += metric('Ambitus', cont.pitch_range_semitones != null ? cont.pitch_range_semitones + ' HT' : '–');
            html += metric('Dichte', cont.notes_per_second != null ? cont.notes_per_second + '/s' : '–');
            html += '</div>';
            if (tf.feel_comment) html += '<p class="text-sm text-gray-700 mt-3">' + tf.feel_comment + '</p>';
            if (vl.smoothness_comment && vl.top_voice_avg_leap_semitones != null)
                html += '<p class="text-sm text-gray-700 mt-1">Oberstimme: ' + vl.smoothness_comment + ' (Ø ' + vl.top_voice_avg_leap_semitones + ' HT, ' + pct(vl.top_voice_stepwise_ratio) + ' schrittweise)</p>';
            html += '</div>';

            if ((d.summary || []).length) {
                html += '<div class="bg-white border border-gray-200 rounded-xl p-6"><h3 class="font-semibold text-gray-900 mb-3">Regel-Zusammenfassung</h3><ul class="space-y-2">';
                d.summary.forEach(s => html += '<li class="text-sm text-gray-700">• ' + s + '</li>');
                html += '</ul></div>';
            }

            const fb = d.feedback || {};
            [['Rhythmus & Time-Feel', fb.rhythm], ['Harmonie', fb.harmony],
             ['Melodie & Linienfuehrung', fb.melody], ['Artikulation & Dynamik', fb.articulation]
            ].forEach(([title, cat]) => {
                if (!cat) return;
                const cc = scoreColor(cat.score);
                html += '<div class="bg-white rounded-2xl shadow-lg p-6">'
                    + '<div class="flex items-center justify-between mb-2"><h3 class="text-xl font-bold">' + title
                    + '</h3><span class="text-2xl font-bold text-' + cc + '-600">' + cat.score.toFixed(1) + '</span></div>'
                    + '<p class="text-gray-700 mb-3">' + (cat.feedback || '') + '</p>'
                    + '<div class="bg-gray-50 rounded-lg p-4"><p class="font-semibold mb-2 text-sm">Tipps:</p><ul class="space-y-1">'
                    + (cat.tips || []).map(t => '<li class="text-sm text-gray-600">• ' + t + '</li>').join('')
                    + '</ul></div></div>';
            });

            document.getElementById('results').innerHTML = html;
            drawPianoRoll(r);
        }
    </script>
</body>
</html>
"""
