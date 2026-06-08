"""
ui_template.py — Eingebettete Web-UI fuer den neuen Known-Changes-Flow.

Aus main.py ausgelagert (war ~480 Zeilen inline). Behaelt die rote
Card-Optik des Originals, ersetzt aber den Tonart-Wahler durch einen
Tune-/Changes-Wahler und rendert den ehrlichen jazzfb-Report
(Akkordton-Bindung, Voicings, Time-Feel, Regel-Zusammenfassung) plus das
Apertus-Score-Feedback. Keine abcjs-Notenschrift mehr (die hing an der
geratenen Harmonie, die wir bewusst nicht mehr machen).
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback — Improvisation ueber bekannte Changes</title>
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
            <p class="text-gray-600">Solo-Klavier ueber <strong>bekannte Changes</strong> — die Engine bewertet die Realisierung, nicht geratene Harmonie.</p>
        </div>

        <div id="aiStatus" class="mb-6"></div>

        <div class="bg-white rounded-2xl shadow-lg p-8 mb-6 space-y-6">
            <input type="file" id="fileInput" accept=".mid,.midi" class="hidden">

            <div id="dropzone" class="border-2 border-dashed border-red-300 rounded-xl p-10 text-center cursor-pointer hover:border-red-500 hover:bg-red-50 transition-all">
                <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg font-semibold text-gray-700 mb-2" id="fileName">MIDI-Datei hochladen</p>
                <p class="text-sm text-gray-500">Solo-Klavier als .mid / .midi (Audio-Upload folgt)</p>
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-2">Standard / Tune</label>
                <select id="tuneSelect" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200 transition-all text-lg">
                    <option value="">— Standard waehlen —</option>
                </select>
            </div>

            <details class="border border-gray-200 rounded-xl p-4">
                <summary class="cursor-pointer text-sm font-semibold text-gray-700">Oder eigene Changes eingeben</summary>
                <p class="text-xs text-gray-500 mt-2 mb-2">Takte mit <code>|</code> trennen, mehrere Akkorde pro Takt mit Leerzeichen. Beispiel: <code>Dm7 | G7 | Cmaj7 | Cmaj7</code>. Wenn ausgefuellt, hat das Vorrang vor dem Standard.</p>
                <textarea id="manualChanges" rows="2" placeholder="Dm7 | G7 | Cmaj7 | Cmaj7" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200 font-mono text-sm"></textarea>
            </details>

            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Taktart (Beats/Takt)</label>
                    <input id="beatsPerBar" type="number" min="2" max="12" value="4" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Tempo (BPM, optional)</label>
                    <input id="bpm" type="number" min="30" max="400" placeholder="auto aus MIDI" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200">
                </div>
            </div>

            <button id="analyzeBtn" disabled class="w-full bg-red-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
                Analyse starten
            </button>
            <p class="text-xs text-gray-400 text-center">Tipp: Aufnahme sollte auf Beat 1 des ersten Takts der Changes beginnen.</p>
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

        function refreshBtn() {
            const hasTune = document.getElementById('tuneSelect').value
                || document.getElementById('manualChanges').value.trim();
            document.getElementById('analyzeBtn').disabled = !(selectedFile && hasTune);
        }

        // AI-Status
        fetch('/ai-status').then(r => r.json()).then(data => {
            const el = document.getElementById('aiStatus');
            if (data.ai_enabled) {
                el.innerHTML = '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="text-sm font-medium text-red-900">Apertus AI aktiv — Feedback wird aus der regelbasierten Analyse generiert.</div></div>';
            } else {
                el.innerHTML = '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">AI deaktiviert — es wird ein einfaches regelbasiertes Feedback gezeigt.</span></div>';
            }
        });

        // Standards laden
        fetch('/standards').then(r => r.json()).then(data => {
            const sel = document.getElementById('tuneSelect');
            (data.standards || []).forEach(name => {
                const o = document.createElement('option');
                o.value = name; o.textContent = name; sel.appendChild(o);
            });
        });

        document.getElementById('dropzone').addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('tuneSelect').addEventListener('change', refreshBtn);
        document.getElementById('manualChanges').addEventListener('input', refreshBtn);

        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && (file.name.endsWith('.mid') || file.name.endsWith('.midi'))) {
                selectedFile = file;
                document.getElementById('fileName').textContent = '✓ ' + file.name;
                const dz = document.getElementById('dropzone');
                dz.classList.add('border-green-400', 'bg-green-50');
                dz.classList.remove('border-red-300');
                refreshBtn();
            } else {
                alert('Bitte eine MIDI-Datei (.mid oder .midi) waehlen.');
            }
        });

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            document.getElementById('loadingText').textContent = 'Upload...';
            document.getElementById('progressBar').style.width = '10%';

            const fd = new FormData();
            fd.append('file', selectedFile);
            fd.append('tune', document.getElementById('tuneSelect').value);
            fd.append('manual_changes', document.getElementById('manualChanges').value);
            fd.append('beats_per_bar', document.getElementById('beatsPerBar').value || '4');
            fd.append('bpm', document.getElementById('bpm').value || '');

            try {
                const res = await fetch('/analyze-jazz', { method: 'POST', body: fd });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({detail: res.statusText}));
                    throw new Error(err.detail || 'Upload fehlgeschlagen');
                }
                const data = await res.json();
                document.getElementById('loadingText').textContent = 'Analysiere...';
                document.getElementById('progressBar').style.width = '20%';
                poll(data.analysis_id);
            } catch (e) {
                alert('Fehler: ' + e.message);
                document.getElementById('loading').classList.add('hidden');
            }
        });

        function poll(id) {
            const maxAttempts = 60;
            let attempts = 0;
            const iv = setInterval(async () => {
                attempts++;
                try {
                    const data = await (await fetch('/result/' + id)).json();
                    if (data.status === 'completed') {
                        clearInterval(iv);
                        document.getElementById('progressBar').style.width = '100%';
                        setTimeout(() => {
                            document.getElementById('loading').classList.add('hidden');
                            render(data.result);
                        }, 400);
                    } else if (data.status === 'error') {
                        clearInterval(iv);
                        alert('Fehler: ' + data.error);
                        document.getElementById('loading').classList.add('hidden');
                    } else {
                        const p = 20 + (attempts / maxAttempts) * 70;
                        document.getElementById('progressBar').style.width = p + '%';
                        document.getElementById('loadingText').textContent =
                            data.stage === 'ai' ? 'Apertus AI Feedback...' : 'Noten & Theorie...';
                    }
                    if (attempts >= maxAttempts) {
                        clearInterval(iv);
                        alert('Timeout');
                        document.getElementById('loading').classList.add('hidden');
                    }
                } catch (e) { console.error('poll', e); }
            }, 2000);
        }

        // --- Rendering ------------------------------------------------------
        const pct = x => (x == null ? '–' : Math.round(x * 100) + '%');
        const scoreColor = s => s >= 8 ? 'green' : s >= 6 ? 'yellow' : 'orange';

        function metric(label, value, sub) {
            return '<div class="bg-white/70 rounded-lg p-3"><div class="text-2xl font-bold text-gray-800">' + value
                + '</div><div class="text-xs text-gray-500">' + label + (sub ? ' · ' + sub : '') + '</div></div>';
        }

        function bar(label, ratio, colorClass) {
            const w = Math.round((ratio || 0) * 100);
            return '<div class="mb-2"><div class="flex justify-between text-xs text-gray-600 mb-1"><span>' + label
                + '</span><span>' + w + '%</span></div><div class="w-full bg-gray-200 rounded-full h-2">'
                + '<div class="' + colorClass + ' h-2 rounded-full" style="width:' + w + '%"></div></div></div>';
        }

        function render(d) {
            const r = d.report || {};
            const line = r.line || {};
            const dist = line.distribution || {};
            const voic = r.voicings || {};
            const vl = r.voice_leading || {};
            const tf = r.time_feel || {};
            const cont = r.contour || {};
            const used = d.used || {};
            let html = '';

            // Kopf
            html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-5">';
            html += '<div class="text-sm opacity-90">Changes (vorgegeben)</div>';
            html += '<div class="text-2xl font-bold">' + (d.tune || 'Eigene Changes') + '</div>';
            html += '<div class="text-sm opacity-90 mt-1">' + (used.n_notes || 0) + ' Noten · '
                + (used.bpm || '?') + ' BPM · ' + (used.beats_per_bar || 4) + '/4 · '
                + (d.ai_generated ? 'Feedback: Apertus AI' : 'Feedback: regelbasiert') + '</div></div>';

            // Gesamtbewertung
            const c = scoreColor(d.overall_score);
            html += '<div class="bg-white rounded-2xl shadow-lg p-6 text-center">'
                + '<h2 class="text-xl font-bold mb-2">Gesamtbewertung</h2>'
                + '<div class="text-5xl font-bold text-' + c + '-600">' + d.overall_score
                + '<span class="text-2xl text-gray-400">/10</span></div></div>';

            // Linie in den Changes
            html += '<div class="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6">';
            html += '<h3 class="font-semibold text-purple-900 mb-4">Linie in den Changes</h3>';
            html += '<div class="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">';
            html += metric('Akkordtoene auf betonten Zeiten', pct(line.chord_tones_on_strong_beats));
            html += metric('Linien-Noten', line.n_notes != null ? line.n_notes : '–');
            html += metric('Avoid auf betonten Zeiten', (line.avoid_notes_on_strong_beats || []).length);
            html += '</div>';
            html += bar('Akkordtoene', dist.chord_tone, 'bg-green-500');
            html += bar('Tensions', dist.tension, 'bg-blue-500');
            html += bar('Chromatik', dist.chromatic, 'bg-yellow-500');
            html += bar('Avoid', dist.avoid, 'bg-red-500');
            const av = line.avoid_notes_on_strong_beats || [];
            if (av.length) {
                html += '<div class="mt-3 text-sm text-red-700">Avoid-Noten: '
                    + av.slice(0, 6).map(a => a.note + ' auf ' + a.chord + ' (T' + a.bar + ')').join(', ')
                    + (av.length > 6 ? ' …' : '') + '</div>';
            }
            html += '</div>';

            // Voicings + Stimmfuehrung
            if (voic.n_voicings) {
                html += '<div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6">';
                html += '<h3 class="font-semibold text-blue-900 mb-4">Voicings & Stimmfuehrung</h3>';
                html += '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">';
                html += metric('Voicings', voic.n_voicings);
                html += metric('Rootless', pct(voic.rootless_ratio));
                html += metric('Leitton-Abdeckung', pct(voic.guide_tone_coverage));
                html += metric('Ø Tensions', voic.avg_tensions_per_voicing);
                html += '</div>';
                if (vl.smoothness_comment) {
                    html += '<p class="text-sm text-gray-700 mt-3">Oberstimme: ' + vl.smoothness_comment
                        + (vl.top_voice_avg_leap_semitones != null
                            ? ' (Ø ' + vl.top_voice_avg_leap_semitones + ' HT, '
                              + pct(vl.top_voice_stepwise_ratio) + ' schrittweise)' : '') + '</p>';
                }
                html += '</div>';
            }

            // Time-Feel + Kontur
            html += '<div class="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6">';
            html += '<h3 class="font-semibold text-red-900 mb-4">Time-Feel & Kontur</h3>';
            html += '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">';
            html += metric('Swing-Ratio', tf.swing_ratio != null ? tf.swing_ratio : '–');
            html += metric('Lage zum Beat', tf.timing_comment || '–');
            html += metric('Ambitus', cont.pitch_range_semitones != null ? cont.pitch_range_semitones + ' HT' : '–');
            html += metric('Dichte', cont.notes_per_second != null ? cont.notes_per_second + '/s' : '–');
            html += '</div>';
            if (tf.feel_comment) html += '<p class="text-sm text-gray-700 mt-3">' + tf.feel_comment + '</p>';
            html += '</div>';

            // Regel-Zusammenfassung (ehrliche Fakten)
            if ((d.summary || []).length) {
                html += '<div class="bg-white border border-gray-200 rounded-xl p-6">';
                html += '<h3 class="font-semibold text-gray-900 mb-3">Regel-Zusammenfassung</h3><ul class="space-y-2">';
                d.summary.forEach(s => html += '<li class="text-sm text-gray-700">• ' + s + '</li>');
                html += '</ul></div>';
            }

            // Apertus-Feedback je Kategorie
            const fb = d.feedback || {};
            const cats = [
                ['Rhythmus & Time-Feel', fb.rhythm],
                ['Harmonie', fb.harmony],
                ['Melodie & Linienfuehrung', fb.melody],
                ['Artikulation & Dynamik', fb.articulation],
            ];
            cats.forEach(([title, cat]) => {
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
        }
    </script>
</body>
</html>
"""
