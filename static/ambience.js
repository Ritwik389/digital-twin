// An original, quiet ambient loop, synthesized locally after a visitor gesture.
(() => {
    const button = document.getElementById('btn-music');
    const volume = document.getElementById('music-volume');
    let preferences = { enabled: true, volume: 10 };
    try { preferences = { ...preferences, ...JSON.parse(localStorage.getItem('jensen_ambience') || '{}') }; } catch { /* defaults */ }
    preferences.volume = Math.min(30, Math.max(0, Number(preferences.volume) || 0));
    let context, master, source, ducked = false, entered = false;
    const remember = () => { try { localStorage.setItem('jensen_ambience', JSON.stringify(preferences)); } catch { /* optional */ } };
    function update() {
        const playing = preferences.enabled && context?.state === 'running';
        button.textContent = preferences.enabled ? '♫ Music on' : '♫ Music off';
        button.setAttribute('aria-pressed', String(preferences.enabled));
        button.dataset.playback = playing ? 'playing' : 'paused';
        button.dataset.ducked = String(ducked);
        volume.value = preferences.volume;
        if (master) master.gain.setTargetAtTime(preferences.enabled ? preferences.volume / 100 * (ducked ? .22 : 1) : 0, context.currentTime, .22);
    }
    function makeLoop(ctx) {
        const duration = 32, rate = ctx.sampleRate;
        const buffer = ctx.createBuffer(2, rate * duration, rate);
        const chords = [[130.81, 164.81, 196], [110, 130.81, 164.81], [87.31, 130.81, 174.61], [98, 146.83, 196]];
        for (let channel = 0; channel < 2; channel++) {
            const samples = buffer.getChannelData(channel);
            for (let i = 0; i < samples.length; i++) {
                const t = i / rate, chord = chords[Math.floor(t / 8)], local = t % 8;
                const envelope = Math.sin(Math.PI * local / 8) ** 2;
                let pad = 0;
                for (const frequency of chord) {
                    const phase = 2 * Math.PI * frequency * t * (channel ? 1.001 : .999);
                    pad += (Math.sin(phase) + .12 * Math.sin(phase * 2)) / 3;
                }
                const beat = t % 4;
                const note = chord[Math.floor(local / 4) % 3] * 2;
                const bell = Math.sin(2 * Math.PI * note * beat) * Math.min(1, beat * 12) * Math.exp(-beat * 1.7);
                samples[i] = .24 * envelope * pad + .035 * envelope * bell;
            }
        }
        return buffer;
    }
    async function start() {
        entered = true;
        if (!preferences.enabled) { update(); return; }
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) throw new Error('Audio unavailable');
            if (!context) {
                context = new AudioContext();
                master = context.createGain(); master.gain.value = 0; master.connect(context.destination);
                source = context.createBufferSource(); source.buffer = makeLoop(context); source.loop = true;
                source.connect(master); source.start();
                context.onstatechange = update;
            }
            await context.resume(); update();
        } catch {
            preferences.enabled = false; update();
            button.title = 'Click to retry background music';
        }
    }
    button.onclick = () => {
        preferences.enabled = !preferences.enabled; remember(); update();
        if (preferences.enabled) void start();
        else if (context) setTimeout(() => { if (!preferences.enabled) void context.suspend(); }, 300);
    };
    volume.oninput = () => { preferences.volume = Number(volume.value); remember(); update(); };
    window.JensenAmbience = { start, duck(value) { ducked = value; update(); } };
    document.addEventListener('pointerdown', () => { if (entered && preferences.enabled && context?.state === 'suspended') void start(); });
    update();
})();
