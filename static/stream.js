'use strict';
// Both endpoints use newline-delimited JSON; a network chunk is not an event.
window.JensenStream = async function(url, payload, signal, onEvent) {
    const response = await fetch(url, {method: 'POST', headers: {'Content-Type':'application/json'}, body:JSON.stringify(payload), signal});
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not complete. Please retry.');
    }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = '', done = false;
    const consume = line => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'error') throw new Error(event.error || 'Generation failed. Please retry.');
        if (event.type === 'done') done = true;
        onEvent(event);
    };
    try {
        for (;;) {
            const chunk = await reader.read();
            buffer += decoder.decode(chunk.value || new Uint8Array(), {stream:!chunk.done});
            let index;
            while ((index = buffer.indexOf('\n')) >= 0) { consume(buffer.slice(0,index)); buffer = buffer.slice(index+1); }
            if (chunk.done) break;
        }
        consume(buffer);
        if (!done) throw new Error('The connection ended before generation finished. Please retry.');
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
};

window.JensenSpeech = (() => {
    let context, controller, jobId, epoch = 0, nextTime = 0, processing = false, pendingText = '', currentReply;
    const queue = [], sources = new Set();
    const state = text => window.dispatchEvent(new CustomEvent('jensen-voice-state',{detail:text}));
    async function unlock() {
        context ||= new (window.AudioContext || window.webkitAudioContext)();
        if (context.state !== 'running') await context.resume();
    }
    function stop() {
        epoch++; queue.length = 0; pendingText = ''; currentReply = null;
        controller?.abort(); controller = null;
        if (jobId) void fetch('/api/speak/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:jobId})}).catch(()=>{});
        jobId = null;
        for (const source of sources) { try { source.stop(); } catch {} }
        sources.clear(); nextTime = context?.currentTime || 0;
        window.JensenAmbience?.duck(false); state('');
    }
    function playPCM(event) {
        const bytes = Uint8Array.from(atob(event.pcm), c => c.charCodeAt(0)), view = new DataView(bytes.buffer);
        const samples = bytes.length / 2, buffer = context.createBuffer(1,samples,event.sample_rate);
        const values = buffer.getChannelData(0);
        for (let i=0;i<samples;i++) values[i] = view.getInt16(i*2,true)/32768;
        const source = context.createBufferSource(); source.buffer=buffer; source.connect(context.destination);
        nextTime=Math.max(nextTime,context.currentTime+.06); sources.add(source); source.start(nextTime); nextTime += buffer.duration;
        window.JensenAmbience?.duck(true); state('Speaking…');
        source.onended = () => { sources.delete(source); if (!sources.size) { window.JensenAmbience?.duck(false); if (!processing && !queue.length) state(''); } };
    }
    async function pump() {
        if (processing) return;
        processing = true;
        let activeVersion = epoch;
        try {
            while(queue.length) {
                const {text,reply,version} = queue.shift();
                if (version !== epoch) continue;
                activeVersion=version;
                await unlock(); if (version !== epoch) continue;
                controller = new AbortController(); jobId = crypto.randomUUID(); state('Voice queued…');
                await window.JensenStream('/api/speak/stream',{text,request_id:jobId},controller.signal,event => {
                    if(version!==epoch)return;
                    if(event.type==='audio') playPCM(event);
                    if(event.type==='status')state(event.status==='queued'?'Voice queued…':'Preparing voice…');
                    if(event.type==='done' && event.audio_file) {
                        reply.audioUrls ||= []; reply.audioUrls.push('/api/audio/'+encodeURIComponent(event.audio_file));
                        window.dispatchEvent(new CustomEvent('jensen-audio-ready',{detail:reply}));
                    }
                });
                controller=null; jobId=null;
                // Bound buffered sound while allowing sentence synthesis to overlap playback.
                while(version===epoch && context && nextTime-context.currentTime>12) await new Promise(resolve=>setTimeout(resolve,200));
            }
        } catch(error) {
            if(error.name!=='AbortError')window.dispatchEvent(new CustomEvent('jensen-voice-error',{detail:error.message}));
            if(activeVersion===epoch && error.name!=='AbortError') { queue.length=0; pendingText=''; currentReply=null; }
        } finally {processing=false; if(!sources.size)state(''); if(queue.length)void pump();}
    }
    function begin(reply) { stop(); currentReply=reply; }
    function append(text, final=false) {
        if (!currentReply) return;
        pendingText += text;
        for (;;) {
            const match = pendingText.match(/^([\s\S]*?[.!?](?:\s*\[\d+\])*(?:\s|$))/);
            let sentence;
            if (match && (match[1].length>=45 || final)) sentence=match[1];
            else if (pendingText.length>400) { let split=pendingText.lastIndexOf(' ',300); sentence=pendingText.slice(0,split>0?split:300); }
            else if (final && pendingText.trim())sentence=pendingText;
            else break;
            pendingText=pendingText.slice(sentence.length);
            const clean=sentence.replace(/\[[^\]\n]*\d[^\]\n]*\]/g,'').replace(/[*#`]/g,'').trim();
            if(clean)queue.push({text:clean,reply:currentReply,version:epoch});
        }
        void pump();
    }
    function speak(reply) { begin(reply); append(reply.text,true); }
    return {unlock,stop,begin,append,speak};
})();
