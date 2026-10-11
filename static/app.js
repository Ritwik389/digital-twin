'use strict';
const $ = id => document.getElementById(id);
const chatHistory = $('chat-history'), userInput = $('user-input'), btnSend = $('btn-send');
const btnReplay = $('btn-replay'), btnStop = $('btn-stop'), memoryBox = $('memory-box');
const audio = new Audio();
let activeUserName = 'Guest', sending = false, pendingSessionId = null, voiceJob = 0, graphTimer, chatController, lastReply;
const storage = {
    get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { showStatus('This browser cannot save chat history.'); } }
};
let activeUserId = storage.get('jensen_visitor_id', null);
if (typeof activeUserId !== 'string' || !activeUserId.startsWith('visitor_')) {
    activeUserId = `visitor_${crypto.randomUUID()}`;
    storage.set('jensen_visitor_id', activeUserId);
}
let sessions = storage.get('jensen_sessions', {});
if (!sessions || typeof sessions !== 'object' || Array.isArray(sessions)) sessions = {};
for (const key of Object.keys(sessions)) {
    if (!sessions[key] || !Array.isArray(sessions[key].messages)) delete sessions[key];
}
let currentSessionId = storage.get('jensen_current_session', null);
if (!sessions[currentSessionId]) newSession();
function newSession() {
    currentSessionId = `session_${Date.now()}_${crypto.randomUUID()}`;
    sessions[currentSessionId] = { title: 'New Conversation', messages: [] };
}
function save() { storage.set('jensen_sessions', sessions); storage.set('jensen_current_session', currentSessionId); }
function showStatus(text) { $('app-status').textContent = text; $('app-status').style.display = text ? 'block' : 'none'; }
function sessionPayload(id = currentSessionId) { return { user_id: activeUserId, session_id: id }; }
async function api(url, body, method = body ? 'POST' : 'GET') {
    const response = await fetch(url, {
        method, headers: body ? { 'Content-Type': 'application/json' } : {},
        ...(body ? { body: JSON.stringify(body) } : {})
    });
    let data;
    try { data = await response.json(); } catch { throw new Error('The server could not respond. Please retry.'); }
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
    return data;
}
function stopAudio() { voiceJob++; window.JensenSpeech.stop(); audio.pause(); audio.currentTime = 0; btnStop.style.display = 'none'; }
async function speakMessage(msg, button, sessionId = currentSessionId) {
    stopAudio(); lastReply=msg;
    if(sessionId!==currentSessionId)return;
    try { await window.JensenSpeech.unlock(); window.JensenSpeech.speak(msg); }
    catch(error) { showStatus(error.message); }
}
window.addEventListener('jensen-voice-state',event=>{ btnStop.style.display=event.detail?'flex':'none'; $('voice-state').textContent=event.detail; });
window.addEventListener('jensen-voice-error',event=>showStatus(event.detail));
window.addEventListener('jensen-audio-ready',()=>{ save(); btnReplay.style.display='flex'; });
function openCitation(citation) {
    $('citation-title').textContent=citation.title;
    $('citation-meta').textContent=`${citation.year || 'Undated'} · ${citation.filename}${citation.start_seconds!==null?' · '+citation.start_seconds+'s':''}`;
    $('citation-quote').textContent=citation.quote;
    const link=$('citation-link'); link.hidden=!citation.url; if(citation.url)link.href=citation.url;
    $('citation-note').textContent=[citation.date_note,citation.start_seconds===null?'This corpus passage has no timestamp metadata.':''].filter(Boolean).join(' ');
    $('citation-dialog').showModal();
}
function citationButton(citation, text=`[${citation.id}]`) {
    const button=document.createElement('button');button.className='citation-button';button.textContent=text;
    button.title=citation.title;button.onclick=()=>openCitation(citation);return button;
}
function replyContent(container,msg) {
    const citations=msg.citations || [], parts=msg.text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g,(_,numbers)=>numbers.match(/\d+/g).map(number=>`[${number}]`).join('')).split(/(\[\d+\])/);
    for(const part of parts) {
        const number=part.match(/^\[(\d+)\]$/), source=number && citations.find(c=>c.id===Number(number[1]));
        container.append(source?citationButton(source,part):document.createTextNode(part));
    }
    if(citations.length) {
        const sources=document.createElement('details'), summary=document.createElement('summary');summary.textContent=`Sources (${citations.length})`;
        sources.append(summary);
        for(const citation of citations)sources.append(citationButton(citation,`[${citation.id}] ${citation.year || ''} · ${citation.title}`));
        container.append(sources);
    }
    if(msg.long_term_memories_used?.length) {
        const used=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`Remembered context used (${msg.long_term_memories_used.length})`;
        used.append(summary);
        for(const fact of msg.long_term_memories_used){const item=document.createElement('p');item.textContent=fact;used.append(item);}
        container.append(used);
    }
    if(msg.graph_paths?.length) {
        const used=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`Graph paths used (${msg.graph_paths.length})`;
        used.append(summary);
        for(const path of msg.graph_paths){
            const line=document.createElement('p'); line.textContent=path.edges.map(edge=>{
                const from=path.nodes.find(n=>n.id===edge.source)?.label || edge.source;
                const to=path.nodes.find(n=>n.id===edge.target)?.label || edge.target;
                return `${from} → ${(edge.predicate || edge.label).replaceAll('_',' ')} → ${to}`;
            }).join(' · ');used.append(line);
        }
        container.append(used);
    }
    if(msg.retrieval_status==='unavailable') {
        const note=document.createElement('small');note.textContent='Source retrieval is temporarily unavailable. Retry shortly; era restrictions remain in effect.';container.append(note);
    } else if(msg.evidence_status==='no_matching_sources') {
        const note=document.createElement('small');note.textContent='No matching source passages in this era. Treat general advice separately from sourced history.';container.append(note);
    }
    if(msg.partial){const note=document.createElement('small');note.textContent='Reply stopped before completion.';container.append(note);}
}
function renderChat() {
    chatHistory.replaceChildren();
    btnReplay.style.display = 'none';
    for (const msg of sessions[currentSessionId].messages) {
        if (typeof msg.text !== 'string') continue;
        const div = document.createElement('div');
        div.className = msg.role === 'user' ? 'msg-user' : 'msg-jensen';
        const label = document.createElement('strong');
        label.textContent = msg.role === 'user' ? 'You: ' : (msg.role === 'error' ? 'System: ' : 'Jensen: ');
        div.append(label); if(msg.role==='jensen')replyContent(div,msg);else div.append(document.createTextNode(msg.text));
        if (msg.role === 'jensen' && msg.text && !msg.generating) {
            const play = document.createElement('button');
            play.className = 'inline-play-btn'; play.textContent = '🔊 Play';
            play.onclick = () => speakMessage(msg, play);
            div.append(document.createElement('br'), play);
        }
        chatHistory.append(div);
    }
    if (sending && pendingSessionId === currentSessionId) {
        const pending = document.createElement('div');
        pending.className = 'msg-jensen pending-reply'; pending.id = 'pending-reply';
        pending.setAttribute('role', 'status'); pending.setAttribute('aria-live', 'polite');
        const dots = document.createElement('span'); dots.className = 'typing-indicator'; dots.setAttribute('aria-hidden', 'true');
        for (let i = 0; i < 3; i++) { const dot = document.createElement('span'); dot.className = 'typing-dot'; dots.append(dot); }
        pending.append(dots, document.createTextNode('Jensen is thinking…')); chatHistory.append(pending);
    }
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
function renderSessions() {
    $('session-list').replaceChildren();
    for (const id of Object.keys(sessions).reverse()) {
        const button = document.createElement('button');
        button.className = 'sidebar-btn'; button.textContent = sessions[id].title || 'Conversation';
        button.style.color = id === currentSessionId ? '#76b900' : 'white';
        button.onclick = () => { stopAudio(); currentSessionId = id; save(); renderSessions(); renderChat(); };
        $('session-list').append(button);
    }
}
function memoryStatus(data) {
    if (!data) return;
    const interval = data.update_every || 5;
    $('memory-status').textContent = !data.configured
        ? `Knowledge graph awaits a Neo4j connection. Updates every ${interval} turns once connected.`
        : data.error || `${data.pending_turns || 0} turns pending · Updates every ${interval} completed turns.`;
}
async function greet() {
    const sessionId = currentSessionId;
    try {
        const data = await api('/api/greet', { name: activeUserName });
        const msg = { role: 'jensen', text: data.text };
        if (!sessions[sessionId]) return;
        sessions[sessionId].messages.push(msg); save();
        if (currentSessionId === sessionId) {
            renderChat();
            if ($('voice-toggle').checked) void speakMessage(msg, null, sessionId);
        }
    } catch (error) { showStatus(error.message); }
}
$('btn-enter').onclick = () => {
    void window.JensenAmbience?.start(); void window.JensenSpeech.unlock().catch(()=>{});
    activeUserName = $('landing-name').value.trim() || 'Guest';
    $('entrance-overlay').style.display = 'none';
    const demo=new URLSearchParams(location.search).get('demo');
    const demos={cuda:'Explain CUDA and why software is part of NVIDIA’s moat.',twins:'Explain how Omniverse and digital twins connect to robotics.',pitch:'Review my startup pitch: a GPU simulation platform for small robotics teams.'};
    if(demos[demo])void sendMessage(demos[demo],null,{mode:demo==='pitch'?'pitch':'beginner'});
    else if (!sessions[currentSessionId].messages.length) void greet();
    void api('/api/graph/' + encodeURIComponent(activeUserId)).then(memoryStatus).catch(error => showStatus(error.message));
};
$('btn-new-chat').onclick = () => { stopAudio(); newSession(); save(); renderSessions(); renderChat(); void greet(); };
$('btn-delete-chat').onclick = () => {
    if (!confirm('Delete this chat from this browser? Saved memory is managed separately.')) return;
    stopAudio(); delete sessions[currentSessionId]; currentSessionId = Object.keys(sessions)[0];
    if (!currentSessionId) newSession();
    save(); renderSessions(); renderChat();
};
$('btn-wipe-all').onclick = () => {
    if (!confirm('Delete all chat sessions from this browser? Saved memory is managed separately.')) return;
    stopAudio(); sessions = {}; newSession(); save(); renderSessions(); renderChat();
};
$('btn-extract-mem').onclick = async () => {
    memoryBox.textContent = 'Saving memory…';
    try { const data = await api('/api/session/save', sessionPayload()); memoryBox.textContent = data.status; memoryStatus(data.memory); if ($('graph-dialog').open) await loadGraph(); }
    catch (error) { memoryBox.textContent = error.message; }
};
$('btn-load-mem').onclick = async () => {
    memoryBox.textContent = 'Loading…';
    try { const data = await api('/api/memory/' + encodeURIComponent(activeUserId));
        memoryBox.textContent = data.memories?.length ? data.memories.map(m => `[${m.category}] ${m.fact}`).join('\n\n') : 'No memories saved yet.';
    } catch (error) { memoryBox.textContent = error.message; }
};
$('btn-clear-mem').onclick = async () => {
    if (!confirm('Clear your saved facts and knowledge graph?')) return;
    try { const data = await api('/api/memory/' + encodeURIComponent(activeUserId), null, 'DELETE');
        memoryBox.textContent = data.status;
        // New session prevents already-cleared chat history from being re-extracted.
        stopAudio(); newSession(); save(); renderSessions(); renderChat(); await loadGraph();
    } catch (error) { memoryBox.textContent = error.message; }
};
$('voice-toggle').onchange = () => { if (!$('voice-toggle').checked) stopAudio(); };
btnReplay.onclick = () => { if(lastReply)void speakMessage(lastReply); };
btnStop.onclick = stopAudio;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.onstart = () => { $('btn-mic').textContent = 'Listening…'; };
    recognition.onend = () => { $('btn-mic').textContent = '🎤 Dictate'; };
    recognition.onerror = () => { showStatus('Voice input is unavailable. You can type your question.'); };
    recognition.onresult = event => { userInput.value += ' ' + event.results[0][0].transcript; };
    $('btn-mic').onclick = () => { try { recognition.start(); } catch { /* already recording */ } };
} else $('btn-mic').style.display = 'none';
async function sendMessage(prompt, knowledgeNodeId = null, options = {}) {
    const text = (typeof prompt === 'string' ? prompt : userInput.value).trim();
    if (!text || sending) return null;
    stopAudio(); showStatus(''); sending = true; btnSend.disabled = true;
    chatController=new AbortController();$('btn-cancel-reply').hidden=false;
    const sessionId = currentSessionId;
    pendingSessionId = sessionId;
    sessions[sessionId].messages.push({ role: 'user', text });
    const msg={role:'jensen',text:'',generating:true};sessions[sessionId].messages.push(msg);lastReply=msg;
    if (sessions[sessionId].title === 'New Conversation') sessions[sessionId].title = text.slice(0, 35);
    renderChat(); renderSessions(); save(); if (typeof prompt !== 'string') userInput.value = ''; $('loading').style.display = 'block';
    if($('voice-toggle').checked)window.JensenSpeech.begin(msg);
    let result, lastRender=0;
    try {
        await window.JensenStream('/api/chat/stream', { ...sessionPayload(sessionId), message: text,
            era: options.compare_years?'all':$('era-select').value, mode:options.mode || $('mode-select').value,
            compare_years:options.compare_years || null, voice_enabled:false, knowledge_node_id:knowledgeNodeId },chatController.signal,event=>{
            if(event.type==='meta')Object.assign(msg,event);
            if(event.type==='delta') {
                msg.text+=event.text;
                if($('voice-toggle').checked && currentSessionId===sessionId)window.JensenSpeech.append(event.text);
                if(options.onDelta)options.onDelta(msg.text);
                if(Date.now()-lastRender>80 && currentSessionId===sessionId){renderChat();lastRender=Date.now();}
            }
            if(event.type==='done') {
                result=event;Object.assign(msg,event,{text:event.answer,generating:false});
                window.JensenSpeech.append('',true);
            }
        });
        if (!sessions[sessionId]) return result;
        if (currentSessionId === sessionId) {
            $('chunk-badge').textContent = `${result.query_type || 'chat'} | ${result.chunks_used || 0} chunks | ${result.graph_paths?.length || 0} graph paths`;
            $('sources-box').textContent = result.sources?.join('\n') || 'No sources retrieved.';
            void api('/api/graph/' + encodeURIComponent(activeUserId)).then(memoryStatus).catch(()=>{});
        }
        if ($('graph-dialog').open) void loadGraph();
        return result;
    } catch (error) {
        stopAudio();msg.generating=false;msg.partial=true;
        const message=error.name==='AbortError'?'Reply stopped.':error.message;
        if (sessions[sessionId]) sessions[sessionId].messages.push({ role: 'error', text: message });
        if (currentSessionId === sessionId && typeof prompt !== 'string') userInput.value = text;
        return { error: message };
    } finally { sending = false; pendingSessionId = null; btnSend.disabled = false; $('btn-cancel-reply').hidden=true; $('loading').style.display = 'none'; renderChat(); save(); }
}
$('btn-cancel-reply').onclick=()=>{chatController?.abort();stopAudio();};
$('btn-close-citation').onclick=()=>$('citation-dialog').close();
window.JensenChat = { ask: sendMessage, isBusy: () => sending };
btnSend.onclick = sendMessage;
userInput.onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendMessage(); } };
$('date-display').textContent = new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
renderSessions(); renderChat(); save();

// Native SVG graph: no third-party script or graph data leaves this app.
const svg = $('memory-graph'), dialog = $('graph-dialog'), detail = $('graph-detail');
const SVG_NS = 'http://www.w3.org/2000/svg';
const colors = { Memory: '#76b900', Person: '#b8e986', Topic: '#79bdff', Project: '#f3bf6b', Skill: '#b7a1ff', Organization: '#ee90af', Fact: '#dce7d4' };
function shape(tag, attrs, text) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text) element.textContent = text;
    return element;
}
function drawGraph(data) {
    svg.replaceChildren();
    detail.textContent = 'Click a node or connection to see its details. Scroll to zoom; drag the background to pan.';
    const positions = new Map();
    const root = data.nodes.find(n => n.kind === 'Memory');
    const rest = data.nodes.filter(n => n !== root);
    if (root) positions.set(root.id, { x: 600, y: 425 });
    rest.forEach((node, i) => {
        const ring = Math.floor(i / 24), start = ring * 24, count = Math.min(24, rest.length - start);
        const angle = -Math.PI / 2 + (i - start) * 2 * Math.PI / Math.max(count, 1);
        const radius = 210 + ring * 180;
        positions.set(node.id, { x: 600 + Math.cos(angle) * radius, y: 425 + Math.sin(angle) * radius });
    });
    for (const edge of data.edges) {
        const a = positions.get(edge.source), b = positions.get(edge.target);
        if (!a || !b) continue;
        const line = shape('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: '#41613b', 'stroke-width': 2 });
        line.append(shape('title', {}, `${edge.label}${edge.evidence ? ': ' + edge.evidence : ''}`));
        line.style.cursor = 'pointer'; line.onclick = () => { detail.textContent = `${edge.label.replaceAll('_', ' ')}\n\n${edge.evidence || 'Saved memory connection.'}`; };
        svg.append(line);
    }
    for (const node of data.nodes) {
        const point = positions.get(node.id); if (!point) continue;
        const group = shape('g', { transform: `translate(${point.x} ${point.y})`, tabindex: 0, role: 'button', 'aria-label': `${node.kind}: ${node.label}` });
        group.style.cursor = 'pointer';
        group.append(shape('circle', { r: node.kind === 'Memory' ? 38 : 20, fill: '#111b11', stroke: colors[node.kind] || '#aaa', 'stroke-width': 3 }));
        const label = node.label.length > 23 ? node.label.slice(0, 21) + '…' : node.label;
        group.append(shape('text', { y: node.kind === 'Memory' ? 58 : 38, 'text-anchor': 'middle', fill: '#eef3e9', 'font-size': 13 }, label));
        group.append(shape('title', {}, `${node.kind}: ${node.label}`));
        const select = () => {
            const links = data.edges.filter(e => e.source === node.id || e.target === node.id);
            detail.textContent = `${node.label}\n${node.kind}${node.category ? ' · ' + node.category : ''}\n\n` + links.map(e => {
                const other = data.nodes.find(n => n.id === (e.source === node.id ? e.target : e.source));
                const direction = e.source === node.id ? '→' : '←';
                return `${direction} ${e.label.replaceAll('_', ' ')}: ${other?.label || ''}${e.evidence ? '\n“' + e.evidence + '”' : ''}`;
            }).join('\n\n');
        };
        group.onclick = select; group.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } };
        svg.append(group);
    }
    const radius = 280 + Math.floor(Math.max(0, rest.length - 1) / 24) * 180;
    svg.dataset.fit = `${600 - radius * 1.3} ${425 - radius} ${radius * 2.6} ${radius * 2}`;
    svg.setAttribute('viewBox', svg.dataset.fit);
}
let loadingGraph = false;
async function loadGraph() {
    if (loadingGraph) return;
    loadingGraph = true;
    try {
        const data = await api('/api/graph/' + encodeURIComponent(activeUserId));
        memoryStatus(data);
        if (!data.configured) {
            svg.replaceChildren(); $('graph-status').textContent = 'Neo4j connection needed. Add NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in the Space’s Secrets.';
            detail.textContent = 'Once connected, this view fills automatically after every five successful chat turns. Use “Update from chat now” to save a shorter conversation.';
        } else if (data.state === 'error') { $('graph-status').textContent = data.error; }
        else {
            $('graph-status').textContent = `${data.nodes.length} nodes · ${data.edges.length} connections · ${data.pending_turns} turns pending${data.error ? ' · ' + data.error : ''}`;
            if (data.nodes.length) drawGraph(data);
            else { svg.replaceChildren(); detail.textContent = 'No saved graph yet. Chat with Jensen or click Update from chat now.'; }
        }
    } catch (error) { $('graph-status').textContent = error.message; }
    finally { loadingGraph = false; }
}
$('btn-memory-graph').onclick = () => { dialog.showModal(); void loadGraph(); clearInterval(graphTimer); graphTimer = setInterval(loadGraph, 6000); };
$('btn-close-graph').onclick = () => dialog.close();
dialog.onclose = () => clearInterval(graphTimer);
$('btn-refresh-graph').onclick = loadGraph;
$('btn-update-graph').onclick = async () => {
    $('btn-update-graph').disabled = true; $('graph-status').textContent = 'Updating memory from chat…';
    try { const data = await api('/api/graph/update', sessionPayload()); memoryStatus(data); await loadGraph(); }
    catch (error) { $('graph-status').textContent = error.message; }
    finally { $('btn-update-graph').disabled = false; }
};
$('btn-fit-graph').onclick = () => svg.setAttribute('viewBox', svg.dataset.fit || '0 0 1200 850');
svg.addEventListener('wheel', event => {
    event.preventDefault();
    const box = svg.getAttribute('viewBox').split(' ').map(Number), scale = event.deltaY > 0 ? 1.12 : .89;
    if (box[2] * scale < 200 || box[2] * scale > 6000) return;
    svg.setAttribute('viewBox', `${box[0] + box[2] * (1 - scale) / 2} ${box[1] + box[3] * (1 - scale) / 2} ${box[2] * scale} ${box[3] * scale}`);
}, { passive: false });
let drag;
svg.onpointerdown = event => { if (event.target !== svg) return; drag = { x: event.clientX, y: event.clientY, box: svg.getAttribute('viewBox').split(' ').map(Number) }; svg.setPointerCapture(event.pointerId); };
svg.onpointermove = event => { if (!drag) return; const rect = svg.getBoundingClientRect(); svg.setAttribute('viewBox', `${drag.box[0] - (event.clientX - drag.x) * drag.box[2] / rect.width} ${drag.box[1] - (event.clientY - drag.y) * drag.box[3] / rect.height} ${drag.box[2]} ${drag.box[3]}`); };
svg.onpointerup = svg.onpointercancel = () => { drag = null; };
