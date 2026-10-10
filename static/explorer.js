(() => {
    const $ = id => document.getElementById(id);
    const dialog = $('map-dialog'), svg = $('jensen-map'), detail = $('map-detail');
    const search = $('map-search'), category = $('map-category'), status = $('map-status');
    const NS = 'http://www.w3.org/2000/svg';
    const palette = ['#aedf73', '#70c8ea', '#c7a7ff', '#efbb77', '#76d2aa', '#f399b7', '#ded083'];
    let data, selectedId, poll, loading = false, selectedReply = new Map();
    const html = (tag, text, className) => { const e = document.createElement(tag); if (text) e.textContent = text; if (className) e.className = className; return e; };
    function shape(tag, attrs, text) {
        const e = document.createElementNS(NS, tag);
        for (const [key, value] of Object.entries(attrs)) e.setAttribute(key, value);
        if (text) e.textContent = text;
        return e;
    }
    function loader() {
        const container = html('div', null, 'pending-reply');
        const dots = html('span', null, 'typing-indicator'); dots.setAttribute('aria-hidden', 'true');
        for (let i = 0; i < 3; i++) dots.append(html('span', null, 'typing-dot'));
        container.append(dots, html('span', 'Jensen is thinking…')); return container;
    }
    function connected(node) {
        const counts = new Map();
        for (const edge of data.edges) {
            if (['COVERS','IN_CATEGORY'].includes(edge.label) || (edge.source !== node.id && edge.target !== node.id)) continue;
            const id = edge.source === node.id ? edge.target : edge.source;
            counts.set(id, (counts.get(id) || 0) + edge.weight);
        }
        return [...counts].sort((a, b) => b[1] - a[1]).map(([id]) => data.nodes.find(n => n.id === id)).filter(Boolean).slice(0, 6);
    }
    function visibleNodes() {
        const entities = data.nodes.filter(n => n.kind !== 'Category' && n.id !== data.root_id);
        const term = search.value.trim().toLowerCase();
        let topics;
        if (term) topics = entities.filter(n => `${n.label} ${n.category} ${n.kind}`.toLowerCase().includes(term));
        else if (category.value) topics = entities.filter(n => n.category === category.value);
        else topics = data.categories.flatMap(c => entities.filter(n => n.category === c).sort((a, b) => b.mentions - a.mentions).slice(0, 4));
        if (selectedId) {
            const selected = data.nodes.find(n => n.id === selectedId);
            if (selected && (!category.value || selected.category === category.value) && !topics.includes(selected)) topics.push(selected);
        }
        const groups = data.nodes.filter(n => n.kind === 'Category' && (!category.value || n.category === category.value) && (!term || topics.some(t => t.category === n.category)));
        return [data.nodes.find(n => n.id === data.root_id), ...groups, ...topics].filter(Boolean);
    }
    function draw() {
        if (!data) return;
        svg.replaceChildren();
        const nodes = visibleNodes(), points = new Map(), root = data.root_id;
        const groups = nodes.filter(n => n.kind === 'Category');
        points.set(root, { x: 600, y: 450 });
        if (category.value || search.value.trim()) {
            groups.forEach((node, i) => {
                const angle = -Math.PI / 2 + i * 2 * Math.PI / groups.length;
                points.set(node.id, { x: 600 + Math.cos(angle) * 140, y: 450 + Math.sin(angle) * 140 });
            });
            const topics = nodes.filter(n => n.kind !== 'Category' && n.id !== root);
            topics.forEach((node, i) => {
                const ring = Math.floor(i / 14), count = Math.min(14, topics.length - ring * 14);
                const angle = -Math.PI / 2 + (i % 14) * 2 * Math.PI / Math.max(count, 1);
                const radius = 240 + ring * 160;
                points.set(node.id, { x: 600 + Math.cos(angle) * radius, y: 450 + Math.sin(angle) * radius });
            });
        } else {
            groups.forEach((group, index) => {
                const angle = -Math.PI / 2 + index * 2 * Math.PI / groups.length;
                points.set(group.id, { x: 600 + Math.cos(angle) * 200, y: 450 + Math.sin(angle) * 200 });
                const members = nodes.filter(n => n.kind !== 'Category' && n.id !== root && n.category === group.category);
                members.forEach((node, i) => {
                    const theta = angle + (i - (members.length - 1) / 2) * .19;
                    const radius = 350 + (i % 2) * 80;
                    points.set(node.id, { x: 600 + Math.cos(theta) * radius, y: 450 + Math.sin(theta) * radius });
                });
            });
            // Use the wide canvas so neighboring labels have room to breathe.
            for (const point of points.values()) point.x = 600 + (point.x - 600) * 1.65;
        }
        for (const edge of data.edges) {
            const a = points.get(edge.source), b = points.get(edge.target);
            if (!a || !b || ($('map-semantic').checked && edge.label==='CO_MENTIONED') || (edge.label === 'CO_MENTIONED' && edge.source !== selectedId && edge.target !== selectedId)) continue;
            const line = shape('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y,
                stroke: edge.label === 'CO_MENTIONED' ? '#a5ce82' : ['COVERS','IN_CATEGORY'].includes(edge.label)?'#39533f':'#ffc76b', 'stroke-width': ['COVERS','IN_CATEGORY'].includes(edge.label)?1.4:3 });
            line.append(shape('title', {}, edge.label.replaceAll('_',' ')));
            if(edge.evidence.length) {
                line.setAttribute('class','map-edge');line.setAttribute('role','button');line.setAttribute('tabindex','0');
                line.setAttribute('aria-label',`Inspect ${data.nodes.find(n=>n.id===edge.source)?.label} ${edge.label.replaceAll('_',' ')} ${data.nodes.find(n=>n.id===edge.target)?.label}`);
                const inspect=()=>{
                    selectedId=null;
                    detail.replaceChildren(html('h3',edge.label.replaceAll('_',' ')),html('p',`${data.nodes.find(n=>n.id===edge.source)?.label} → ${data.nodes.find(n=>n.id===edge.target)?.label}`));
                    if(edge.label==='CO_MENTIONED')detail.append(html('p','These topics appear together; this does not establish a causal or business relationship.'));
                    for(const evidence of edge.evidence)appendEvidence(evidence);
                };
                line.style.cursor='pointer';line.onclick=inspect;
                line.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect();}};
            }
            svg.append(line);
        }
        for (const node of nodes) {
            const point = points.get(node.id); if (!point) continue;
            const isRoot = node.id === root, isGroup = node.kind === 'Category';
            const color = palette[data.categories.indexOf(node.category) % palette.length];
            const group = shape('g', { transform: `translate(${point.x} ${point.y})`, class: 'map-node', tabindex: 0,
                role: 'button', 'aria-label': `${isGroup ? 'Explore category' : 'Ask Jensen about'} ${node.label}`, 'data-node-id': node.id });
            group.style.cursor = 'pointer';
            group.append(shape('circle', { r: isRoot ? 42 : isGroup ? 25 : 22, fill: '#142119', stroke: node.id === selectedId ? '#fff' : color, 'stroke-width': isRoot ? 4 : 2.5 }));
            const label = shape('text', { y: isRoot ? 70 : isGroup ? 52 : 47, 'text-anchor': 'middle', fill: color, 'font-size': isRoot ? 28 : isGroup ? 23 : 25, 'font-weight': isRoot || isGroup ? 600 : 400 });
            const words = node.label.split(' '), lines = [''];
            for (const word of words) {
                if ((lines.at(-1) + word).length > 19 && lines.at(-1)) lines.push('');
                lines[lines.length - 1] += (lines.at(-1) ? ' ' : '') + word;
            }
            lines.forEach((line, i) => label.append(shape('tspan', { x: 0, dy: i ? 27 : 0 }, line)));
            group.append(label, shape('title', {}, `${node.label} · ${isGroup ? 'Click to expand' : node.mentions + ' source passages'}`));
            const activate = () => {
                if (isGroup) { category.value = node.category; search.value = ''; selectedId = null; draw(); fit(); }
                else void selectNode(node, true);
            };
            group.onclick = activate;
            group.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); } };
            svg.append(group);
        }
        const all = [...points.values()];
        const x = Math.min(...all.map(p => p.x)) - 130, y = Math.min(...all.map(p => p.y)) - 60;
        const width = Math.max(...all.map(p => p.x)) - x + 130, height = Math.max(...all.map(p => p.y)) - y + 100;
        svg.dataset.fit = `${x} ${y} ${Math.max(650, width)} ${Math.max(600, height)}`;
        if (!svg.dataset.initialized) { fit(); svg.dataset.initialized = 'true'; }
        const total = data.nodes.filter(n => n.kind !== 'Category').length;
        status.textContent = `${nodes.filter(n => n.kind !== 'Category').length} of ${total} topics · ${data.source_count} sources · ${data.document_count} passages · ${data.semantic_relation_count || 0} extracted relationships · ${data.relation_state || 'waiting'}`;
    }
    function fit() { svg.setAttribute('viewBox', svg.dataset.fit || '0 0 1200 900'); }
    function appendEvidence(evidence) {
        detail.append(html('blockquote',evidence.quote),html('div',`${evidence.title || evidence.source} · ${evidence.year || 'Undated'} · ${evidence.filename}`,'map-meta'));
        if(evidence.url) {
            const link=html('a','Original source ↗','map-meta');link.href=evidence.url;link.target='_blank';link.rel='noopener noreferrer';detail.append(link);
        }
    }
    async function selectNode(node, ask) {
        selectedId = node.id; draw(); detail.replaceChildren();
        detail.append(html('h3', node.label), html('div', `${node.kind} · ${node.mentions} passages · ${node.source_count || 0} sources`, 'map-meta'));
        const askButton = html('button', 'Ask about this topic', 'map-action');
        askButton.onclick = () => void selectNode(node, true); detail.append(askButton);
        detail.append(html('h4', 'Jensen’s explanation'));
        const reply = html('div', selectedReply.get(node.id) || 'Click “Ask about this topic” for an explanation.', 'map-reply');
        reply.dataset.nodeId = node.id; detail.append(reply);
        const continueButton = html('button', 'Continue in chat →', 'map-action');
        continueButton.onclick = () => { dialog.close(); $('user-input').focus(); }; detail.append(continueButton);
        const related = connected(node);
        if (related.length) {
            detail.append(html('h4', 'Connected topics'));
            const chips = html('div', null, 'map-related');
            for (const other of related) { const button = html('button', other.label); button.onclick = () => void selectNode(other, true); chips.append(button); }
            detail.append(chips);
        }
        detail.append(html('h4', 'From the source collection'));
        for (const evidence of node.evidence || []) {
            appendEvidence(evidence);
        }
        if (!ask) return;
        if (window.JensenChat.isBusy()) { reply.textContent = 'Jensen is finishing the previous reply. Click “Ask about this topic” afterward.'; return; }
        askButton.disabled = true; reply.replaceChildren(loader());
        const result = await window.JensenChat.ask(node.query, node.id,{onDelta:text=>{if(selectedId===node.id)reply.textContent=text;}});
        if (result?.answer) selectedReply.set(node.id, result.answer);
        if (selectedId === node.id) {
            detail.querySelector('.map-reply').textContent = result?.answer || result?.error || 'Please try this topic again.';
            detail.querySelector('.map-action').disabled = false;
        }
    }
    async function load() {
        if (loading) return;
        loading = true;
        try {
            const response = await fetch('/api/knowledge-map');
            if (!response.ok) throw new Error('The source map could not load. Try Refresh.');
            const snapshot = await response.json();
            if (snapshot.state !== 'ready') {
                status.textContent = snapshot.error || 'Preparing the source map…';
                if (!poll && dialog.open) poll = setInterval(load, 3500);
                return;
            }
            if (poll && !['building','waiting'].includes(snapshot.relation_state)) { clearInterval(poll); poll = null; }
            if (!poll && dialog.open && ['building','waiting'].includes(snapshot.relation_state))poll=setInterval(load,6000);
            if (!data || data.fingerprint !== snapshot.fingerprint || data.semantic_relation_count!==snapshot.semantic_relation_count || data.relation_state!==snapshot.relation_state) {
                data = snapshot;
                category.replaceChildren(new Option('Overview · all categories', ''));
                for (const name of data.categories) category.append(new Option(name, name));
                draw();
            }
        } catch (error) { status.textContent = error.message; }
        finally { loading = false; }
    }
    $('btn-jensen-map').onclick = () => { dialog.showModal(); void load(); };
    $('btn-close-map').onclick = () => dialog.close();
    $('btn-map-refresh').onclick = load;
    $('btn-map-reset').onclick = fit;
    dialog.addEventListener('close', () => { clearInterval(poll); poll = null; });
    $('map-semantic').onchange=()=>draw();
    search.oninput = category.onchange = () => { selectedId = null; draw(); fit(); };
    svg.addEventListener('wheel', event => {
        event.preventDefault();
        const box = svg.getAttribute('viewBox').split(' ').map(Number), scale = event.deltaY > 0 ? 1.12 : .89;
        if (box[2] * scale < 200 || box[2] * scale > 6500) return;
        svg.setAttribute('viewBox', `${box[0] + box[2] * (1 - scale) / 2} ${box[1] + box[3] * (1 - scale) / 2} ${box[2] * scale} ${box[3] * scale}`);
    }, { passive: false });
    let drag;
    svg.onpointerdown = event => {
        if (event.target.closest('.map-node, .map-edge')) return;
        drag = { x: event.clientX, y: event.clientY, box: svg.getAttribute('viewBox').split(' ').map(Number) };
        svg.setPointerCapture(event.pointerId);
    };
    svg.onpointermove = event => {
        if (!drag) return;
        const rect = svg.getBoundingClientRect();
        svg.setAttribute('viewBox', `${drag.box[0] - (event.clientX - drag.x) * drag.box[2] / rect.width} ${drag.box[1] - (event.clientY - drag.y) * drag.box[3] / rect.height} ${drag.box[2]} ${drag.box[3]}`);
    };
    svg.onpointerup = svg.onpointercancel = () => { drag = null; };
    window.JensenMap={open:()=>{dialog.showModal();void load();}};
    // Preload the already-built map before it is opened, with no chat required.
    void load();
})();
