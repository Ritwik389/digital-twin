(() => {
    const $=id=>document.getElementById(id);
    const create=(tag,text)=>{const element=document.createElement(tag);if(text)element.textContent=text;return element;};
    async function request(url,body,method='GET') {
        const response=await fetch(url,{method,headers:body?{'Content-Type':'application/json'}:{},...(body?{body:JSON.stringify(body)}:{})});
        const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Could not save this change.');return data;
    }
    const visitor=()=>JSON.parse(localStorage.getItem('jensen_visitor_id'));
    function drawFacts(facts) {
        $('memory-facts').replaceChildren();
        if(!facts.length)$('memory-facts').append(create('p','No saved memories yet.'));
        for(const item of facts) {
            const row=create('form');row.className='fact-row';
            const input=create('textarea');input.value=item.fact;input.maxLength=1000;input.required=true;input.setAttribute('aria-label','Remembered fact');
            const category=create('select');category.setAttribute('aria-label','Memory category');
            for(const name of ['background','interest','expertise','project','preference','general'])category.append(new Option(name,name));category.value=item.category;
            const save=create('button','Save correction');save.className='map-action';save.type='submit';
            const forget=create('button','Forget');forget.className='map-action';forget.type='button';
            row.append(input,category,save,forget);
            row.onsubmit=async event=>{
                event.preventDefault();save.disabled=true;
                try {const data=await request('/api/memory/fact',{user_id:visitor(),old_fact:item.fact,fact:input.value.trim(),category:category.value,importance:item.importance || 2},'PATCH');drawFacts(data.memories);$('facts-status').textContent='Memory corrected.';}
                catch(error){$('facts-status').textContent=error.message;save.disabled=false;}
            };
            forget.onclick=async()=>{
                if(!confirm('Forget this saved fact?'))return;
                forget.disabled=true;
                try {const data=await request('/api/memory/fact',{user_id:visitor(),fact:item.fact},'DELETE');drawFacts(data.memories);$('facts-status').textContent='Fact forgotten. Its exact wording will not be learned again from the old chat.';}
                catch(error){$('facts-status').textContent=error.message;forget.disabled=false;}
            };
            $('memory-facts').append(row);
        }
    }
    $('btn-manage-memory').onclick=async()=>{
        $('facts-dialog').showModal();$('facts-status').textContent='Loading saved facts…';
        try {const data=await request('/api/memory/'+encodeURIComponent(visitor()));drawFacts(data.memories);$('facts-status').textContent='Inspect a fact, correct it, or forget it.';}
        catch(error){$('facts-status').textContent=error.message;}
    };
    $('btn-close-facts').onclick=()=>$('facts-dialog').close();
    let timeline=[];
    $('btn-timeline').onclick=async()=>{
        $('timeline-dialog').showModal();$('timeline-status').textContent='Loading dated sources…';
        try {
            timeline=(await request('/api/timeline')).years;
            $('timeline-years').replaceChildren();$('compare-from').replaceChildren();$('compare-to').replaceChildren();
            for(const item of timeline) {
                const card=create('button');card.className='year-card';
                card.append(create('strong',String(item.year)),create('span',`${item.passages} passages · ${item.sources.length} sources`),create('small',item.topics.map(t=>typeof t==='string'?t:t.label).join(' · ')));
                card.onclick=()=>{ $('compare-from').value=item.year; $('timeline-status').textContent=`${item.year}: ${item.sources.join(', ')}`; };
                $('timeline-years').append(card);$('compare-from').append(new Option(item.year,item.year));$('compare-to').append(new Option(item.year,item.year));
            }
            if(timeline.length)$('compare-to').value=timeline.at(-1).year;
            $('timeline-status').textContent='Choose two years and a topic to compare the actual source coverage.';
        }catch(error){$('timeline-status').textContent=error.message;}
    };
    $('btn-close-timeline').onclick=()=>$('timeline-dialog').close();
    $('btn-compare-years').onclick=()=>{
        const years=[Number($('compare-from').value),Number($('compare-to').value)],topic=$('compare-topic').value.trim() || 'accelerated computing';
        if(years[0]===years[1]){$('timeline-status').textContent='Choose two different years.';return;}
        if(window.JensenChat.isBusy()){$('timeline-status').textContent='Wait for the current reply, or stop it first.';return;}
        $('timeline-dialog').close();void window.JensenChat.ask(`Compare what the source collection says about ${topic} in ${years[0]} and ${years[1]}.`,null,{mode:'compare',compare_years:years});
    };
})();
