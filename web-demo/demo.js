'use strict';
// These authored event sequences illustrate the paper's controller, not model inference.
const scenarios={turn:{events:['REJECT','INCOMPLETE','INCOMPLETE','WAIT','INCOMPLETE','COMPLETE','REJECT'],note:'A valid utterance remains in progress until COMPLETE triggers a response.'},pause:{events:['INCOMPLETE','INCOMPLETE','WAIT','WAIT','INCOMPLETE','COMPLETE'],note:'WAIT preserves the receiving state: a brief pause does not end the user’s turn.'},barge:{events:['INCOMPLETE','COMPLETE','REJECT','WAIT','BARGE','INCOMPLETE','COMPLETE'],note:'Perception stays active during a response. BARGE stops output and returns to receiving.'}};
const definitions={REJECT:['Waiting for a signed utterance','No valid signed utterance requiring a response.'],INCOMPLETE:['The signed utterance continues','A valid utterance is still in progress.'],WAIT:['A pause within the utterance','Brief deceleration or a posture hold; the user has not finished.'],COMPLETE:['The utterance is complete','Submit the segment and begin the downstream response.'],BARGE:['A new utterance interrupts','New valid signing begins while the system produces output.']};
let currentScenario='turn',step=0,state='RECEIVING',timer=null,ticks=0;
const byId=(id)=>document.getElementById(id);
const eventButtons=[...document.querySelectorAll('[data-event]')];
const scenarioButtons=[...document.querySelectorAll('[data-scenario]')];
function stop(){if(timer){clearInterval(timer);timer=null;}byId('play').textContent='Play scenario';}
function renderEvent(event){
 const previous=state;
 if(state==='RECEIVING'&&event==='COMPLETE')state='RESPONDING';
 else if(state==='RESPONDING'&&event==='BARGE')state='RECEIVING';
 const responding=state==='RESPONDING';
 eventButtons.forEach(b=>{const selected=b.dataset.event===event;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});
 byId('current-event').textContent=event;
 byId('input-title').textContent=definitions[event][0];byId('input-detail').textContent=definitions[event][1];
 if(event==='BARGE'&&previous!=='RESPONDING')byId('input-detail').textContent='BARGE applies during a system response. The system is currently receiving.';
 byId('input-signal').className='signal '+({INCOMPLETE:'moving',WAIT:'paused',COMPLETE:'complete',BARGE:'barge'}[event]||'');
 byId('state-pill').textContent=state;byId('state-pill').classList.toggle('responding',responding);
 byId('state-center').classList.toggle('responding',responding);
 byId('state-title').textContent=responding?'Responding':'Receiving';
 byId('state-subtitle').textContent=responding?'Generate system output':'Continue observing';
 byId('response-flow').classList.toggle('active',responding);
 byId('action-detail').textContent=event==='BARGE'&&previous==='RESPONDING'?'Output stopped. Return to receiving.':responding?'SLT → LLM → TTS. Visual perception stays active.':'The system keeps receiving visual input.';
 byId('announcement').textContent=event+'. System state: '+state+'. '+byId('action-detail').textContent;
}
function renderProgress(){const sequence=scenarios[currentScenario].events;const value=Math.round(step/(sequence.length-1)*100);byId('progress').style.width=value+'%';document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow',String(value));byId('step-count').textContent=(step+1)+' / '+sequence.length;byId('elapsed').textContent=String(Math.floor(ticks/60)).padStart(2,'0')+':'+String(ticks%60).padStart(2,'0');}
function reset(){stop();step=0;ticks=0;state='RECEIVING';renderEvent(scenarios[currentScenario].events[0]);renderProgress();byId('scenario-note').textContent=scenarios[currentScenario].note;}
scenarioButtons.forEach(b=>b.addEventListener('click',()=>{currentScenario=b.dataset.scenario;scenarioButtons.forEach(t=>{const active=t===b;t.classList.toggle('active',active);t.setAttribute('aria-pressed',String(active));});reset();}));
eventButtons.forEach(b=>b.addEventListener('click',()=>{stop();renderEvent(b.dataset.event);}));
byId('play').addEventListener('click',()=>{if(timer){stop();return;}if(step===scenarios[currentScenario].events.length-1)reset();byId('play').textContent='Pause scenario';timer=setInterval(()=>{ticks++;if(ticks%2===0){step++;renderEvent(scenarios[currentScenario].events[step]);renderProgress();if(step===scenarios[currentScenario].events.length-1)stop();}else renderProgress();},1000);});
byId('reset').addEventListener('click',reset);
[...byId('input-signal').children].forEach((bar,i)=>bar.style.setProperty('--height',(18+((i*17+13)%44))+'px'));
reset();
