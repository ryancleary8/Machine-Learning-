// Front-end logic for Draw & Compare Digits (Web)
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const brush = document.getElementById('brush');
const clearBtn = document.getElementById('clearBtn');
const invertBtn = document.getElementById('invertBtn');
const predictBtn = document.getElementById('predictBtn');
const modelSel = document.getElementById('model');
const predEl = document.getElementById('pred');
const probsEl = document.getElementById('probs');
const digitsDiv = document.getElementById('digitButtons');

// init canvas black
ctx.fillStyle = 'black';
ctx.fillRect(0,0,canvas.width, canvas.height);

let drawing = false;
let lastX = 0, lastY = 0;
let brushSize = parseInt(brush.value,10);

function startDraw(e){
  drawing = true;
  const {x,y} = pos(e);
  lastX = x; lastY = y;
  drawDot(x,y);
}
function draw(e){
  if(!drawing) return;
  const {x,y} = pos(e);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = brushSize;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(lastX,lastY);
  ctx.lineTo(x,y);
  ctx.stroke();
  lastX = x; lastY = y;
}
function endDraw(e){ drawing = false; }
function drawDot(x,y){
  ctx.fillStyle = 'white';
  ctx.beginPath();
  ctx.arc(x,y, brushSize/2, 0, Math.PI*2);
  ctx.fill();
}
function pos(e){
  const rect = canvas.getBoundingClientRect();
  let cx = e.touches ? e.touches[0].clientX : e.clientX;
  let cy = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: cx - rect.left, y: cy - rect.top };
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e); }, {passive:false});
canvas.addEventListener('touchmove', e => { e.preventDefault(); draw(e); }, {passive:false});
canvas.addEventListener('touchend', endDraw);

brush.addEventListener('input', e => brushSize = parseInt(e.target.value,10));

clearBtn.addEventListener('click', ()=>{
  ctx.fillStyle='black'; ctx.fillRect(0,0,canvas.width,canvas.height);
  predEl.textContent = '—';
  probsEl.textContent = '';
});

invertBtn.addEventListener('click', ()=>{
  const imgData = ctx.getImageData(0,0,canvas.width,canvas.height);
  const d = imgData.data;
  for(let i=0;i<d.length;i+=4){
    d[i]   = 255 - d[i];
    d[i+1] = 255 - d[i+1];
    d[i+2] = 255 - d[i+2];
  }
  ctx.putImageData(imgData,0,0);
});

// Model selector
MODEL_NAMES.forEach((m,i)=>{
  const opt = document.createElement('option');
  opt.value = m; opt.textContent = m;
  modelSel.appendChild(opt);
});
modelSel.selectedIndex = 0;

// Prediction
async function predict(){
  const dataURL = canvas.toDataURL('image/png');
  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataURL, model: modelSel.value })
  });
  const js = await resp.json();
  if(js.error){
    predEl.textContent = 'Error';
    probsEl.textContent = js.error;
    return;
  }
  predEl.textContent = js.prediction;
  const lines = js.probs.map((p,i)=> `${i}: ${p.toFixed(3)}`).join('\n');
  probsEl.textContent = lines;
  lastPrediction = js.prediction;
}
predictBtn.addEventListener('click', predict);

// Score & accuracy (client-side)
let histories = {}; // model -> {accs:[], correct:0, total:0}
let lastPrediction = null;

function ensureHistory(model){
  if(!histories[model]) histories[model] = {accs:[], correct:0, total:0};
  return histories[model];
}

function markTruth(digit){
  if(lastPrediction === null) return;
  const model = modelSel.value;
  const h = ensureHistory(model);
  h.total += 1;
  if (lastPrediction === digit) h.correct += 1;
  h.accs.push(h.correct / h.total);
  renderChart();
}

for(let d=0; d<10; d++){
  const b = document.createElement('button');
  b.textContent = String(d);
  b.addEventListener('click', ()=> markTruth(d));
  digitsDiv.appendChild(b);
}

// Chart.js
const ctxChart = document.getElementById('chart').getContext('2d');
let chart = new Chart(ctxChart, {
  type: 'line',
  data: {
    labels: [],
    datasets: MODEL_NAMES.map((m,idx)=> ({
      label: m,
      data: [],
      fill: false,
      tension: 0.1
    }))
  },
  options: {
    responsive: true,
    scales: {
      y: { min: 0, max: 1, ticks: { callback: v => v.toFixed(1)} }
    },
    plugins: { legend: { position: 'bottom' } }
  }
});

function renderChart(){
  // Determine max length of any history
  const maxN = Math.max(0, ...Object.values(histories).map(h => h.accs.length));
  chart.data.labels = Array.from({length:maxN}, (_,i)=> String(i+1));
  chart.data.datasets.forEach(ds => {
    const h = histories[ds.label];
    ds.data = h ? h.accs : [];
  });
  chart.update();
}
