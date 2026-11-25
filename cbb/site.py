"""Writes docs/index.html (served by GitHub Pages) with all data embedded."""
import json
import math

from . import store

DOCS = store.ROOT / "docs"


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def render(slate, ratings, rec, bt, p, now, calibrated=True):
    tab = ratings.table().head(400)
    data = {
        "updated": now.strftime("%a %b %-d, %-I:%M %p ET"),
        "params": p.to_dict(),
        "calibrated": calibrated,
        "slate": slate,
        "record": rec,
        "backtest": bt,
        "hca": ratings.hca, "mu": ratings.mu,
        "ratings": tab[[c for c in ("team", "off", "def", "net", "games", "adj_o", "adj_d", "adj_em", "tempo")
                        if c in tab.columns]].round(2).to_dict("records"),
    }
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(_clean(data), default=float, separators=(",", ":")).replace("</", "<\\/")
    (DOCS / "index.html").write_text(TEMPLATE.replace("__DATA__", payload))
    (DOCS / ".nojekyll").write_text("")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBB Model</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#2a2f3a;--tx:#e8eaee;--mut:#8b93a3;--good:#3ecf8e;--bad:#f06a6a;--acc:#6aa8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 -apple-system,system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:12px 14px 0;border-bottom:1px solid var(--line);z-index:2}
h1{font-size:18px;margin:0}.sub{color:var(--mut);font-size:12px;margin:2px 0 8px}
nav{display:flex;gap:4px;overflow-x:auto}nav button{flex:1;background:none;border:0;color:var(--mut);padding:9px 6px;font:inherit;font-weight:600;border-bottom:2px solid transparent}
nav button.on{color:var(--tx);border-color:var(--acc)}
main{padding:12px 12px 40px;max-width:760px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:10px}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.mut{color:var(--mut);font-size:12px}.teams{font-weight:600}
.mk{display:grid;grid-template-columns:62px 1fr auto;gap:6px;align-items:center;padding:6px 0;border-top:1px solid var(--line);font-size:14px}
.tag{font-size:11px;font-weight:700;padding:2px 7px;border-radius:99px;background:#22324a;color:var(--acc)}
.play{background:#173a2b;color:var(--good)}
.ev{font-variant-numeric:tabular-nums;font-weight:600}.pos{color:var(--good)}.neg{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 4px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}
.filters{display:flex;gap:8px;margin-bottom:10px;align-items:center;flex-wrap:wrap}
.filters label{font-size:13px;color:var(--mut);display:flex;gap:6px;align-items:center}
.warn{color:#e7b35a}
h2{font-size:15px;margin:16px 0 8px}p.note{color:var(--mut);font-size:13px;margin:6px 0}
</style></head><body>
<header><h1>CBB Model</h1><div class="sub" id="upd"></div>
<nav><button data-v="slate" class="on">Slate</button><button data-v="record">Record</button><button data-v="backtest">Backtest</button><button data-v="ratings">Ratings</button></nav></header>
<main id="m"></main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const fmt=(x,d=1)=>x==null?'–':(+x).toFixed(d);
const sgn=(x,d=1)=>x==null?'–':(x>0?'+':'')+(+x).toFixed(d);
const am=x=>x==null?'–':(x>0?'+':'')+Math.round(x);
const pct=(x,d=1)=>x==null?'–':(x*100).toFixed(d)+'%';
const MIN=D.params.min_ev;
// "Priced at" = ratings blended with the market line using the backtest-fit weights
$('#upd').textContent='Updated '+D.updated+' · flag at EV ≥ '+pct(MIN,0);
let onlyPlays=false;
const CAL=D.calibrated?'':'<div class="card warn">Calibrating: historical closing lines are still loading (about a day). Until then the market-blend weights and error sizes are defaults, so treat EVs as rough and don\'t bet off them yet.</div>';

function betLine(g,mk,b){
  if(mk==='spread'){const t=b.side==='home'?g.home:g.away;return t+' '+sgn(b.line)}
  if(mk==='total'){return (b.side==='over'?'Over ':'Under ')+fmt(b.line)}
  return (b.side==='home'?g.home:g.away)+' ML';
}
function slate(){
  const S=D.slate.filter(g=>!onlyPlays||Object.values(g.bets).some(b=>b.ev>=MIN));
  let h=CAL+`<div class="filters"><label><input type="checkbox" id="op" ${onlyPlays?'checked':''}> Only flagged plays</label><span class="mut">${D.slate.length} games in the next 40h</span></div>`;
  if(!S.length)h+='<p class="note">Nothing to show. Games appear once ESPN posts lines.</p>';
  for(const g of S){
    const t=new Date(g.date).toLocaleString('en-US',{weekday:'short',hour:'numeric',minute:'2-digit',timeZone:'America/New_York'});
    const thin=Math.min(g.home_games,g.away_games)<5;
    h+=`<div class="card"><div class="row"><span class="teams">${g.away} ${g.neutral?'vs':'@'} ${g.home}</span><span class="mut">${t}</span></div>
    <div class="mut">Ratings: ${g.home} ${sgn(-g.pred_margin)} · total ${fmt(g.pred_total)} (${fmt(g.pred_away,0)}–${fmt(g.pred_home,0)})${g.neutral?' · neutral':''}
    <br>Priced at: ${g.home} ${sgn(-g.blend_margin)} · total ${fmt(g.blend_total)}
    <br>Market: ${g.spread!=null?g.home+' '+sgn(g.spread):'no spread'} · ${g.total!=null?'o/u '+fmt(g.total):'no total'} · ML ${am(g.ml_away)}/${am(g.ml_home)}${g.spread_open!=null?' · opened '+sgn(g.spread_open):''}
    ${thin?'<br><span class="warn">Under 5 games this season for one side: rating leans on last year</span>':''}</div>`;
    for(const mk of ['spread','total','ml']){const b=g.bets[mk];if(!b)continue;const play=b.ev>=MIN;
      h+=`<div class="mk"><span class="tag ${play?'play':''}">${mk==='ml'?'ML':mk[0].toUpperCase()+mk.slice(1)}</span>
      <span>${betLine(g,mk,b)} <span class="mut">${am(b.odds)} · ${pct(b.p,0)}${b.market_p!=null?' vs mkt '+pct(b.market_p,0):''}</span></span>
      <span class="ev ${b.ev>0?'pos':'neg'}">${sgn(b.ev*100)}%</span></div>`}
    h+='</div>';
  }
  $('#m').innerHTML=h;$('#op').onchange=e=>{onlyPlays=e.target.checked;slate()};
}
function record(){
  const R=D.record;let h=`<p class="note">Flagged plays (EV ≥ ${pct(MIN,0)}) graded at the price when the model first saw the line. CLV compares that price with the last line before tip. <b>CLV is the early signal</b>: a few hundred bets are needed before win/loss means much.</p>`;
  h+='<div class="card"><table><tr><th>Market</th><th>Bets</th><th>W-L-P</th><th>Units</th><th>ROI</th><th>Avg CLV</th><th>CLV+</th></tr>';
  for(const mk of ['spread','total','ml']){const o=R[mk];
    h+=`<tr><td>${mk==='ml'?'Moneyline':mk[0].toUpperCase()+mk.slice(1)}</td><td>${o.n}</td><td>${o.w}-${o.l}-${o.push}</td><td>${sgn(o.units,2)}</td><td>${pct(o.roi)}</td><td>${o.avg_clv==null?'–':sgn(o.avg_clv,2)+(mk==='ml'?' pp':' pts')}</td><td>${o.clv_n?pct(o.clv_pos/o.clv_n,0):'–'}</td></tr>`}
  h+='</table></div>';$('#m').innerHTML=h;
}
function btTable(b){
  let h='<table><tr><th>EV bucket</th><th>Bets</th><th>Win%</th><th>ROI</th><th>±1 SE</th><th>Units</th></tr>';
  for(const r of b){h+=r.n?`<tr><td>${r.ev}</td><td>${r.n}</td><td>${pct(r.win_pct)}</td><td class="${r.roi>0?'pos':''}">${pct(r.roi)}</td><td>${r.roi_se==null?'–':pct(r.roi_se)}</td><td>${sgn(r.units,1)}</td></tr>`:`<tr><td>${r.ev}</td><td>0</td><td colspan=4></td></tr>`}
  return h+'</table>';
}
function segTable(sg){
  if(!sg)return '';let h='<p class="note" style="margin-top:10px">Flagged plays by game type. Lots of splits means some look good by luck: only trust a split whose ROI beats ±2 SE and holds next season.</p><table><tr><th>Split</th><th>Bets</th><th>ROI</th><th>±1 SE</th></tr>';
  const lab={month:'',conf:'',size:'Spread '};
  for(const k of ['month','conf','size'])for(const r of (sg[k]||[]).sort((a,b)=>a.group<b.group?-1:1))
    h+=`<tr><td>${lab[k]}${r.group}</td><td>${r.n}</td><td class="${r.roi-2*(r.roi_se||9)>0?'pos':''}">${pct(r.roi)}</td><td>${r.roi_se==null?'–':pct(r.roi_se)}</td></tr>`;
  return h+'</table>';
}
function backtest(){
  const B=D.backtest;if(!B||!B.this_season){$('#m').innerHTML='<p class="note">Backtest not run yet.</p>';return}
  let h=`<p class="note">Walk-forward: each day is priced using only games played before it, against ESPN's posted line, graded on the final. Last season was used to pick the 3 rating knobs; this season is fully out of sample. If the model's error is bigger than the line's, expect edges to be mostly noise.</p>`;
  for(const [k,lab] of [['this_season','This season (out of sample)'],['last_season','Last season']]){const s=B[k];if(!s||!s.games)continue;
    h+=`<h2>${lab}: ${s.games} games</h2><div class="card"><table><tr><th>Avg miss</th><th>Model</th><th>Market line</th></tr>
    <tr><td>Margin</td><td>${fmt(s.model_margin_mae_same_games??s.model_margin_mae,2)}</td><td>${fmt(s.line_margin_mae,2)}</td></tr>
    <tr><td>Total</td><td>${fmt(s.model_total_mae_same_games??s.model_total_mae,2)}</td><td>${fmt(s.line_total_mae,2)}</td></tr></table>
    <tr><td class="mut">· points model</td><td class="mut">${fmt(s.pts_margin_mae_same_games,2)} / ${fmt(s.pts_total_mae_same_games,2)}</td><td></td></tr>
    <tr><td class="mut">· efficiency model</td><td class="mut">${fmt(s.eff_margin_mae_same_games,2)} / ${fmt(s.eff_total_mae_same_games,2)}</td><td></td></tr></table>
    <p class="note">Games with lines: ${s.n_spread_lines||0} spreads, ${s.n_total_lines||0} totals. Lower is better. "Model" is the tuned mix of the two rating models.</p></div>`;
    for(const mk of ['spread','total','ml'])if(s.buckets&&s.buckets[mk])h+=`<div class="card"><b>${mk==='ml'?'Moneyline':mk[0].toUpperCase()+mk.slice(1)}</b> by model EV${btTable(s.buckets[mk])}${segTable(s.segments&&s.segments[mk])}</div>`;
  }
  const p=D.params;h+=`<p class="note">Ratings forecast = ${pct(1-p.mix_margin,0)} points model + ${pct(p.mix_margin,0)} efficiency model for margin (${pct(1-p.mix_total,0)} / ${pct(p.mix_total,0)} for total). Bets are priced on a blend of that with the market: ${pct(p.w_margin,0)} ratings / ${pct(1-p.w_margin,0)} line for margin, ${pct(p.w_total,0)} / ${pct(1-p.w_total,0)} for total (all fit on last season). A ratings weight near 0% means they add little beyond the line.</p><p class="note">Params: half-life ${p.half_life}d · ridge ${p.lam} · last-season weight ${p.prev_w} · margin σ ${fmt(p.margin_sd,2)} · total σ ${fmt(p.total_sd,2)} · home court ${fmt(D.hca,2)} pts</p>`;
  $('#m').innerHTML=h;
}
function ratings(){
  const E=D.ratings.length&&D.ratings[0].adj_o!=null;
  let h=E?`<p class="note">Per 100 possessions vs D-I average (KenPom-style). AdjO = points scored, AdjD = points allowed (lower is better), EM = AdjO − AdjD, Tempo = possessions per game vs an average-pace opponent.</p>`
         :`<p class="note">Points per game vs an average D-I team. Off = points scored above avg, Def = points allowed above avg (lower is better). Net = Off − Def.</p>`;
  h+=`<div class="card"><table><tr><th>#</th><th style="text-align:left">Team</th>${E?'<th>EM</th><th>AdjO</th><th>AdjD</th><th>Tempo</th>':'<th>Net</th><th>Off</th><th>Def</th>'}<th>GP</th></tr>`;
  const RS=E?[...D.ratings].sort((a,b)=>b.adj_em-a.adj_em):D.ratings;
  RS.forEach((r,i)=>{h+=`<tr><td>${i+1}</td><td style="text-align:left">${r.team}</td>${E?`<td>${sgn(r.adj_em)}</td><td>${fmt(r.adj_o)}</td><td>${fmt(r.adj_d)}</td><td>${fmt(r.tempo)}</td>`:`<td>${sgn(r.net)}</td><td>${sgn(r.off)}</td><td>${sgn(r.def)}</td>`}<td>${r.games}</td></tr>`});
  $('#m').innerHTML=h+'</table></div>';
}
const V={slate,record,backtest,ratings};
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));V[b.dataset.v]()});
slate();
</script></body></html>"""
