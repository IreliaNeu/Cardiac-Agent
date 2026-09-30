"use strict";
const data = window.CARDIAC_DEMO;
const byId = id => document.getElementById(id);
const text = (id, value) => { byId(id).textContent = value; };
const fmt = (value, digits = 4) => value == null ? "不可用" : Number(value).toFixed(digits);
function list(id, values) {
  const parent = byId(id); parent.replaceChildren();
  for (const value of values) { const li = document.createElement("li"); li.textContent = value; parent.append(li); }
}
function render(id) {
  const c = data.cases.find(item => item.patient === id) || data.cases[0];
  byId("case-select").value = c.patient;
  text("case-title", `${c.patient} · ${c.title}`);
  text("case-purpose", c.purpose);
  text("status", c.verdict === "refer" ? "需人工复核 · refer" : "受限研究输出 · limited");
  byId("status").className = c.verdict;
  const metrics = byId("metrics"); metrics.replaceChildren();
  for (const [label, value] of [["ED/ES 平均 Dice",fmt(c.dice)], ["预测 FAC",c.fac==null?"不可用":`${fmt(c.fac,2)}%`],
    ["FAC 绝对误差",c.fac_error==null?"不可用":`${fmt(c.fac_error,2)} pp`], ["角色回合", "3 次 / 1 轮"]]) {
    const item = document.createElement("div"); item.className = "metric";
    const name = document.createElement("span"); name.textContent = label;
    const valueNode = document.createElement("strong"); valueNode.textContent = value; item.append(name,valueNode); metrics.append(item);
  }
  const images = byId("images"); images.replaceChildren();
  for (const phase of ["ED","ES"]) {
    const row = document.createElement("div"); row.className="phase";
    const heading=document.createElement("h3"); heading.textContent=phase==="ED"?"ED 舒张末期":"ES 收缩末期";
    const grid=document.createElement("div"); grid.className="image-row";
    for (const [suffix, caption, cls] of [["original","原图",""],["prediction","模型预测叠加 · 绿色",""],["comparison","预测轮廓 青色 / 人工 LV 轮廓 橙色","reference"]]) {
      const fig=document.createElement("figure"); fig.className=cls;
      const img=document.createElement("img"); img.src=`cases/${c.patient}/${phase}_${suffix}.png`; img.alt=`${c.patient} ${phase} ${caption}`;
      const cap=document.createElement("figcaption"); cap.textContent=caption;
      if(suffix==="prediction") {const link=document.createElement("a"); link.href=`cases/${c.patient}/${phase}_mask.png`; link.textContent=" 二值预测"; link.target="_blank"; cap.append(link);}
      fig.append(img,cap); grid.append(fig);
    }
    row.append(heading,grid); images.append(row);
  }
  list("talking-points",c.talking_points); text("editor-note",c.editor_note);
  text("narrative",c.narrative); list("concerns",c.concerns);
  byId("report-link").href=`cases/${c.patient}/report.md`;
  const evidence=byId("evidence"); evidence.replaceChildren();
  for (const row of [["ED 预测/参考像素数",c.pred_ed,c.ref_ed],["ES 预测/参考像素数",c.pred_es,c.ref_es],
    ["FAC (%)",fmt(c.fac),fmt(c.reference_fac)],["ES/ED",fmt(c.ratio,6),fmt(c.ref_es/c.ref_ed,6)],
    ["ED Dice",fmt(c.ed_dice),"独立评估"],["ES Dice",fmt(c.es_dice),"独立评估"]]) {
    const tr=document.createElement("tr"); for(const value of row){const td=document.createElement("td");td.textContent=value;tr.append(td);} evidence.append(tr);
  }
  text("qc",`QC：${c.qc_flags.length?c.qc_flags.join("；"):"未触发当前规则"}。RWMA：${c.rwma_status}。`);
  text("limits",c.limitations);
  const roles=byId("roles"); roles.replaceChildren();
  for(const [key,label] of [["planner","Planner 计划"],["analyst","Analyst 草稿"],["reviewer","Reviewer 审查"]]) {
    const detail=document.createElement("details"), summary=document.createElement("summary"), p=document.createElement("p"), note=document.createElement("p"), a=document.createElement("a");
    summary.textContent=label; p.className="prewrap";p.textContent=c.roles[key];note.className="role-note";note.textContent="保存的原始角色输出，未重新调用模型。";
    a.href=`cases/${c.patient}/${key}.json`;a.textContent="结构化原始记录";detail.append(summary,p,note,a);roles.append(detail);
  }
  text("cohort",`原批次 ${data.summary.requested} 例；98 例完成数值计算，2 例工具失败；100 份报告。全百例 Dice ${fmt(data.summary.all_attempts_mean_dice)}，IoU ${fmt(data.summary.all_attempts_mean_iou)}；FAC MAE ${fmt(data.summary.fac_mae_pp,2)} pp（98 例）。100 个 Planner 均选择面积分析。`);
}
for(const c of data.cases){const option=document.createElement("option");option.value=c.patient;option.textContent=`${c.patient} · ${c.title}`;byId("case-select").append(option);}
byId("case-select").addEventListener("change",event=>{location.hash=event.target.value;render(event.target.value);});
byId("show-reference").addEventListener("change",event=>document.body.classList.toggle("hide-reference",!event.target.checked));
window.addEventListener("hashchange",()=>render(location.hash.slice(1)));
render(location.hash.slice(1));
