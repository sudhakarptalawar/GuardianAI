const defaults={theme:'dark',mode:'Home',source:'0',detectors:['person','fire','object']};
let preferences;
try{preferences={...defaults,...JSON.parse(localStorage.getItem('guardianai-preferences')||'{}')}}catch{preferences={...defaults}}
function savePreferences(){localStorage.setItem('guardianai-preferences',JSON.stringify(preferences))}
function setTheme(value){preferences.theme=value==='light'?'light':'dark';document.documentElement.dataset.theme=preferences.theme;const label=document.querySelector('#theme-label');if(label)label.textContent=preferences.theme==='dark'?'Dark':'Light';const select=document.querySelector('#settings-theme');if(select)select.value=preferences.theme;savePreferences()}
function setMode(value){preferences.mode=value;const head=document.querySelector('#mode-select');if(head)head.value=value;const select=document.querySelector('#settings-mode');if(select)select.value=value;savePreferences()}
window.guardianPreferences=()=>preferences;
window.guardianSavePreferences=savePreferences;
window.guardianSetMode=setMode;
window.guardianSetTheme=setTheme;
document.documentElement.dataset.theme=preferences.theme;
const themeLabel=document.querySelector('#theme-label');if(themeLabel)themeLabel.textContent=preferences.theme==='dark'?'Dark':'Light';
const modeSelect=document.querySelector('#mode-select');if(modeSelect){modeSelect.value=preferences.mode;modeSelect.addEventListener('change',e=>setMode(e.target.value))}
const themeToggle=document.querySelector('#theme-toggle');if(themeToggle)themeToggle.addEventListener('click',()=>setTheme(preferences.theme==='dark'?'light':'dark'));
const dateLabel=document.querySelector('#date-label');if(dateLabel)dateLabel.textContent=new Intl.DateTimeFormat(undefined,{weekday:'long',month:'long',day:'numeric'}).format(new Date()).toUpperCase();
const greeting=document.querySelector('#greeting');if(greeting){const h=new Date().getHours();greeting.innerHTML=`${h<12?'Good morning':h<18?'Good afternoon':'Good evening'}<span class="heading-dot">.</span>`}
