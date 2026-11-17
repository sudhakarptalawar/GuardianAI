/**
 * GuardianAI – shared app shell
 * Handles: theme toggle, auth guard, sidebar user info
 */

// ── Auth guard ────────────────────────────────────────────────────────────────
(function guardRoute() {
  const pub = ['/login/', '/signup/'];
  const path = location.pathname;
  const isPublic = pub.some(p => path.startsWith(p));
  const token = localStorage.getItem('guardian_token') || sessionStorage.getItem('guardian_token');

  if (!isPublic && !token) {
    location.href = '/login/';
    return;
  }
})();

// ── Theme ─────────────────────────────────────────────────────────────────────
const THEME_KEY = 'guardian_theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const lbl  = document.getElementById('theme-label');
  const icon = document.getElementById('theme-icon');
  if (lbl)  lbl.textContent = theme === 'dark' ? 'Dark' : 'Light';
  if (icon) icon.textContent = theme === 'dark' ? '◐' : '○';
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(saved);
  document.getElementById('theme-toggle')?.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });
}

// ── Sidebar user info ─────────────────────────────────────────────────────────
function initUserUI() {
  try {
    const raw  = localStorage.getItem('guardian_user');
    if (!raw) return;
    const user = JSON.parse(raw);

    const avatarEl = document.querySelector('.account-avatar');
    const nameEl   = document.querySelector('.sidebar-bottom strong');
    const roleEl   = document.querySelector('.sidebar-bottom small');

    if (avatarEl) {
      avatarEl.textContent = (user.name || 'U')[0].toUpperCase();
      avatarEl.style.background = `linear-gradient(145deg, ${user.avatar_col || '#4ce0c0'}, #5758b8)`;
    }
    if (nameEl) nameEl.textContent = user.name  || 'Operator';
    if (roleEl) roleEl.textContent = (user.role || 'operator').charAt(0).toUpperCase() + (user.role || 'operator').slice(1);
  } catch (_) {}
}

// ── Auth API helpers ──────────────────────────────────────────────────────────
function getToken() {
  return localStorage.getItem('guardian_token') || sessionStorage.getItem('guardian_token') || '';
}

function authHeaders() {
  return { 'Authorization': `Bearer ${getToken()}`, 'Content-Type': 'application/json' };
}

function logout() {
  localStorage.removeItem('guardian_token');
  localStorage.removeItem('guardian_user');
  sessionStorage.removeItem('guardian_token');
  location.href = '/login/';
}

// Attach logout to any element with id="logout-btn"
function initLogout() {
  document.getElementById('logout-btn')?.addEventListener('click', logout);
}

// ── Greeting ──────────────────────────────────────────────────────────────────
function initGreeting() {
  const el = document.getElementById('greeting');
  if (!el) return;
  const h = new Date().getHours();
  const prefix = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
  try {
    const user = JSON.parse(localStorage.getItem('guardian_user') || '{}');
    const name = user.name ? `, ${user.name.split(' ')[0]}` : '';
    el.innerHTML = `${prefix}${name}<span class="heading-dot">.</span>`;
  } catch (_) {
    el.innerHTML = `${prefix}<span class="heading-dot">.</span>`;
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initUserUI();
  initLogout();
  initGreeting();
});
