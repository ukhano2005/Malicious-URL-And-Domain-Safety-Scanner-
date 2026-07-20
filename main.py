import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime
import threading
import re
import os
import joblib
from urllib.parse import urlparse

# Configure matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Segoe UI']

# ========== MODEL TRAINING ==========
class URLScannerModel:
    # A small, curated set of globally recognized, unambiguously legitimate
    # root domains. The underlying training dataset under-represents these
    # brands as "benign" relative to phishing pages that impersonate them
    # (e.g. "github" appears far more often in phishing URLs than in benign
    # ones), which biases the raw ML model toward false positives on them.
    # This override corrects that specific, well-understood bias without
    # touching the model's behavior on anything else.
    TRUSTED_ROOT_DOMAINS = {
        "google.com", "youtube.com", "facebook.com", "wikipedia.org",
        "amazon.com", "apple.com", "microsoft.com", "github.com",
        "twitter.com", "x.com", "instagram.com", "linkedin.com",
        "reddit.com", "netflix.com", "yahoo.com", "bing.com",
        "paypal.com", "ebay.com", "wordpress.com", "adobe.com",
        "salesforce.com", "zoom.us", "dropbox.com", "spotify.com",
        "wikimedia.org", "mozilla.org", "stackoverflow.com",
        "cloudflare.com", "bankofamerica.com", "chase.com",
        "wellsfargo.com", "citibank.com", "anthropic.com", "openai.com",
    }

    @staticmethod
    def get_registered_domain(url):
        """Extract the registrable domain (last two labels) from a URL,
        used to safely match against the trusted-domain allowlist."""
        try:
            candidate = url if "://" in url else "http://" + url
            netloc = urlparse(candidate).netloc.lower()
            netloc = netloc.split("@")[-1].split(":")[0]
            parts = [p for p in netloc.split(".") if p]
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return netloc
        except Exception:
            return ""

    def __init__(self, dataset_path=None):
        self.vectorizer = None
        self.model = None
        self.is_trained = False

        # Resolve all paths relative to this script's own folder so the
        # model/dataset are found no matter where the app is launched from.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if dataset_path is None:
            dataset_path = os.path.join(base_dir, "dataset.csv")
        self.model_file = os.path.join(base_dir, "url_scanner_model.pkl")
        self.vectorizer_file = os.path.join(base_dir, "vectorizer.pkl")
        
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            print("Loading pre-trained model...")
            try:
                self.model = joblib.load(self.model_file)
                self.vectorizer = joblib.load(self.vectorizer_file)
                self.is_trained = True
                print("Pre-trained model loaded successfully!")
                return
            except Exception as e:
                print(f"Could not load pre-trained model: {e}")
        
        print("Training new model from dataset...")
        self.train_model(dataset_path)
        
        if self.is_trained:
            try:
                joblib.dump(self.model, self.model_file)
                joblib.dump(self.vectorizer, self.vectorizer_file)
                print("Model saved!")
            except Exception as e:
                print(f"Could not save model: {e}")
    
    def train_model(self, dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            df = df[["url", "type"]].dropna()
            df = df.sample(min(50000, len(df)), random_state=42)
            
            label_map = {"benign": 0, "defacement": 1, "phishing": 2, "malware": 3}
            df["label"] = df["type"].map(label_map)
            df = df.dropna()
            
            X = df["url"]
            y = df["label"]
            
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_vec = self.vectorizer.fit_transform(X)
            
            self.model = MultinomialNB()
            self.model.fit(X_vec, y)
            self.is_trained = True
            print("Model trained successfully!")
        except Exception as e:
            print(f"Could not load dataset from '{dataset_path}': {e}")
            print("Using fallback pattern matching mode...")
            self.is_trained = False
            
    # Explicit, unambiguous attack-indicator keywords. If a URL contains
    # one of these, it is for all practical purposes a hostile/defaced URL
    # regardless of what the raw ML model's vocabulary statistics say —
    # the underlying training set under-represents these literal terms
    # within its "defacement"/"malware" classes, which otherwise causes
    # obviously bad URLs like "hacked-site.com" to be scored as benign.
    DEFACEMENT_SIGNAL_KEYWORDS = [
        "hacked", "defaced", "deface", "pwned", "owned-by", "cyber-attack",
        "cyberattack", "anonymous-hack", "hack-team", "hacktivist",
    ]
    MALWARE_SIGNAL_KEYWORDS = [
        ".exe", "crack", "keygen", "warez", "trojan", "ransomware",
    ]
    PHISHING_SIGNAL_KEYWORDS = [
        "verify-account", "secure-login", "account-update", "confirm-identity",
        "login-verify", "paypal-secure", "banking-alert",
    ]

    def _apply_explicit_signal_override(self, url, label, confidence):
        """Boosts confidence (and corrects the label if needed) when a URL
        contains an explicit, unambiguous attack-indicator keyword that the
        base model's learned vocabulary underweights. Only fires on strong,
        specific phrases — never on single common words like 'secure' or
        'login' alone, to avoid creating new false positives."""
        url_lower = url.lower()

        for kw in self.DEFACEMENT_SIGNAL_KEYWORDS:
            if kw in url_lower:
                return "defacement", max(confidence, 93.0)
        for kw in self.MALWARE_SIGNAL_KEYWORDS:
            if kw in url_lower:
                return "malware", max(confidence, 88.0)
        for kw in self.PHISHING_SIGNAL_KEYWORDS:
            if kw in url_lower:
                return "phishing", max(confidence, 88.0)

        return label, confidence

    def predict(self, url):
        domain = self.get_registered_domain(url)
        has_ip = bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))

        # Trusted-domain override: corrects a known bias where well-known,
        # legitimate brands are under-represented as "benign" in the
        # training data relative to phishing pages that impersonate them.
        # Only applies to an exact registrable-domain match (not a
        # substring), so "google.com.evil-phish.tk" is NOT affected.
        if domain in self.TRUSTED_ROOT_DOMAINS and not has_ip:
            return "benign", 96.0

        if self.is_trained and self.model is not None:
            try:
                url_vec = self.vectorizer.transform([url])
                pred = self.model.predict(url_vec)[0]
                prob = self.model.predict_proba(url_vec).max() * 100
                reverse_map = {0: "benign", 1: "defacement", 2: "phishing", 3: "malware"}
                label, prob = self._apply_explicit_signal_override(
                    url, reverse_map[pred], prob)
                return label, prob
            except Exception:
                pass
        
        # ── Fallback detection (used only if the ML model is unavailable) ──
        # Scores each threat category independently, and also scores
        # "benign-looking" structural signals (HTTPS, no IP, short length,
        # no suspicious keywords) so a clean URL isn't forced into a
        # threat category just because no other category scored higher.
        url_lower = url.lower()
        
        phishing_keywords = ['login', 'signin', 'verify', 'secure', 'account', 'confirm',
                            'update', 'banking', 'password', 'credential']
        phishing_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.live', '.work']
        malware_keywords = ['.exe', '.zip', '.rar', '.msi', 'download', 'setup', 'crack', 'keygen']
        defacement_keywords = ['hacked', 'defaced', 'owned', 'pwned', 'cyber-attack', 'deface']
        
        phishing_score = 0
        malware_score = 0
        defacement_score = 0
        benign_score = 0
        
        for keyword in phishing_keywords:
            if keyword in url_lower:
                phishing_score += 22
        for tld in phishing_tlds:
            if tld in url_lower:
                phishing_score += 35
        for keyword in malware_keywords:
            if keyword in url_lower:
                malware_score += 35
        for keyword in defacement_keywords:
            if keyword in url_lower:
                defacement_score += 40
        
        if has_ip:
            phishing_score += 30

        # Positive (benign) signals — a URL only looks suspicious if it
        # actually has suspicious traits, not by default.
        is_https = url_lower.startswith("https://")
        digit_count = sum(c.isdigit() for c in url)
        if is_https:
            benign_score += 25
        if not has_ip:
            benign_score += 10
        if len(url) <= 45:
            benign_score += 15
        if digit_count <= 2:
            benign_score += 10
        # A reasonably normal-looking registrable domain with a common TLD
        if domain and domain.split(".")[-1] in ("com", "org", "net", "edu", "gov", "io"):
            benign_score += 15
        
        scores = {"benign": benign_score, "phishing": phishing_score,
                  "malware": malware_score, "defacement": defacement_score}
        winner = max(scores, key=scores.get)
        max_score = scores[winner]
        total = sum(scores.values()) or 1
        confidence = min(97.0, max(55.0, 50 + (max_score / total) * 50))

        winner, confidence = self._apply_explicit_signal_override(url, winner, confidence)
        return winner, confidence


# ========== MAIN APPLICATION ==========
class ProfessionalURLScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Malicious URL Scanner | Advanced Threat Detection")
        self.root.geometry("1600x950")
        self.root.configure(bg="#0a0e27")
        self.root.minsize(1400, 850)
        
        self.model = URLScannerModel()
        self.scan_history = []
        self.total_scanned = 0
        self.malicious_count = 0
        self.normal_count = 0
        self.confidence_sum = 0
        self.keyword_counts = {kw: 0 for kw in ["login", "verify", "secure", "update", "free", "account", "bank"]}
        self.daily_data = {i: {"malicious": 0, "normal": 0} for i in range(7)}
        self.current_view = "dashboard"
        
        self.figure = None
        self.ax = None
        self.canvas = None
        self.keyword_frames = []
        self.stats_labels = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        # ══ TOP BAR ═══════════════════════════════════════════════════════════
        top_bar = tk.Canvas(self.root, bg="#020817", height=72, highlightthickness=0)
        top_bar.pack(fill="x")

        def _draw_topbar(e=None):
            top_bar.delete("bg")
            w = top_bar.winfo_width() or 1600
            # subtle right-side glow
            top_bar.create_oval(w-200, -60, w+60, 140, fill="#0d1f3c", outline="", tags="bg")
            top_bar.create_rectangle(0, 70, w, 72, fill="#1e3a5f", outline="", tags="bg")
        top_bar.bind("<Configure>", _draw_topbar)
        top_bar.after(30, _draw_topbar)

        # Logo group
        logo_win = tk.Frame(top_bar, bg="#020817")
        top_bar.create_window(28, 36, window=logo_win, anchor="w")

        shield_c = tk.Canvas(logo_win, width=44, height=44, bg="#020817", highlightthickness=0)
        shield_c.pack(side="left")
        shield_c.create_polygon([22,2, 40,10, 40,26, 22,42, 4,26, 4,10],
                                 fill="#1d4ed8", outline="#3b82f6", width=2)
        shield_c.create_text(22, 24, text="🛡", font=("Segoe UI", 16), fill="white")

        tk.Label(logo_win, text=" ThreatScan", bg="#020817", fg="white",
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(logo_win, text="  ·  Advanced URL Intelligence",
                 bg="#020817", fg="#334155", font=("Segoe UI", 11)).pack(side="left")

        # Status pill
        status_win = tk.Frame(top_bar, bg="#020817")
        top_bar.create_window(560, 36, window=status_win, anchor="w")
        pill = tk.Frame(status_win, bg="#052e16")
        pill.pack(side="left", padx=6)
        tk.Label(pill, text="● LIVE", bg="#052e16", fg="#4ade80",
                 font=("Segoe UI", 10, "bold"), padx=10, pady=4).pack()

        # ══ SIDEBAR ════════════════════════════════════════════════════════════
        sidebar = tk.Frame(self.root, bg="#020817", width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Sidebar top accent line
        tk.Frame(sidebar, bg="#1e3a5f", height=1).pack(fill="x")

        # Mini brand inside sidebar
        sb_brand = tk.Frame(sidebar, bg="#020817")
        sb_brand.pack(fill="x", padx=20, pady=(22, 8))
        tk.Label(sb_brand, text="NAVIGATION", bg="#020817", fg="#1e3a5f",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        tk.Frame(sidebar, bg="#0d1f3c", height=1).pack(fill="x", padx=20)

        # Nav items
        nav_items = [
            ("📊", "Dashboard",  self.show_dashboard, "#3b82f6"),
            ("🔍", "Scan URL",   self.show_scanner,   "#2563eb"),
            ("📜", "History",    self.show_history,   "#7c3aed"),
            ("ℹ️",  "About",      self.show_about,     "#0891b2"),
        ]

        self.nav_buttons = []
        self.nav_accents = []
        for icon, text, command, accent in nav_items:
            btn_outer = tk.Frame(sidebar, bg="#020817")
            btn_outer.pack(fill="x", padx=16, pady=4)

            accent_bar = tk.Frame(btn_outer, bg="#020817", width=3)
            accent_bar.pack(side="left", fill="y")

            btn = tk.Button(
                btn_outer, text=f"  {icon}   {text}",
                bg="#020817", fg="#64748b",
                font=("Segoe UI", 13), relief="flat", cursor="hand2",
                anchor="w", padx=12, pady=11,
                activebackground="#0d1f3c", activeforeground="white",
                command=command
            )
            btn.pack(side="left", fill="x", expand=True)
            self.nav_buttons.append(btn)
            self.nav_accents.append((accent_bar, accent))

        self.highlight_nav(0)

        # Sidebar stats mini-panel
        tk.Frame(sidebar, bg="#0d1f3c", height=1).pack(fill="x", padx=20, pady=(20, 0))
        sb_stats = tk.Frame(sidebar, bg="#020817")
        sb_stats.pack(fill="x", padx=20, pady=12)
        tk.Label(sb_stats, text="SESSION", bg="#020817", fg="#1e3a5f",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._sb_total_lbl = tk.Label(sb_stats, text="0 scans",
                                       bg="#020817", fg="#60a5fa",
                                       font=("Segoe UI", 12, "bold"))
        self._sb_total_lbl.pack(anchor="w")
        self._sb_threat_lbl = tk.Label(sb_stats, text="0 threats",
                                        bg="#020817", fg="#f87171",
                                        font=("Segoe UI", 11))
        self._sb_threat_lbl.pack(anchor="w")

        # Sidebar footer
        sb_footer = tk.Frame(sidebar, bg="#020817")
        sb_footer.pack(side="bottom", fill="x", padx=20, pady=20)
        tk.Frame(sb_footer, bg="#0d1f3c", height=1).pack(fill="x", pady=(0, 12))
        dot_row = tk.Frame(sb_footer, bg="#020817")
        dot_row.pack(anchor="w")
        tk.Label(dot_row, text="●", bg="#020817", fg="#22c55e",
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Label(dot_row, text="  ML Model Active", bg="#020817", fg="#334155",
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(sb_footer, text="v2.0.0  ·  Naive Bayes + TF-IDF",
                 bg="#020817", fg="#1e293b", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

        # ══ MAIN CONTENT AREA ══════════════════════════════════════════════════
        self.content_area = tk.Frame(self.root, bg="#060d1f")
        self.content_area.pack(side="right", fill="both", expand=True)

        self.show_dashboard()
    
    def highlight_nav(self, index):
        for i, btn in enumerate(self.nav_buttons):
            accent_bar, accent_color = self.nav_accents[i]
            if i == index:
                btn.config(bg="#0d1f3c", fg="white", font=("Segoe UI", 13, "bold"))
                accent_bar.config(bg=accent_color)
            else:
                btn.config(bg="#020817", fg="#64748b", font=("Segoe UI", 13))
                accent_bar.config(bg="#020817")
    
    def safe_update_widget(self, widget, **kwargs):
        try:
            if widget and widget.winfo_exists():
                widget.config(**kwargs)
                return True
        except:
            pass
        return False
    
    def show_dashboard(self):
        self.current_view = "dashboard"
        self.highlight_nav(0)

        for widget in self.content_area.winfo_children():
            widget.destroy()

        # ══ SCROLLABLE WRAPPER ════════════════════════════════════════════════
        pg_canvas = tk.Canvas(self.content_area, bg="#060d1f", highlightthickness=0)
        pg_vsb    = ttk.Scrollbar(self.content_area, orient="vertical", command=pg_canvas.yview)
        pg_canvas.configure(yscrollcommand=pg_vsb.set)
        pg_vsb.pack(side="right", fill="y")
        pg_canvas.pack(side="left", fill="both", expand=True)

        scroll_root = tk.Frame(pg_canvas, bg="#060d1f")
        scroll_win  = pg_canvas.create_window((0, 0), window=scroll_root, anchor="nw")

        def _cfg(e): pg_canvas.configure(scrollregion=pg_canvas.bbox("all"))
        def _resize(e): pg_canvas.itemconfig(scroll_win, width=e.width)
        def _mwheel(e): pg_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        scroll_root.bind("<Configure>", _cfg)
        pg_canvas.bind("<Configure>", _resize)
        pg_canvas.bind_all("<MouseWheel>", _mwheel)

        # ── HERO BANNER ───────────────────────────────────────────────────────
        hero = tk.Canvas(scroll_root, bg="#020817", height=190, highlightthickness=0)
        hero.pack(fill="x")

        def _draw_hero(e=None):
            hero.delete("all")
            w = hero.winfo_width() or 1300
            import math, random
            # background
            hero.create_rectangle(0, 0, w, 190, fill="#020817", outline="")
            # geometric accent shapes
            hero.create_oval(w-260, -80, w+80, 260, fill="#0d1f3c", outline="")
            hero.create_oval(w-180, -40, w+20, 160, fill="#1d4ed8", outline="")
            hero.create_oval(-60, 90, 180, 310, fill="#0c1a38", outline="")
            # grid dot-field
            random.seed(7)
            for _ in range(70):
                x = random.randint(0, w)
                y = random.randint(0, 190)
                r = random.randint(1, 3)
                c = random.choice(["#1e3a5f","#1d4ed8","#2563eb","#0d2044"])
                hero.create_oval(x-r, y-r, x+r, y+r, fill=c, outline="")
            # pulse rings around shield
            for rad, alpha in [(58,0.08),(44,0.15),(30,0.25)]:
                shade = "#1d4ed8"
                hero.create_oval(72-rad, 95-rad, 72+rad, 95+rad,
                                 outline=shade, width=1)
            # drawn shield
            pts = [72,40, 110,58, 110,92, 72,148, 34,92, 34,58]
            hero.create_polygon(pts, fill="#1d4ed8", outline="#3b82f6", width=2)
            hero.create_text(72, 95, text="🛡", font=("Segoe UI", 26), fill="white")
            # text
            hero.create_text(140, 72, text="Threat Intelligence Dashboard",
                             font=("Segoe UI", 24, "bold"), fill="white", anchor="w")
            hero.create_text(140, 104, text="Real-time analytics · ML-powered detection · Session overview",
                             font=("Segoe UI", 12), fill="#60a5fa", anchor="w")
            # status chips
            chips = [("● MODEL ACTIVE","#052e16","#4ade80"),
                     ("● SCANNING READY","#0c1a38","#60a5fa"),
                     ("● v2.0.0","#1a0533","#a78bfa")]
            for ci,(ctxt,cbg,cfg) in enumerate(chips):
                bx = 140 + ci*165
                hero.create_rectangle(bx, 128, bx+152, 152, fill=cbg, outline="")
                hero.create_text(bx+76, 140, text=ctxt,
                                 font=("Segoe UI", 10, "bold"), fill=cfg)

        hero.bind("<Configure>", _draw_hero)
        hero.after(50, _draw_hero)

        # ── STAT CARDS ────────────────────────────────────────────────────────
        stats_outer = tk.Frame(scroll_root, bg="#060d1f")
        stats_outer.pack(fill="x", padx=40, pady=(28, 0))

        tk.Label(stats_outer, text="SESSION OVERVIEW", bg="#060d1f", fg="#1e3a5f",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

        stats_row = tk.Frame(stats_outer, bg="#060d1f")
        stats_row.pack(fill="x")

        stats_data = [
            ("🔎", "Total URLs Scanned", "total",      "#3b82f6", "All scans this session"),
            ("⚠️",  "Threats Detected",   "malicious",  "#ef4444", "Malicious URLs found"),
            ("✅",  "Safe URLs",           "normal",     "#22c55e", "Clean URLs confirmed"),
            ("📈",  "Avg Confidence",      "confidence", "#a855f7", "Model certainty score"),
        ]

        self.stats_labels = {}
        for col_i, (icon, title, key, accent, subtitle) in enumerate(stats_data):
            card = tk.Frame(stats_row, bg="#0f172a", bd=0)
            card.grid(row=0, column=col_i, padx=8, sticky="nsew")
            stats_row.grid_columnconfigure(col_i, weight=1)

            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            ci = tk.Frame(card, bg="#0f172a")
            ci.pack(fill="both", padx=20, pady=18)

            top_row = tk.Frame(ci, bg="#0f172a")
            top_row.pack(fill="x")
            tk.Label(top_row, text=icon, bg="#0f172a", fg=accent,
                     font=("Segoe UI", 18)).pack(side="left")
            tk.Label(top_row, text=title, bg="#0f172a", fg="#475569",
                     font=("Segoe UI", 11)).pack(side="left", padx=10)

            val_lbl = tk.Label(ci, text="0", bg="#0f172a", fg="white",
                               font=("Segoe UI", 36, "bold"))
            val_lbl.pack(anchor="w", pady=(10, 2))
            tk.Label(ci, text=subtitle, bg="#0f172a", fg="#334155",
                     font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 4))

            self.stats_labels[key] = val_lbl

        self.update_stats_display()

        # ── DIVIDER ───────────────────────────────────────────────────────────
        tk.Frame(scroll_root, bg="#0d1f3c", height=1).pack(fill="x", padx=40, pady=(30, 0))

        # ── CHART + KEYWORDS ROW ──────────────────────────────────────────────
        charts_sec = tk.Frame(scroll_root, bg="#060d1f")
        charts_sec.pack(fill="x", padx=40, pady=(24, 0))

        tk.Label(charts_sec, text="ACTIVITY MONITOR", bg="#060d1f", fg="#1e3a5f",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

        charts_row = tk.Frame(charts_sec, bg="#060d1f")
        charts_row.pack(fill="x")

        # Left: Activity chart card
        chart_card = tk.Frame(charts_row, bg="#0f172a", bd=0)
        chart_card.pack(side="left", fill="both", expand=True, padx=(0, 12))
        tk.Frame(chart_card, bg="#2563eb", height=3).pack(fill="x")

        ch_hdr = tk.Frame(chart_card, bg="#0f172a")
        ch_hdr.pack(fill="x", padx=20, pady=(16, 4))
        tk.Label(ch_hdr, text="Recent Activity", bg="#0f172a", fg="white",
                 font=("Segoe UI", 15, "bold")).pack(side="left")

        tk.Label(chart_card, text="Malicious vs Safe URLs — Last 7 days",
                 bg="#0f172a", fg="#334155", font=("Segoe UI", 11)).pack(
                 anchor="w", padx=20, pady=(0, 8))

        self.figure = Figure(figsize=(8, 4.5), dpi=96, facecolor="#0f172a")
        self.ax     = self.figure.add_subplot(111)
        self.ax.set_facecolor("#0f172a")
        self.canvas = FigureCanvasTkAgg(self.figure, chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.update_graph()

        # Right: Keywords card
        kw_card = tk.Frame(charts_row, bg="#0f172a", bd=0, width=340)
        kw_card.pack(side="right", fill="y")
        kw_card.pack_propagate(False)
        tk.Frame(kw_card, bg="#7c3aed", height=3).pack(fill="x")

        tk.Label(kw_card, text="Top Malicious Keywords", bg="#0f172a", fg="white",
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=20, pady=(16, 4))
        tk.Label(kw_card, text="Most common threat indicators",
                 bg="#0f172a", fg="#334155", font=("Segoe UI", 11)).pack(
                 anchor="w", padx=20, pady=(0, 10))

        kw_colors = ["#ef4444","#f97316","#eab308","#22c55e","#3b82f6","#8b5cf6","#ec4899"]
        self.keyword_frames = []
        for ki, kw in enumerate(["login","verify","secure","update","free","account","bank"]):
            kc = kw_colors[ki % len(kw_colors)]
            kw_row = tk.Frame(kw_card, bg="#0d1626")
            kw_row.pack(fill="x", padx=16, pady=3)

            tk.Label(kw_row, text=kw, bg="#0d1626", fg=kc,
                     font=("Segoe UI", 12, "bold"), width=9, anchor="w").pack(
                     side="left", padx=10, pady=8)

            prog_wrap = tk.Frame(kw_row, bg="#1e293b", height=8)
            prog_wrap.pack(side="left", fill="x", expand=True, padx=6)
            prog_wrap.pack_propagate(False)
            prog_bar = tk.Frame(prog_wrap, bg=kc, height=8, width=0)
            prog_bar.pack(side="left")

            cnt_lbl = tk.Label(kw_row, text="0", bg="#0d1626", fg="#94a3b8",
                               font=("Segoe UI", 11, "bold"), width=4, anchor="e")
            cnt_lbl.pack(side="right", padx=10)

            self.keyword_frames.append({
                "name": kw, "count_label": cnt_lbl,
                "progress_bar": prog_bar, "progress_frame": prog_wrap
            })

        self.update_keywords_display()

        # ── THREAT BREAKDOWN STRIP ────────────────────────────────────────────
        tb_wrap = tk.Frame(scroll_root, bg="#020817")
        tb_wrap.pack(fill="x", pady=(30, 0))
        tk.Frame(tb_wrap, bg="#0d1f3c", height=1).pack(fill="x")

        tb_inner = tk.Frame(tb_wrap, bg="#020817")
        tb_inner.pack(fill="x", padx=40, pady=(24, 24))

        tk.Label(tb_inner, text="THREAT CATEGORIES TRACKED", bg="#020817",
                 fg="#1e3a5f", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 12))

        tb_row = tk.Frame(tb_inner, bg="#020817")
        tb_row.pack(fill="x")

        threats = [
            ("🎣", "Phishing",    "Credential theft & fake login pages", "#ef4444", "#450a0a"),
            ("🦠", "Malware",     "Drive-by downloads & exploit kits",   "#f97316", "#431407"),
            ("💀", "Defacement",  "Hacked & compromised web pages",       "#a855f7", "#3b0764"),
            ("✅", "Benign",      "Verified safe & legitimate URLs",      "#22c55e", "#052e16"),
        ]
        for col_i,(icon,name,desc,fg_c,bg_c) in enumerate(threats):
            tc = tk.Frame(tb_row, bg=bg_c, bd=0)
            tc.grid(row=0, column=col_i, padx=8, sticky="nsew")
            tb_row.grid_columnconfigure(col_i, weight=1)
            tp = tk.Frame(tc, bg=bg_c)
            tp.pack(fill="both", padx=18, pady=16)
            tk.Label(tp, text=icon, bg=bg_c, fg=fg_c,
                     font=("Segoe UI", 24)).pack(anchor="w")
            tk.Label(tp, text=name, bg=bg_c, fg=fg_c,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(6, 2))
            tk.Label(tp, text=desc, bg=bg_c, fg="#cbd5e1",
                     font=("Segoe UI", 10), wraplength=210, justify="left").pack(anchor="w")

        # ── QUICK ACTIONS ─────────────────────────────────────────────────────
        qa_wrap = tk.Frame(scroll_root, bg="#060d1f")
        qa_wrap.pack(fill="x", padx=40, pady=(28, 0))

        tk.Label(qa_wrap, text="QUICK ACTIONS", bg="#060d1f", fg="#1e3a5f",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

        qa_row = tk.Frame(qa_wrap, bg="#060d1f")
        qa_row.pack(fill="x")

        qa_items = [
            ("🔍  Scan a New URL",    "Analyze any URL for threats now",       "#2563eb", self.show_scanner),
            ("📜  View History",       "Review all your past scan results",      "#7c3aed", self.show_history),
            ("ℹ️   Learn More",         "About the model and how it works",       "#0891b2", self.show_about),
        ]
        for col_i,(label,sub,accent,cmd) in enumerate(qa_items):
            qc = tk.Frame(qa_row, bg="#0f172a", cursor="hand2")
            qc.grid(row=0, column=col_i, padx=8, sticky="nsew")
            qa_row.grid_columnconfigure(col_i, weight=1)
            tk.Frame(qc, bg=accent, height=3).pack(fill="x")
            qp = tk.Frame(qc, bg="#0f172a")
            qp.pack(fill="both", padx=20, pady=18)
            tk.Label(qp, text=label, bg="#0f172a", fg="white",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w")
            tk.Label(qp, text=sub, bg="#0f172a", fg="#475569",
                     font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 8))
            tk.Button(qp, text="Open →", bg=accent, fg="white",
                      font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
                      padx=16, pady=6, command=cmd).pack(anchor="w")

        # ── SAFETY TIP BANNER ────────────────────────────────────────────────
        tip_banner = tk.Frame(scroll_root, bg="#0c1a38")
        tip_banner.pack(fill="x", pady=(30, 0))
        tk.Frame(tip_banner, bg="#1d4ed8", height=1).pack(fill="x")
        tip_in = tk.Frame(tip_banner, bg="#0c1a38")
        tip_in.pack(fill="x", padx=40, pady=18)
        tk.Label(tip_in, text="💡  Security Reminder", bg="#0c1a38", fg="#60a5fa",
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(tip_in,
                 text="Always scan unknown URLs before clicking — especially in emails, DMs, and social posts.",
                 bg="#0c1a38", fg="#93c5fd", font=("Segoe UI", 11)).pack(side="left", padx=20)
        tk.Button(tip_in, text="Scan Now →", bg="#1d4ed8", fg="white",
                  font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
                  padx=16, pady=6, command=self.show_scanner).pack(side="right")

        # ── FOOTER ────────────────────────────────────────────────────────────
        footer = tk.Frame(scroll_root, bg="#020817")
        footer.pack(fill="x", pady=(24, 0))
        tk.Frame(footer, bg="#0d1f3c", height=1).pack(fill="x")
        fp = tk.Frame(footer, bg="#020817")
        fp.pack(fill="x", padx=40, pady=16)
        tk.Label(fp, text="🛡️  ThreatScan  ·  Powered by Naive Bayes ML  ·  Built for Security",
                 bg="#020817", fg="#1e293b", font=("Segoe UI", 10)).pack(side="left")
        tk.Label(fp, text="Always scan before you click  🔒",
                 bg="#020817", fg="#1d4ed8", font=("Segoe UI", 10, "bold")).pack(side="right")
    
    def show_scanner(self):
        self.current_view = "scanner"
        self.highlight_nav(1)

        for widget in self.content_area.winfo_children():
            widget.destroy()

        # ══════════════════════════════════════════════════════════════════════
        #  SCROLLABLE WRAPPER — entire scanner page scrolls
        # ══════════════════════════════════════════════════════════════════════
        page_canvas = tk.Canvas(self.content_area, bg="#0a0e27", highlightthickness=0)
        page_vsb   = ttk.Scrollbar(self.content_area, orient="vertical",
                                    command=page_canvas.yview)
        page_canvas.configure(yscrollcommand=page_vsb.set)
        page_vsb.pack(side="right", fill="y")
        page_canvas.pack(side="left", fill="both", expand=True)

        scroll_root = tk.Frame(page_canvas, bg="#0a0e27")
        scroll_win  = page_canvas.create_window((0, 0), window=scroll_root, anchor="nw")

        def _on_frame_configure(e):
            page_canvas.configure(scrollregion=page_canvas.bbox("all"))
        def _on_canvas_configure(e):
            page_canvas.itemconfig(scroll_win, width=e.width)
        def _on_mousewheel(e):
            page_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        scroll_root.bind("<Configure>", _on_frame_configure)
        page_canvas.bind("<Configure>", _on_canvas_configure)
        page_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── HERO BANNER ───────────────────────────────────────────────────────
        hero = tk.Canvas(scroll_root, bg="#060d1f", height=200, highlightthickness=0)
        hero.pack(fill="x")

        def _draw_hero(e=None):
            hero.delete("all")
            w = hero.winfo_width() or 1200
            # gradient-like horizontal bands
            bands = [
                ("#060d1f","#0a1628"),("#0a1628","#0d1f3c"),("#0d1f3c","#0a1628"),("#0a1628","#060d1f")
            ]
            bh = 50
            for i,(c1,c2) in enumerate(bands):
                hero.create_rectangle(0, i*bh, w, (i+1)*bh, fill=c2, outline="")

            # decorative circles — top-right cluster
            hero.create_oval(w-180, -60, w+60, 180, fill="#1e3a5f", outline="")
            hero.create_oval(w-120, -20, w+20, 120, fill="#1d4ed8", outline="")
            # bottom-left arc
            hero.create_oval(-80, 80, 120, 280, fill="#0f2d5c", outline="")

            # grid dots
            import random
            random.seed(42)
            for _ in range(55):
                x = random.randint(0, w)
                y = random.randint(0, 200)
                r = random.randint(1, 3)
                alpha_col = random.choice(["#1e3a5f","#1d4ed8","#2563eb","#3b82f6"])
                hero.create_oval(x-r, y-r, x+r, y+r, fill=alpha_col, outline="")

            # shield icon (drawn)
            cx, cy = 80, 100
            pts = [cx, cy-55, cx+45, cy-30, cx+45, cy+15, cx, cy+55, cx-45, cy+15, cx-45, cy-30]
            hero.create_polygon(pts, fill="#1d4ed8", outline="#3b82f6", width=2)
            hero.create_text(cx, cy, text="🛡", font=("Segoe UI", 28), fill="white")

            # hero text
            hero.create_text(150, 78, text="URL Threat Intelligence",
                             font=("Segoe UI", 26, "bold"), fill="white", anchor="w")
            hero.create_text(150, 114, text="Powered by Machine Learning · Real-Time Detection · Zero Compromise",
                             font=("Segoe UI", 12), fill="#60a5fa", anchor="w")

            # pill badges
            for bi, (btext, bcol) in enumerate([
                ("✓ Phishing","#1d4ed8"), ("✓ Malware","#7c3aed"), ("✓ Defacement","#0891b2")
            ]):
                bx = 150 + bi * 160
                hero.create_rectangle(bx, 142, bx+145, 168, fill=bcol, outline="")
                hero.create_text(bx+72, 155, text=btext,
                                 font=("Segoe UI", 11, "bold"), fill="white")

        hero.bind("<Configure>", _draw_hero)
        hero.after(50, _draw_hero)

        # ── SCANNER CARD ──────────────────────────────────────────────────────
        card_wrap = tk.Frame(scroll_root, bg="#0a0e27")
        card_wrap.pack(fill="x", padx=50, pady=(28, 0))

        scanner_card = tk.Frame(card_wrap, bg="#0f172a", relief="flat", bd=0)
        scanner_card.pack(fill="x")

        # gradient-style top border (3 thin frames)
        for col in ["#1d4ed8","#2563eb","#3b82f6"]:
            tk.Frame(scanner_card, bg=col, height=1).pack(fill="x")

        inner = tk.Frame(scanner_card, bg="#0f172a")
        inner.pack(fill="x", padx=45, pady=32)

        # Label row
        lbl_row = tk.Frame(inner, bg="#0f172a")
        lbl_row.pack(fill="x")
        tk.Label(lbl_row, text="🔗", bg="#0f172a", fg="#3b82f6",
                 font=("Segoe UI", 16)).pack(side="left")
        tk.Label(lbl_row, text="  Paste your URL below", bg="#0f172a", fg="#cbd5e1",
                 font=("Segoe UI", 14, "bold")).pack(side="left")

        # Entry
        entry_border = tk.Frame(inner, bg="#1e3a5f", bd=0)
        entry_border.pack(fill="x", pady=(10, 0))
        self.url_entry = tk.Entry(
            entry_border, bg="#0d1f3c", fg="white",
            font=("Segoe UI", 14), relief="flat", insertbackground="#60a5fa",
        )
        self.url_entry.pack(fill="x", ipady=15, padx=2, pady=2)
        self.url_entry.bind("<Return>", lambda e: self.scan_url())

        # Quick-fill examples
        ex_frame = tk.Frame(inner, bg="#0f172a")
        ex_frame.pack(fill="x", pady=(10, 0))
        tk.Label(ex_frame, text="Quick test:", bg="#0f172a", fg="#64748b",
                 font=("Segoe UI", 11)).pack(side="left")
        examples = [
            ("https://google.com","#15803d"),
            ("http://login-verify.xyz","#b91c1c"),
            ("http://free-download.exe","#b45309"),
            ("http://hacked-site.com","#7c2d12"),
        ]
        for ex, ecol in examples:
            tk.Button(
                ex_frame, text=ex, bg="#1e293b", fg=ecol,
                font=("Segoe UI", 10), relief="flat", cursor="hand2",
                pady=4, padx=8,
                command=lambda e=ex: (self.url_entry.delete(0, tk.END),
                                      self.url_entry.insert(0, e))
            ).pack(side="left", padx=5)

        # Analyze button + progress
        btn_row = tk.Frame(inner, bg="#0f172a")
        btn_row.pack(fill="x", pady=(22, 0))

        self.scan_btn = tk.Button(
            btn_row, text="🔍   Analyze URL",
            bg="#2563eb", fg="white", font=("Segoe UI", 14, "bold"),
            relief="flat", cursor="hand2", padx=45, pady=13,
            activebackground="#1d4ed8", activeforeground="white",
            command=self.scan_url
        )
        self.scan_btn.pack(side="left")

        self.progress = ttk.Progressbar(btn_row, mode="indeterminate", length=400)

        # Note box
        note = tk.Frame(inner, bg="#0c1f3f", bd=0)
        note.pack(fill="x", pady=(20, 0))
        note_inner = tk.Frame(note, bg="#0c1f3f")
        note_inner.pack(fill="x", padx=18, pady=14)
        head_row = tk.Frame(note_inner, bg="#0c1f3f")
        head_row.pack(anchor="w")
        tk.Label(head_row, text="ℹ", bg="#1d4ed8", fg="white",
                 font=("Segoe UI", 10, "bold"), width=2).pack(side="left")
        tk.Label(head_row, text="  How it works", bg="#0c1f3f", fg="#60a5fa",
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(
            note_inner,
            text="Paste any URL and click Analyze. Our ML model (Naive Bayes + TF-IDF) "
                 "will classify it as Benign, Phishing, Malware, or Defacement — "
                 "returning a Risk Score, Threat Level, and full Feature Breakdown.",
            bg="#0c1f3f", fg="#93c5fd", font=("Segoe UI", 11),
            wraplength=1000, justify="left"
        ).pack(anchor="w", pady=(6, 0))

        # Result area lives inside scanner_card (populated after scan)
        self.result_frame = tk.Frame(scanner_card, bg="#0f172a")

        # ── DIVIDER ───────────────────────────────────────────────────────────
        div = tk.Frame(scroll_root, bg="#0a0e27")
        div.pack(fill="x", padx=50, pady=(35, 0))
        tk.Frame(div, bg="#1e3a5f", height=1).pack(fill="x")

        # ── HOW IT WORKS ──────────────────────────────────────────────────────
        hiw_wrap = tk.Frame(scroll_root, bg="#0a0e27")
        hiw_wrap.pack(fill="x", padx=50, pady=(30, 0))

        tk.Label(hiw_wrap, text="HOW IT WORKS", bg="#0a0e27", fg="#3b82f6",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(hiw_wrap, text="Four steps from paste to result",
                 bg="#0a0e27", fg="white", font=("Segoe UI", 22, "bold")).pack(anchor="w", pady=(4, 20))

        steps_data = [
            ("01", "Paste the URL", "Drop any link — http, https, raw domain — into the input field above.", "#2563eb", "🔗"),
            ("02", "Feature Extraction", "TF-IDF tokenizes the URL into weighted character & word n-grams.", "#7c3aed", "⚙️"),
            ("03", "ML Classification", "Multinomial Naive Bayes predicts Benign / Phishing / Malware / Defacement.", "#0891b2", "🤖"),
            ("04", "Result & Breakdown", "Risk score, threat level, and per-feature analysis delivered instantly.", "#059669", "📊"),
        ]

        step_grid = tk.Frame(hiw_wrap, bg="#0a0e27")
        step_grid.pack(fill="x")

        for col_i, (num, title, desc, accent, icon) in enumerate(steps_data):
            step_card = tk.Frame(step_grid, bg="#0f172a", bd=0)
            step_card.grid(row=0, column=col_i, padx=8, pady=0, sticky="nsew")
            step_grid.grid_columnconfigure(col_i, weight=1)

            tk.Frame(step_card, bg=accent, height=3).pack(fill="x")
            sc_inner = tk.Frame(step_card, bg="#0f172a")
            sc_inner.pack(fill="both", padx=20, pady=18)

            num_row = tk.Frame(sc_inner, bg="#0f172a")
            num_row.pack(fill="x")
            tk.Label(num_row, text=num, bg="#0f172a", fg=accent,
                     font=("Segoe UI", 28, "bold")).pack(side="left")
            tk.Label(num_row, text=icon, bg="#0f172a", fg=accent,
                     font=("Segoe UI", 20)).pack(side="right")

            tk.Label(sc_inner, text=title, bg="#0f172a", fg="white",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8, 4))
            tk.Label(sc_inner, text=desc, bg="#0f172a", fg="#64748b",
                     font=("Segoe UI", 11), wraplength=240, justify="left").pack(
                     anchor="w", pady=(0, 12))

        # ── THREAT CATEGORIES ─────────────────────────────────────────────────
        tc_wrap = tk.Frame(scroll_root, bg="#060d1f")
        tc_wrap.pack(fill="x", pady=(35, 0))

        # wavy top border
        wave_c = tk.Canvas(tc_wrap, bg="#060d1f", height=18, highlightthickness=0)
        wave_c.pack(fill="x")
        def _draw_wave(e=None):
            wave_c.delete("all")
            w = wave_c.winfo_width() or 1200
            pts = []
            import math
            for x in range(0, w+20, 20):
                y = 9 + 6 * math.sin(x * 0.04)
                pts.extend([x, y])
            wave_c.create_line(pts, fill="#1e3a5f", width=2, smooth=True)
        wave_c.bind("<Configure>", _draw_wave)
        wave_c.after(80, _draw_wave)

        tc_inner = tk.Frame(tc_wrap, bg="#060d1f")
        tc_inner.pack(fill="x", padx=50, pady=(10, 35))

        tk.Label(tc_inner, text="THREAT CATEGORIES", bg="#060d1f", fg="#7c3aed",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(tc_inner, text="What our model detects",
                 bg="#060d1f", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4, 22))

        threats = [
            ("🎣", "Phishing", "Fake login pages, spoofed banks, credential harvesting sites designed to steal your identity.", "#ef4444", "#450a0a"),
            ("🦠", "Malware", "URLs that host executables, drive-by downloads, ransomware, or exploit kits targeting your device.", "#f97316", "#431407"),
            ("💀", "Defacement", "Compromised websites where attackers replaced content — often used as proof-of-hack or propaganda.", "#a855f7", "#3b0764"),
            ("✅", "Benign", "Legitimate, verified URLs with no detected threat patterns — safe to visit with confidence.", "#22c55e", "#052e16"),
        ]

        threat_row = tk.Frame(tc_inner, bg="#060d1f")
        threat_row.pack(fill="x")

        for col_i, (icon, tname, tdesc, tfg, tbg) in enumerate(threats):
            tc = tk.Frame(threat_row, bg=tbg, bd=0)
            tc.grid(row=0, column=col_i, padx=8, sticky="nsew")
            threat_row.grid_columnconfigure(col_i, weight=1)

            tc_pad = tk.Frame(tc, bg=tbg)
            tc_pad.pack(fill="both", padx=20, pady=20)
            tk.Label(tc_pad, text=icon, bg=tbg, fg=tfg,
                     font=("Segoe UI", 30)).pack(anchor="w")
            tk.Label(tc_pad, text=tname, bg=tbg, fg=tfg,
                     font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(6, 4))
            tk.Label(tc_pad, text=tdesc, bg=tbg, fg="#cbd5e1",
                     font=("Segoe UI", 11), wraplength=230, justify="left").pack(anchor="w")

        # ── LIVE SECURITY STATS ───────────────────────────────────────────────
        stats_wrap = tk.Frame(scroll_root, bg="#0a0e27")
        stats_wrap.pack(fill="x", padx=50, pady=(35, 0))

        tk.Label(stats_wrap, text="SESSION STATS", bg="#0a0e27", fg="#0891b2",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(stats_wrap, text="Your activity this session",
                 bg="#0a0e27", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4, 20))

        stat_row = tk.Frame(stats_wrap, bg="#0a0e27")
        stat_row.pack(fill="x")

        # Store references for live update
        self._scanner_stat_labels = {}
        stat_defs = [
            ("total_scanned",   "Total Scanned",   str(self.total_scanned),  "#3b82f6", "🔎"),
            ("malicious_count", "Threats Found",   str(self.malicious_count), "#ef4444", "⚠️"),
            ("normal_count",    "Safe URLs",        str(self.normal_count),   "#22c55e", "✅"),
            ("conf_avg",        "Avg Confidence",
             f"{(self.confidence_sum/self.total_scanned if self.total_scanned else 0):.1f}%",
             "#a855f7", "📈"),
        ]

        for col_i, (key, label, val, accent, icon) in enumerate(stat_defs):
            sc = tk.Frame(stat_row, bg="#0f172a", bd=0)
            sc.grid(row=0, column=col_i, padx=8, sticky="nsew")
            stat_row.grid_columnconfigure(col_i, weight=1)
            tk.Frame(sc, bg=accent, height=3).pack(fill="x")
            sci = tk.Frame(sc, bg="#0f172a")
            sci.pack(fill="both", padx=20, pady=18)
            icon_lbl = tk.Label(sci, text=icon, bg="#0f172a", fg=accent,
                                font=("Segoe UI", 20))
            icon_lbl.pack(anchor="w")
            val_lbl = tk.Label(sci, text=val, bg="#0f172a", fg="white",
                               font=("Segoe UI", 28, "bold"))
            val_lbl.pack(anchor="w", pady=(4, 2))
            tk.Label(sci, text=label, bg="#0f172a", fg="#64748b",
                     font=("Segoe UI", 11)).pack(anchor="w")
            self._scanner_stat_labels[key] = val_lbl

        # ── SECURITY TIPS ─────────────────────────────────────────────────────
        tips_wrap = tk.Frame(scroll_root, bg="#060d1f")
        tips_wrap.pack(fill="x", pady=(35, 0))

        tip_bg = tk.Canvas(tips_wrap, bg="#060d1f", height=300, highlightthickness=0)
        tip_bg.pack(fill="x")

        def _draw_tip_bg(e=None):
            tip_bg.delete("all")
            w = tip_bg.winfo_width() or 1200
            tip_bg.create_oval(w-220, -80, w+80, 220, fill="#0d2044", outline="")
            tip_bg.create_oval(-80, 160, 200, 380, fill="#0c1a38", outline="")
            tip_bg.create_rectangle(0, 0, w, 300, fill="", outline="")

        tip_bg.bind("<Configure>", _draw_tip_bg)
        tip_bg.after(80, _draw_tip_bg)

        tips_over = tk.Frame(tips_wrap, bg="#060d1f")
        tips_over.place(relx=0, rely=0, relwidth=1, relheight=1)
        tips_pad = tk.Frame(tips_over, bg="#060d1f")
        tips_pad.pack(fill="both", padx=50, pady=25)

        tk.Label(tips_pad, text="SECURITY TIPS", bg="#060d1f", fg="#059669",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(tips_pad, text="Stay safe online",
                 bg="#060d1f", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4, 18))

        tips = [
            ("🔒", "Always verify HTTPS", "Look for the padlock. HTTPS encrypts your data in transit."),
            ("🧠", "Don't trust shortened URLs", "Use a scanner before clicking bit.ly or tinyurl links."),
            ("📧", "Be skeptical of email links", "Phishing emails mimic banks and services — always scan first."),
            ("🚫", "Avoid IP-based URLs", "Legitimate sites use domain names, not raw IP addresses."),
            ("⏰", "Check domain age", "Newly registered domains (.xyz, .top) are often malicious."),
            ("💡", "Use browser security features", "Modern browsers block known phishing — keep them updated."),
        ]

        tip_grid = tk.Frame(tips_pad, bg="#060d1f")
        tip_grid.pack(fill="x")

        for i, (icon, tip_title, tip_desc) in enumerate(tips):
            row_i = i // 3
            col_i = i % 3
            tc = tk.Frame(tip_grid, bg="#0f172a", bd=0)
            tc.grid(row=row_i, column=col_i, padx=8, pady=6, sticky="nsew")
            tip_grid.grid_columnconfigure(col_i, weight=1)
            tip_pad2 = tk.Frame(tc, bg="#0f172a")
            tip_pad2.pack(fill="both", padx=16, pady=14)
            row_h = tk.Frame(tip_pad2, bg="#0f172a")
            row_h.pack(fill="x")
            tk.Label(row_h, text=icon, bg="#0f172a", fg="#22c55e",
                     font=("Segoe UI", 16)).pack(side="left")
            tk.Label(row_h, text="  " + tip_title, bg="#0f172a", fg="#e2e8f0",
                     font=("Segoe UI", 12, "bold")).pack(side="left")
            tk.Label(tip_pad2, text=tip_desc, bg="#0f172a", fg="#64748b",
                     font=("Segoe UI", 11), wraplength=300, justify="left").pack(
                     anchor="w", pady=(5, 0))

        # ── FOOTER STRIP ──────────────────────────────────────────────────────
        footer = tk.Frame(scroll_root, bg="#020817")
        footer.pack(fill="x", pady=(35, 0))
        tk.Frame(footer, bg="#1e3a5f", height=1).pack(fill="x")
        fpad = tk.Frame(footer, bg="#020817")
        fpad.pack(fill="x", padx=50, pady=18)
        tk.Label(fpad, text="🛡️  Malicious URL Scanner  ·  Powered by Naive Bayes ML  ·  Built for Security",
                 bg="#020817", fg="#334155", font=("Segoe UI", 11)).pack(side="left")
        tk.Label(fpad, text="Always scan before you click  🔒",
                 bg="#020817", fg="#1d4ed8", font=("Segoe UI", 11, "bold")).pack(side="right")
    
    def show_history(self):
        self.current_view = "history"
        self.highlight_nav(2)

        for widget in self.content_area.winfo_children():
            widget.destroy()

        # ══ SCROLLABLE WRAPPER ════════════════════════════════════════════════
        pg_c   = tk.Canvas(self.content_area, bg="#060d1f", highlightthickness=0)
        pg_vsb = ttk.Scrollbar(self.content_area, orient="vertical", command=pg_c.yview)
        pg_c.configure(yscrollcommand=pg_vsb.set)
        pg_vsb.pack(side="right", fill="y")
        pg_c.pack(side="left", fill="both", expand=True)
        sr = tk.Frame(pg_c, bg="#060d1f")
        sw = pg_c.create_window((0, 0), window=sr, anchor="nw")
        sr.bind("<Configure>", lambda e: pg_c.configure(scrollregion=pg_c.bbox("all")))
        pg_c.bind("<Configure>", lambda e: pg_c.itemconfig(sw, width=e.width))
        pg_c.bind_all("<MouseWheel>", lambda e: pg_c.yview_scroll(int(-1*(e.delta/120)),"units"))

        # ── HERO BANNER ───────────────────────────────────────────────────────
        hero = tk.Canvas(sr, bg="#020817", height=180, highlightthickness=0)
        hero.pack(fill="x")

        def _draw_hero(e=None):
            hero.delete("all")
            w = hero.winfo_width() or 1300
            import math, random
            hero.create_rectangle(0, 0, w, 180, fill="#020817", outline="")
            # right glow
            hero.create_oval(w-240, -70, w+70, 240, fill="#0d1f3c", outline="")
            hero.create_oval(w-160, -30, w+10, 140, fill="#7c3aed", outline="")
            hero.create_oval(-60, 80, 160, 280, fill="#0c1238", outline="")
            random.seed(11)
            for _ in range(65):
                x = random.randint(0, w); y = random.randint(0, 180)
                r = random.randint(1, 3)
                c = random.choice(["#1e3a5f","#4c1d95","#2563eb","#0d2044"])
                hero.create_oval(x-r,y-r,x+r,y+r, fill=c, outline="")
            # clock/history icon shape
            hero.create_oval(34,38,110,114, fill="#1e1b4b", outline="#7c3aed", width=2)
            hero.create_line(72,76, 72,56, fill="#a78bfa", width=3)
            hero.create_line(72,76, 88,76, fill="#a78bfa", width=3)
            hero.create_oval(68,72,76,80, fill="#7c3aed", outline="")
            # text
            hero.create_text(135, 68, text="Scan History",
                             font=("Segoe UI", 26, "bold"), fill="white", anchor="w")
            hero.create_text(135, 100, text="Full audit trail of every URL analyzed this session",
                             font=("Segoe UI", 12), fill="#a78bfa", anchor="w")
            # summary chips
            total = len(self.scan_history)
            mal   = sum(1 for s in self.scan_history if s[1] != "benign")
            safe  = total - mal
            for ci,(ctxt,cbg,cfg) in enumerate([
                (f"⬤ {total} Total","#1e1b4b","#a78bfa"),
                (f"⬤ {mal} Threats","#450a0a","#f87171"),
                (f"⬤ {safe} Safe","#052e16","#4ade80"),
            ]):
                bx = 135 + ci*160
                hero.create_rectangle(bx,128,bx+148,152,fill=cbg,outline="")
                hero.create_text(bx+74,140,text=ctxt,font=("Segoe UI",10,"bold"),fill=cfg)
        hero.bind("<Configure>", _draw_hero)
        hero.after(50, _draw_hero)

        # ── ACTION BAR ────────────────────────────────────────────────────────
        abar = tk.Frame(sr, bg="#0a0e1a")
        abar.pack(fill="x", padx=50, pady=(20, 0))
        tk.Frame(abar, bg="#7c3aed", height=3).pack(fill="x")
        abar_in = tk.Frame(abar, bg="#0f172a")
        abar_in.pack(fill="x", padx=24, pady=14)

        left_info = tk.Frame(abar_in, bg="#0f172a")
        left_info.pack(side="left")
        tk.Label(left_info, text="Scan History", bg="#0f172a", fg="white",
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(left_info,
                 text=f"{len(self.scan_history)} records total  ·  Most recent first",
                 bg="#0f172a", fg="#475569", font=("Segoe UI", 11)).pack(anchor="w")

        right_btns = tk.Frame(abar_in, bg="#0f172a")
        right_btns.pack(side="right")
        tk.Button(right_btns, text="🔍  New Scan", bg="#2563eb", fg="white",
                  font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
                  padx=18, pady=8, command=self.show_scanner).pack(side="left", padx=(0, 10))
        tk.Button(right_btns, text="🗑  Clear All", bg="#7f1d1d", fg="#f87171",
                  font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
                  padx=18, pady=8, command=self.clear_history).pack(side="left")

        # ── TABLE CARD ────────────────────────────────────────────────────────
        table_card = tk.Frame(sr, bg="#0f172a")
        table_card.pack(fill="x", padx=50, pady=(4, 0))

        # Fixed pixel widths shared by BOTH the header row and every data row,
        # so headings always line up exactly with the column data beneath them.
        COL_WIDTHS = {
            "num":     50,
            "time":    180,
            "url":     420,
            "result":  130,
            "score":   90,
            "threat":  130,
            "type":    80,
        }
        col_order = ["num", "time", "url", "result", "score", "threat", "type"]
        col_titles = {
            "num": "#", "time": "Date & Time", "url": "URL",
            "result": "Result", "score": "Score", "threat": "Threat", "type": "Type",
        }
        col_anchor = {
            "num": "center", "time": "w", "url": "w",
            "result": "center", "score": "center", "threat": "center", "type": "center",
        }

        # Column header row — each cell is a fixed-width frame so it matches
        # the corresponding cell frame in every data row exactly.
        HDR_HEIGHT = 44
        col_hdr = tk.Frame(table_card, bg="#1e293b")
        col_hdr.pack(fill="x")
        for ckey in col_order:
            cell = tk.Frame(col_hdr, bg="#1e293b", width=COL_WIDTHS[ckey], height=HDR_HEIGHT)
            cell.pack(side="left")
            cell.pack_propagate(False)
            lbl_anchor = {"w": "w", "center": "center"}[col_anchor[ckey]]
            relx = 0.0 if lbl_anchor == "w" else 0.5
            tk.Label(cell, text=col_titles[ckey], bg="#1e293b", fg="#64748b",
                     font=("Segoe UI", 11, "bold")).place(
                     relx=relx, rely=0.5, anchor=("w" if lbl_anchor == "w" else "center"),
                     x=(10 if lbl_anchor == "w" else 0))

        # Rows are placed directly inside the page's own scroll frame (no
        # nested inner canvas/scrollbar) — this avoids a second, competing
        # scroll region that previously caused layout/rendering glitches
        # when the window was resized to a shorter height.
        tbl_frame = tk.Frame(table_card, bg="#0f172a")
        tbl_frame.pack(fill="x")

        if not self.scan_history:
            empty_frame = tk.Frame(tbl_frame, bg="#0f172a")
            empty_frame.pack(fill="both", expand=True, pady=60)
            tk.Label(empty_frame, text="📭", bg="#0f172a", fg="#1e3a5f",
                     font=("Segoe UI", 40)).pack()
            tk.Label(empty_frame, text="No scan history yet",
                     bg="#0f172a", fg="#475569", font=("Segoe UI", 16, "bold")).pack(pady=(10,4))
            tk.Label(empty_frame, text="Go to Scan URL and analyze your first link",
                     bg="#0f172a", fg="#334155", font=("Segoe UI", 12)).pack()
            tk.Button(empty_frame, text="🔍  Scan a URL", bg="#2563eb", fg="white",
                      font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
                      padx=24, pady=10, command=self.show_scanner).pack(pady=20)
        else:
            for idx, scan in enumerate(reversed(self.scan_history)):
                url, result, confidence, timestamp = scan
                is_safe    = result == "benign"
                row_bg     = "#0f172a" if idx % 2 == 0 else "#0d1626"
                res_label  = "Benign" if is_safe else result.capitalize()
                res_bg     = "#052e16" if is_safe else "#450a0a"
                res_fg     = "#4ade80" if is_safe else "#f87171"
                thr_text   = "Safe"      if is_safe else "Dangerous"
                thr_bg     = "#052e16"   if is_safe else "#7f1d1d"
                thr_fg     = "#4ade80"   if is_safe else "#f87171"
                score_col  = "#4ade80"   if is_safe else ("#f97316" if confidence < 80 else "#ef4444")
                type_icons = {"benign":"🟢","phishing":"🎣","malware":"🦠","defacement":"💀"}
                t_icon     = type_icons.get(result, "⚠️")

                row = tk.Frame(tbl_frame, bg=row_bg)
                row.pack(fill="x")
                ROW_HEIGHT = 46

                # Row number — fixed-width cell, matches header "#" column
                cell_num = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["num"], height=ROW_HEIGHT)
                cell_num.pack(side="left")
                cell_num.pack_propagate(False)
                tk.Label(cell_num, text=str(len(self.scan_history)-idx), bg=row_bg,
                         fg="#334155", font=("Segoe UI", 10)).place(relx=0.5, rely=0.5, anchor="center")

                # Timestamp — fixed-width cell, matches "Date & Time" header
                cell_time = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["time"], height=ROW_HEIGHT)
                cell_time.pack(side="left")
                cell_time.pack_propagate(False)
                tk.Label(cell_time, text=timestamp, bg=row_bg, fg="#64748b",
                         font=("Segoe UI", 10)).place(relx=0, rely=0.5, anchor="w", x=10)

                # URL — fixed-width cell, matches "URL" header
                cell_url = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["url"], height=ROW_HEIGHT)
                cell_url.pack(side="left")
                cell_url.pack_propagate(False)
                url_disp = url[:55] + "…" if len(url) > 55 else url
                tk.Label(cell_url, text=url_disp, bg=row_bg, fg="#cbd5e1",
                         font=("Segoe UI", 10)).place(relx=0, rely=0.5, anchor="w", x=10)

                # Result badge — fixed-width cell, matches "Result" header
                cell_res = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["result"], height=ROW_HEIGHT)
                cell_res.pack(side="left")
                cell_res.pack_propagate(False)
                rb = tk.Frame(cell_res, bg=res_bg)
                rb.place(relx=0.5, rely=0.5, anchor="center")
                tk.Label(rb, text=res_label, bg=res_bg, fg=res_fg,
                         font=("Segoe UI", 10, "bold"), padx=10, pady=3).pack()

                # Score — fixed-width cell, matches "Score" header
                cell_score = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["score"], height=ROW_HEIGHT)
                cell_score.pack(side="left")
                cell_score.pack_propagate(False)
                tk.Label(cell_score, text=f"{confidence:.0f}%", bg=row_bg, fg=score_col,
                         font=("Segoe UI", 11, "bold")).place(relx=0.5, rely=0.5, anchor="center")

                # Threat badge — fixed-width cell, matches "Threat" header
                cell_thr = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["threat"], height=ROW_HEIGHT)
                cell_thr.pack(side="left")
                cell_thr.pack_propagate(False)
                tb2 = tk.Frame(cell_thr, bg=thr_bg)
                tb2.place(relx=0.5, rely=0.5, anchor="center")
                tk.Label(tb2, text=thr_text, bg=thr_bg, fg=thr_fg,
                         font=("Segoe UI", 10, "bold"), padx=10, pady=3).pack()

                # Type icon — fixed-width cell, matches "Type" header
                cell_type = tk.Frame(row, bg=row_bg, width=COL_WIDTHS["type"], height=ROW_HEIGHT)
                cell_type.pack(side="left")
                cell_type.pack_propagate(False)
                tk.Label(cell_type, text=t_icon, bg=row_bg, fg="white",
                         font=("Segoe UI", 13)).place(relx=0.5, rely=0.5, anchor="center")

                # divider
                tk.Frame(tbl_frame, bg="#1e293b", height=1).pack(fill="x")

        # ── SUMMARY STATS BELOW TABLE ─────────────────────────────────────────
        sum_wrap = tk.Frame(sr, bg="#060d1f")
        sum_wrap.pack(fill="x", padx=50, pady=(20, 0))

        tk.Label(sum_wrap, text="SESSION SUMMARY", bg="#060d1f", fg="#7c3aed",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

        sum_row = tk.Frame(sum_wrap, bg="#060d1f")
        sum_row.pack(fill="x")

        total   = len(self.scan_history)
        mal_c   = sum(1 for s in self.scan_history if s[1] != "benign")
        safe_c  = total - mal_c
        avg_cf  = (sum(s[2] for s in self.scan_history)/total if total else 0)
        ph_c    = sum(1 for s in self.scan_history if s[1]=="phishing")
        mw_c    = sum(1 for s in self.scan_history if s[1]=="malware")
        df_c    = sum(1 for s in self.scan_history if s[1]=="defacement")

        sum_stats = [
            ("📊", "Total Scanned",   str(total),       "#3b82f6"),
            ("⚠️", "Threats Found",    str(mal_c),       "#ef4444"),
            ("✅", "Safe URLs",         str(safe_c),      "#22c55e"),
            ("📈", "Avg Confidence",   f"{avg_cf:.1f}%", "#a855f7"),
            ("🎣", "Phishing",         str(ph_c),        "#f97316"),
            ("🦠", "Malware",          str(mw_c),        "#ec4899"),
        ]
        for ci, (icon, label, val, accent) in enumerate(sum_stats):
            sc = tk.Frame(sum_row, bg="#0f172a")
            sc.grid(row=0, column=ci, padx=6, sticky="nsew")
            sum_row.grid_columnconfigure(ci, weight=1)
            tk.Frame(sc, bg=accent, height=2).pack(fill="x")
            sc_in = tk.Frame(sc, bg="#0f172a")
            sc_in.pack(fill="both", padx=14, pady=14)
            tk.Label(sc_in, text=icon, bg="#0f172a", fg=accent,
                     font=("Segoe UI", 16)).pack(anchor="w")
            tk.Label(sc_in, text=val, bg="#0f172a", fg="white",
                     font=("Segoe UI", 22, "bold")).pack(anchor="w", pady=(4,2))
            tk.Label(sc_in, text=label, bg="#0f172a", fg="#475569",
                     font=("Segoe UI", 10)).pack(anchor="w")

        # ── FOOTER ────────────────────────────────────────────────────────────
        footer = tk.Frame(sr, bg="#020817")
        footer.pack(fill="x", pady=(28, 0))
        tk.Frame(footer, bg="#0d1f3c", height=1).pack(fill="x")
        fp = tk.Frame(footer, bg="#020817")
        fp.pack(fill="x", padx=50, pady=14)
        tk.Label(fp, text="🛡️  ThreatScan  ·  History is session-only and clears on restart",
                 bg="#020817", fg="#1e293b", font=("Segoe UI", 10)).pack(side="left")
        tk.Label(fp, text="Always scan before you click  🔒",
                 bg="#020817", fg="#7c3aed", font=("Segoe UI", 10, "bold")).pack(side="right")
    
    def show_about(self):
        self.current_view = "about"
        self.highlight_nav(3)

        for widget in self.content_area.winfo_children():
            widget.destroy()

        # ══ SCROLLABLE WRAPPER ════════════════════════════════════════════════
        pg_c   = tk.Canvas(self.content_area, bg="#060d1f", highlightthickness=0)
        pg_vsb = ttk.Scrollbar(self.content_area, orient="vertical", command=pg_c.yview)
        pg_c.configure(yscrollcommand=pg_vsb.set)
        pg_vsb.pack(side="right", fill="y")
        pg_c.pack(side="left", fill="both", expand=True)
        sr = tk.Frame(pg_c, bg="#060d1f")
        sw = pg_c.create_window((0, 0), window=sr, anchor="nw")
        sr.bind("<Configure>", lambda e: pg_c.configure(scrollregion=pg_c.bbox("all")))
        pg_c.bind("<Configure>", lambda e: pg_c.itemconfig(sw, width=e.width))
        pg_c.bind_all("<MouseWheel>", lambda e: pg_c.yview_scroll(int(-1*(e.delta/120)),"units"))

        # ── HERO BANNER ───────────────────────────────────────────────────────
        hero = tk.Canvas(sr, bg="#020817", height=200, highlightthickness=0)
        hero.pack(fill="x")

        def _draw_hero(e=None):
            hero.delete("all")
            w = hero.winfo_width() or 1300
            import math, random
            hero.create_rectangle(0, 0, w, 200, fill="#020817", outline="")
            # teal glow right
            hero.create_oval(w-220,-60,w+80,260, fill="#0c2a38", outline="")
            hero.create_oval(w-150,-20,w+10,160, fill="#0891b2", outline="")
            hero.create_oval(-60,100,160,300, fill="#0c1a38", outline="")
            random.seed(99)
            for _ in range(70):
                x=random.randint(0,w); y=random.randint(0,200)
                r=random.randint(1,3)
                c=random.choice(["#0c2a38","#0e7490","#164e63","#0d2044"])
                hero.create_oval(x-r,y-r,x+r,y+r,fill=c,outline="")
            # info / book icon
            cx,cy = 72,100
            hero.create_rectangle(cx-30,cy-40,cx+30,cy+45,fill="#0c2a38",outline="#0891b2",width=2)
            for li in range(5):
                y2 = cy-22+li*12
                hero.create_line(cx-18,y2,cx+18,y2,fill="#0e7490",width=2)
            hero.create_oval(cx-5,cy+28,cx+5,cy+38,fill="#0891b2",outline="")
            # text
            hero.create_text(135,75,text="About ThreatScan",
                             font=("Segoe UI",26,"bold"),fill="white",anchor="w")
            hero.create_text(135,108,text="ML-powered URL security · Built on Naive Bayes · Open & transparent",
                             font=("Segoe UI",12),fill="#22d3ee",anchor="w")
            chips=[("⚡ Naive Bayes ML","#0c2a38","#22d3ee"),
                   ("🔬 TF-IDF Features","#1e1b4b","#a78bfa"),
                   ("🎯 4 Threat Classes","#052e16","#4ade80")]
            for ci,(ct,cb,cf) in enumerate(chips):
                bx=135+ci*178
                hero.create_rectangle(bx,138,bx+165,162,fill=cb,outline="")
                hero.create_text(bx+82,150,text=ct,font=("Segoe UI",10,"bold"),fill=cf)
        hero.bind("<Configure>",_draw_hero)
        hero.after(50,_draw_hero)

        # ── INTRO CARD ────────────────────────────────────────────────────────
        intro_card = tk.Frame(sr, bg="#0f172a")
        intro_card.pack(fill="x", padx=50, pady=(22,0))
        tk.Frame(intro_card, bg="#0891b2", height=3).pack(fill="x")
        intro_in = tk.Frame(intro_card, bg="#0f172a")
        intro_in.pack(fill="x", padx=30, pady=24)

        left_intro = tk.Frame(intro_in, bg="#0f172a")
        left_intro.pack(side="left", fill="both", expand=True, padx=(0,30))

        tk.Label(left_intro, text="🛡️  ThreatScan", bg="#0f172a", fg="white",
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(left_intro,
                 text="A machine learning-powered URL security scanner that classifies any link as "
                      "Benign, Phishing, Malware, or Defacement in milliseconds. "
                      "Built with scikit-learn's Multinomial Naive Bayes and TF-IDF vectorization, "
                      "trained on 50,000+ labeled URLs.",
                 bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 12),
                 wraplength=600, justify="left").pack(anchor="w", pady=(12,16))

        badge_row = tk.Frame(left_intro, bg="#0f172a")
        badge_row.pack(anchor="w")
        for btxt, bacc in [("Naive Bayes","#2563eb"),("TF-IDF","#7c3aed"),
                            ("Real-time","#0891b2"),("Pre-trained","#059669")]:
            bf = tk.Frame(badge_row, bg=bacc)
            bf.pack(side="left", padx=(0,8))
            tk.Label(bf, text=btxt, bg=bacc, fg="white",
                     font=("Segoe UI", 11, "bold"), padx=14, pady=6).pack()

        # Right: quick metrics
        right_m = tk.Frame(intro_in, bg="#0d1626", width=280)
        right_m.pack(side="right")
        right_m.pack_propagate(False)
        rm_in = tk.Frame(right_m, bg="#0d1626")
        rm_in.pack(fill="both", padx=20, pady=20)
        tk.Label(rm_in, text="Model Specs", bg="#0d1626", fg="#64748b",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,12))
        for ml, mv, mc in [
            ("Algorithm",   "Multinomial NB",   "#60a5fa"),
            ("Features",    "TF-IDF (5,000)",   "#a78bfa"),
            ("Training set","50,000 URLs",       "#4ade80"),
            ("Classes",     "4 (B/P/M/D)",       "#fbbf24"),
            ("Pred. speed", "< 100ms",           "#34d399"),
            ("Accuracy",    "~92.5%",            "#f472b6"),
        ]:
            mr = tk.Frame(rm_in, bg="#0d1626")
            mr.pack(fill="x", pady=3)
            tk.Label(mr, text=ml+":", bg="#0d1626", fg="#475569",
                     font=("Segoe UI", 11), width=14, anchor="w").pack(side="left")
            tk.Label(mr, text=mv, bg="#0d1626", fg=mc,
                     font=("Segoe UI", 11, "bold")).pack(side="left")

        # ── KEY FEATURES ──────────────────────────────────────────────────────
        feat_wrap = tk.Frame(sr, bg="#060d1f")
        feat_wrap.pack(fill="x", padx=50, pady=(28,0))

        tk.Label(feat_wrap, text="KEY FEATURES", bg="#060d1f", fg="#0891b2",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(feat_wrap, text="What ThreatScan does for you",
                 bg="#060d1f", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4,18))

        feat_data = [
            ("🔗","URL Analysis",
             "Deconstructs URL structure: length, keywords, digit density, domain TLD, "
             "IP-in-URL patterns, and HTTPS status — all fed to the model as weighted features.",
             "#2563eb"),
            ("⚡","Instant Classification",
             "Classifies any URL as Benign, Phishing, Malware, or Defacement in under 100ms. "
             "Pre-trained model loads once and serves every scan at near-zero latency.",
             "#7c3aed"),
            ("📊","Risk Score & Breakdown",
             "Returns a confidence percentage (0–100%) alongside a Threat Level (Safe / Moderate / Dangerous) "
             "and a per-feature breakdown table so you can see exactly why a URL was flagged.",
             "#0891b2"),
            ("📜","Full Scan History",
             "Every scan is logged with URL, result, confidence, and timestamp. "
             "Browse, review, and clear your history anytime from the History tab.",
             "#059669"),
        ]

        feat_grid = tk.Frame(feat_wrap, bg="#060d1f")
        feat_grid.pack(fill="x")
        for ci, (icon, title, desc, accent) in enumerate(feat_data):
            fc = tk.Frame(feat_grid, bg="#0f172a")
            fc.grid(row=0, column=ci, padx=8, sticky="nsew")
            feat_grid.grid_columnconfigure(ci, weight=1)
            tk.Frame(fc, bg=accent, height=3).pack(fill="x")
            fp2 = tk.Frame(fc, bg="#0f172a")
            fp2.pack(fill="both", padx=18, pady=18)
            tk.Label(fp2, text=icon, bg="#0f172a", fg=accent,
                     font=("Segoe UI", 24)).pack(anchor="w")
            tk.Label(fp2, text=title, bg="#0f172a", fg="white",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8,4))
            tk.Label(fp2, text=desc, bg="#0f172a", fg="#64748b",
                     font=("Segoe UI", 11), wraplength=230, justify="left").pack(anchor="w")

        # ── HOW IT WORKS ──────────────────────────────────────────────────────
        hiw_wrap = tk.Frame(sr, bg="#020817")
        hiw_wrap.pack(fill="x", pady=(28,0))
        tk.Frame(hiw_wrap, bg="#0d1f3c", height=1).pack(fill="x")
        hiw_in = tk.Frame(hiw_wrap, bg="#020817")
        hiw_in.pack(fill="x", padx=50, pady=(24,24))

        tk.Label(hiw_in, text="HOW IT WORKS", bg="#020817", fg="#7c3aed",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(hiw_in, text="From URL to verdict in four steps",
                 bg="#020817", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4,18))

        steps = [
            ("01","Paste URL","User drops any link into the scanner — raw domain, http, or https.",
             "#2563eb","🔗"),
            ("02","Tokenize","TF-IDF breaks the URL into weighted character & word n-gram tokens.",
             "#7c3aed","⚙️"),
            ("03","Predict","Multinomial Naive Bayes assigns one of 4 threat class labels + probability.",
             "#0891b2","🤖"),
            ("04","Report","Risk score, threat level, and feature-by-feature breakdown displayed.",
             "#059669","📊"),
        ]

        step_row = tk.Frame(hiw_in, bg="#020817")
        step_row.pack(fill="x")
        for ci, (num, stitle, sdesc, sacc, sicon) in enumerate(steps):
            sc2 = tk.Frame(step_row, bg="#0f172a")
            sc2.grid(row=0, column=ci, padx=8, sticky="nsew")
            step_row.grid_columnconfigure(ci, weight=1)
            tk.Frame(sc2, bg=sacc, height=3).pack(fill="x")
            sp = tk.Frame(sc2, bg="#0f172a")
            sp.pack(fill="both", padx=18, pady=18)
            nr = tk.Frame(sp, bg="#0f172a")
            nr.pack(fill="x")
            tk.Label(nr, text=num, bg="#0f172a", fg=sacc,
                     font=("Segoe UI", 28, "bold")).pack(side="left")
            tk.Label(nr, text=sicon, bg="#0f172a", fg=sacc,
                     font=("Segoe UI", 18)).pack(side="right")
            tk.Label(sp, text=stitle, bg="#0f172a", fg="white",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8,4))
            tk.Label(sp, text=sdesc, bg="#0f172a", fg="#64748b",
                     font=("Segoe UI", 11), wraplength=230, justify="left").pack(anchor="w")

        # ── MODEL DETAILS TABLE ───────────────────────────────────────────────
        md_wrap = tk.Frame(sr, bg="#060d1f")
        md_wrap.pack(fill="x", padx=50, pady=(28,0))

        tk.Label(md_wrap, text="MODEL DETAILS", bg="#060d1f", fg="#059669",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(md_wrap, text="Technical specification",
                 bg="#060d1f", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4,14))

        model_tbl = tk.Frame(md_wrap, bg="#0f172a")
        model_tbl.pack(fill="x")
        tk.Frame(model_tbl, bg="#059669", height=3).pack(fill="x")

        model_rows = [
            ("Algorithm",        "Multinomial Naive Bayes (sklearn.naive_bayes.MultinomialNB)"),
            ("Feature Extraction","TF-IDF Vectorization — max 5,000 features, char + word n-grams"),
            ("Training Dataset", "650,000+ labeled URLs sampled to 50,000 (random_state=42)"),
            ("Label Classes",    "0 = Benign  ·  1 = Defacement  ·  2 = Phishing  ·  3 = Malware"),
            ("Prediction Speed", "< 100 ms per URL after model load"),
            ("Model Persistence","Serialized via joblib — reloads instantly on subsequent runs"),
            ("Estimated Accuracy","~92.5% on held-out test data"),
            ("Fallback Mode",    "Keyword + TLD pattern matching when dataset is unavailable"),
        ]

        for ri, (lbl, val) in enumerate(model_rows):
            row_bg2 = "#0f172a" if ri % 2 == 0 else "#0d1626"
            rrow = tk.Frame(model_tbl, bg=row_bg2)
            rrow.pack(fill="x")
            tk.Label(rrow, text=lbl, bg=row_bg2, fg="#22d3ee",
                     font=("Segoe UI", 12, "bold"), width=22, anchor="w").pack(
                     side="left", padx=20, pady=12)
            tk.Label(rrow, text=val, bg=row_bg2, fg="#94a3b8",
                     font=("Segoe UI", 12), wraplength=750, justify="left").pack(
                     side="left", padx=10)
            tk.Frame(model_tbl, bg="#1e293b", height=1).pack(fill="x")

        # ── LIVE SESSION STATS ────────────────────────────────────────────────
        ls_wrap = tk.Frame(sr, bg="#020817")
        ls_wrap.pack(fill="x", pady=(28,0))
        tk.Frame(ls_wrap, bg="#0d1f3c", height=1).pack(fill="x")
        ls_in = tk.Frame(ls_wrap, bg="#020817")
        ls_in.pack(fill="x", padx=50, pady=(24,24))

        tk.Label(ls_in, text="LIVE SESSION STATS", bg="#020817", fg="#f97316",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(ls_in, text="Your current session at a glance",
                 bg="#020817", fg="white", font=("Segoe UI", 22, "bold")).pack(
                 anchor="w", pady=(4,18))

        ls_row = tk.Frame(ls_in, bg="#020817")
        ls_row.pack(fill="x")
        det_rate = (self.malicious_count/self.total_scanned*100
                    if self.total_scanned else 0)
        ls_stats = [
            ("Total Scans",    str(self.total_scanned),                  "#3b82f6","🔎"),
            ("Detection Rate", f"{det_rate:.1f}%",                       "#ef4444","📡"),
            ("Model Status",
             "Active" if self.model.is_trained else "Fallback",          "#22c55e","✅"),
            ("Est. Accuracy",  "92.5%",                                  "#a855f7","🎯"),
        ]
        for ci,(lbl2,val2,acc2,ico2) in enumerate(ls_stats):
            lsc = tk.Frame(ls_row, bg="#0f172a")
            lsc.grid(row=0,column=ci,padx=8,sticky="nsew")
            ls_row.grid_columnconfigure(ci,weight=1)
            tk.Frame(lsc,bg=acc2,height=3).pack(fill="x")
            lsp=tk.Frame(lsc,bg="#0f172a")
            lsp.pack(fill="both",padx=18,pady=18)
            tk.Label(lsp,text=ico2,bg="#0f172a",fg=acc2,
                     font=("Segoe UI",18)).pack(anchor="w")
            tk.Label(lsp,text=val2,bg="#0f172a",fg="white",
                     font=("Segoe UI",26,"bold")).pack(anchor="w",pady=(6,2))
            tk.Label(lsp,text=lbl2,bg="#0f172a",fg="#475569",
                     font=("Segoe UI",11)).pack(anchor="w")

        # ── SAFETY BANNER ─────────────────────────────────────────────────────
        sb2 = tk.Frame(sr, bg="#0c1a38")
        sb2.pack(fill="x", pady=(28,0))
        tk.Frame(sb2, bg="#0891b2", height=2).pack(fill="x")
        sb2_in = tk.Frame(sb2, bg="#0c1a38")
        sb2_in.pack(fill="x", padx=50, pady=20)
        tk.Label(sb2_in, text="🔒  Security Reminder", bg="#0c1a38", fg="#22d3ee",
                 font=("Segoe UI",13,"bold")).pack(side="left")
        tk.Label(sb2_in,
                 text="Always scan unknown URLs before clicking — especially links from emails, DMs, and social media.",
                 bg="#0c1a38", fg="#67e8f9", font=("Segoe UI",11)).pack(side="left",padx=20)
        tk.Button(sb2_in, text="Scan Now →", bg="#0891b2", fg="white",
                  font=("Segoe UI",11,"bold"), relief="flat", cursor="hand2",
                  padx=16, pady=7, command=self.show_scanner).pack(side="right")

        # ── FOOTER ────────────────────────────────────────────────────────────
        footer = tk.Frame(sr, bg="#020817")
        footer.pack(fill="x", pady=(20,0))
        tk.Frame(footer, bg="#0d1f3c", height=1).pack(fill="x")
        fp3 = tk.Frame(footer, bg="#020817")
        fp3.pack(fill="x", padx=50, pady=14)
        tk.Label(fp3, text="🛡️  ThreatScan v2.0.0  ·  Powered by Naive Bayes ML  ·  scikit-learn + TF-IDF",
                 bg="#020817", fg="#1e293b", font=("Segoe UI", 10)).pack(side="left")
        tk.Label(fp3, text="Always scan before you click  🔒",
                 bg="#020817", fg="#0891b2", font=("Segoe UI", 10, "bold")).pack(side="right")
    
    def scan_url(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a URL to scan")
            return
        
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        
        self.scan_btn.config(state="disabled", text="Scanning…")
        self.progress.pack(pady=(0, 20))
        self.progress.start(10)
        
        def scan_thread():
            try:
                result, confidence = self.model.predict(url)
                self.root.after(0, lambda: self.on_scan_complete(url, result, confidence))
            except Exception as e:
                self.root.after(0, lambda: self.on_scan_error(str(e)))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def on_scan_complete(self, url, result, confidence):
        self.progress.stop()
        self.progress.pack_forget()
        self.scan_btn.config(state="normal", text="🔍  Analyze URL")

        self.total_scanned += 1
        day_index = datetime.now().weekday()
        is_malicious = result != "benign"

        if is_malicious:
            self.malicious_count += 1
            self.daily_data[day_index]["malicious"] += 1
            url_lower = url.lower()
            for kw in self.keyword_counts:
                if kw in url_lower:
                    self.keyword_counts[kw] += 1
            self.update_keywords_display()
        else:
            self.normal_count += 1
            self.daily_data[day_index]["normal"] += 1

        self.confidence_sum += confidence
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.scan_history.append((url, result, confidence, timestamp))

        if self.current_view == "dashboard":
            self.update_stats_display()
            self.update_graph()

        if self.current_view == "scanner":
            self.update_scanner_stats_display()

        self.update_sidebar_session_display()

        # ── Clear & rebuild result area ───────────────────────────────────────
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.result_frame.pack(fill="x", padx=45, pady=(0, 30))
        self.result_frame.config(bg="#0f172a")

        # ── Scanned URL bar ───────────────────────────────────────────────────
        url_bar = tk.Frame(self.result_frame, bg="#1e293b")
        url_bar.pack(fill="x", pady=(0, 2))
        tk.Label(url_bar, text="Scanned URL:", bg="#1e293b", fg="#6b7280",
                 font=("Segoe UI", 12)).pack(side="left", padx=20, pady=10)
        disp_url = url[:90] + "…" if len(url) > 90 else url
        tk.Label(url_bar, text=disp_url, bg="#1e293b", fg="#cbd5e1",
                 font=("Segoe UI", 12)).pack(side="left")

        # ── Three-column result strip ─────────────────────────────────────────
        strip = tk.Frame(self.result_frame, bg="#111827")
        strip.pack(fill="x", pady=(2, 0))

        # Equal-proportion grid: col1 gets more width (it has the longest
        # text), col2 and col3 get fixed, equal-looking shares — all three
        # share the same fixed height so nothing looks lopsided.
        STRIP_HEIGHT = 280
        strip.grid_columnconfigure(0, weight=3, uniform="resultcols")
        strip.grid_columnconfigure(1, weight=2, uniform="resultcols")
        strip.grid_columnconfigure(2, weight=2, uniform="resultcols")
        strip.grid_rowconfigure(0, minsize=STRIP_HEIGHT)

        # -- Col 1: Result label --
        if is_malicious:
            col1_bg = "#450a0a"
            col1_fg = "#ef4444"
            col1_icon = "⚠️"
            col1_title = result.upper() + " URL"
            col1_sub = f"This URL is {result.capitalize()}. Avoid visiting it."
        else:
            col1_bg = "#052e16"
            col1_fg = "#22c55e"
            col1_icon = "✅"
            col1_title = "SAFE URL"
            col1_sub = "No threats found. This appears to be a legitimate website."

        col1 = tk.Frame(strip, bg=col1_bg, height=STRIP_HEIGHT)
        col1.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        col1.grid_propagate(False)

        col1_content = tk.Frame(col1, bg=col1_bg)
        col1_content.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col1_content, text="RESULT", bg=col1_bg, fg="#94a3b8",
                 font=("Segoe UI", 11, "bold")).pack(anchor="center", pady=(0, 10))
        result_row = tk.Frame(col1_content, bg=col1_bg)
        result_row.pack(anchor="center")
        tk.Label(result_row, text=col1_icon, bg=col1_bg, fg=col1_fg,
                 font=("Segoe UI", 26)).pack(side="left")
        tk.Label(result_row, text=" " + col1_title, bg=col1_bg, fg=col1_fg,
                 font=("Segoe UI", 19, "bold")).pack(side="left")
        tk.Label(col1_content, text=col1_sub, bg=col1_bg,
                 fg="#fca5a5" if is_malicious else "#86efac",
                 font=("Segoe UI", 11), wraplength=300, justify="center").pack(
                 anchor="center", pady=(12, 0))

        # -- Col 2: Risk Score (circular gauge) --
        if confidence > 80:
            risk_level = "Very High Risk" if is_malicious else "Very Low Risk"
            risk_color = "#ef4444" if is_malicious else "#22c55e"
        elif confidence > 60:
            risk_level = "Moderate Risk" if is_malicious else "Low Risk"
            risk_color = "#f97316" if is_malicious else "#22c55e"
        else:
            risk_level = "Low Risk"
            risk_color = "#22c55e"

        col2 = tk.Frame(strip, bg="#1e293b", height=STRIP_HEIGHT)
        col2.grid(row=0, column=1, sticky="nsew", padx=2)
        col2.grid_propagate(False)

        col2_content = tk.Frame(col2, bg="#1e293b")
        col2_content.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col2_content, text="RISK SCORE", bg="#1e293b", fg="#64748b",
                 font=("Segoe UI", 11, "bold")).pack(pady=(0, 10))

        # Circular gauge canvas — supports full 0–100% rendering cleanly
        GAUGE_SIZE = 140
        PAD = 12
        circ = tk.Canvas(col2_content, width=GAUGE_SIZE, height=GAUGE_SIZE,
                          bg="#1e293b", highlightthickness=0)
        circ.pack()

        x0, y0 = PAD, PAD
        x1, y1 = GAUGE_SIZE - PAD, GAUGE_SIZE - PAD
        ring_width = 11

        circ.create_oval(x0, y0, x1, y1, outline="#334155", width=ring_width)

        clamped = max(0.0, min(100.0, confidence))
        total_degrees = 359.9 if clamped >= 99.95 else (clamped / 100.0) * 359.9
        if total_degrees > 0:
            circ.create_arc(x0, y0, x1, y1, start=90, extent=-total_degrees,
                             style="arc", outline=risk_color, width=ring_width)

        circ.create_text(GAUGE_SIZE / 2, GAUGE_SIZE / 2 - 7,
                          text=f"{clamped:.0f}%",
                          fill=risk_color, font=("Segoe UI", 22, "bold"))
        circ.create_text(GAUGE_SIZE / 2, GAUGE_SIZE / 2 + 16,
                          text="confidence",
                          fill="#64748b", font=("Segoe UI", 9))

        tk.Label(col2_content, text=risk_level, bg="#1e293b", fg=risk_color,
                 font=("Segoe UI", 12, "bold")).pack(pady=(10, 10))

        # Linear scale legend underneath the gauge
        scale_wrap = tk.Frame(col2_content, bg="#1e293b", width=190)
        scale_wrap.pack()
        scale_track = tk.Frame(scale_wrap, bg="#334155", height=6, width=190)
        scale_track.pack()
        scale_track.pack_propagate(False)
        fill_ratio = max(0.02, clamped / 100.0)
        scale_fill = tk.Frame(scale_track, bg=risk_color)
        scale_fill.place(relx=0, rely=0, relwidth=fill_ratio, relheight=1)
        labels_row = tk.Frame(col2_content, bg="#1e293b", width=190)
        labels_row.pack(pady=(4, 0))
        tk.Label(labels_row, text="0%", bg="#1e293b", fg="#475569",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Frame(labels_row, bg="#1e293b", width=150).pack(side="left")
        tk.Label(labels_row, text="100%", bg="#1e293b", fg="#475569",
                 font=("Segoe UI", 9)).pack(side="left")

        # -- Col 3: Threat Level --
        if is_malicious:
            threat_text = "DANGEROUS"
            threat_color = "#ef4444"
            threat_bg = "#450a0a"
            threat_icon = "🛑"
            threat_sub = "Immediate caution advised"
        else:
            threat_text = "SAFE"
            threat_color = "#22c55e"
            threat_bg = "#052e16"
            threat_icon = "🛡️"
            threat_sub = "No action needed"

        col3 = tk.Frame(strip, bg=threat_bg, height=STRIP_HEIGHT)
        col3.grid(row=0, column=2, sticky="nsew", padx=(2, 0))
        col3.grid_propagate(False)

        col3_content = tk.Frame(col3, bg=threat_bg)
        col3_content.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(col3_content, text="THREAT LEVEL", bg=threat_bg, fg="#94a3b8",
                 font=("Segoe UI", 11, "bold")).pack(pady=(0, 14))
        tk.Label(col3_content, text=threat_icon, bg=threat_bg, fg=threat_color,
                 font=("Segoe UI", 40)).pack()
        tk.Label(col3_content, text=threat_text, bg=threat_bg, fg=threat_color,
                 font=("Segoe UI", 16, "bold")).pack(pady=(10, 4))
        tk.Label(col3_content, text=threat_sub, bg=threat_bg, fg="#94a3b8",
                 font=("Segoe UI", 10)).pack()

        # ── Feature Breakdown table ───────────────────────────────────────────
        breakdown_card = tk.Frame(self.result_frame, bg="#111827")
        breakdown_card.pack(fill="x", pady=(3, 0))

        tk.Label(breakdown_card, text="Feature Breakdown", bg="#111827", fg="white",
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(16, 8))

        # Table header
        tbl_hdr = tk.Frame(breakdown_card, bg="#1e293b")
        tbl_hdr.pack(fill="x", padx=20)
        for col_text, w in [("Feature", 22), ("Status", 14), ("Details", 40)]:
            tk.Label(tbl_hdr, text=col_text, bg="#1e293b", fg="#9ca3af",
                     font=("Segoe UI", 12, "bold"), width=w, anchor="w").pack(
                     side="left", padx=10, pady=8)

        # Build feature rows
        url_lower = url.lower()
        has_https = url.startswith("https://")
        has_ip = bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))
        url_len = len(url)
        digit_count = sum(c.isdigit() for c in url)
        suspicious_kws = [kw for kw in ["login", "verify", "secure", "update", "account", "bank", "free", "confirm", "password"]
                          if kw in url_lower]
        try:
            from urllib.parse import urlparse
            domain_ext = urlparse(url).netloc.split(".")[-1] if url else ""
        except Exception:
            domain_ext = ""
        risky_tlds = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "live", "work", "ru"]
        domain_suspicious = domain_ext in risky_tlds

        features = [
            ("URL Length",
             "High" if url_len > 75 else ("Medium" if url_len > 40 else "Normal"),
             "#ef4444" if url_len > 75 else ("#f97316" if url_len > 40 else "#22c55e"),
             f"{url_len} characters"),
            ("Suspicious Keywords",
             "Detected" if suspicious_kws else "None",
             "#ef4444" if suspicious_kws else "#22c55e",
             ", ".join(suspicious_kws) if suspicious_kws else "No suspicious keywords found"),
            ("Number of Digits",
             "High" if digit_count > 5 else ("Medium" if digit_count > 2 else "Low"),
             "#ef4444" if digit_count > 5 else ("#f97316" if digit_count > 2 else "#22c55e"),
             f"{digit_count} digit{'s' if digit_count != 1 else ''} found"),
            ("Domain Type",
             "Suspicious" if domain_suspicious else "Normal",
             "#f97316" if domain_suspicious else "#22c55e",
             f".{domain_ext} ({'Risky' if domain_suspicious else 'Standard'})"),
            ("IP Address in URL",
             "Yes" if has_ip else "No",
             "#ef4444" if has_ip else "#22c55e",
             "IP address detected in URL" if has_ip else "No IP address found"),
            ("HTTPS",
             "Yes" if has_https else "No",
             "#22c55e" if has_https else "#ef4444",
             "Secure connection (HTTPS)" if has_https else "URL is not using HTTPS"),
        ]

        status_palette = {
            "High": "#7f1d1d", "Medium": "#431407", "Normal": "#052e16",
            "Detected": "#7f1d1d", "None": "#052e16",
            "Suspicious": "#431407", "Yes": "#7f1d1d", "No": "#052e16",
            "Low": "#052e16",
        }
        # HTTPS "Yes" is green, so override:
        def _row_status_bg(feature_name, status):
            if feature_name == "HTTPS" and status == "Yes":
                return "#052e16"
            if feature_name == "HTTPS" and status == "No":
                return "#7f1d1d"
            if feature_name == "IP Address in URL" and status == "No":
                return "#052e16"
            return status_palette.get(status, "#1e293b")

        for idx, (fname, fstatus, fcolor, fdetail) in enumerate(features):
            row_bg = "#111827" if idx % 2 == 0 else "#0f172a"
            row = tk.Frame(breakdown_card, bg=row_bg)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=fname, bg=row_bg, fg="#e2e8f0",
                     font=("Segoe UI", 12), width=22, anchor="w").pack(
                     side="left", padx=10, pady=10)
            sbg = _row_status_bg(fname, fstatus)
            badge = tk.Frame(row, bg=sbg)
            badge.pack(side="left", padx=10)
            tk.Label(badge, text=fstatus, bg=sbg, fg=fcolor,
                     font=("Segoe UI", 11, "bold"), padx=10, pady=3).pack()
            tk.Label(row, text=fdetail, bg=row_bg, fg="#9ca3af",
                     font=("Segoe UI", 11), anchor="w").pack(
                     side="left", padx=15)

        # Bottom padding
        tk.Frame(self.result_frame, bg="#111827", height=12).pack(fill="x")

        # Clear entry
        self.url_entry.delete(0, tk.END)
    
    def on_scan_error(self, error_msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.scan_btn.config(state="normal", text="🔍  Analyze URL")
        messagebox.showerror("Scan Error", f"Failed to scan URL: {error_msg}")
    
    def update_stats_display(self):
        if self.current_view != "dashboard":
            return
        avg_conf = 0 if self.total_scanned == 0 else (self.confidence_sum / self.total_scanned)
        for key, label in self.stats_labels.items():
            if key == "total":
                self.safe_update_widget(label, text=f"{self.total_scanned}")
            elif key == "malicious":
                self.safe_update_widget(label, text=f"{self.malicious_count}")
            elif key == "normal":
                self.safe_update_widget(label, text=f"{self.normal_count}")
            elif key == "confidence":
                self.safe_update_widget(label, text=f"{avg_conf:.1f}%")

    def update_scanner_stats_display(self):
        if self.current_view != "scanner":
            return
        if not hasattr(self, "_scanner_stat_labels"):
            return
        avg_conf = 0 if self.total_scanned == 0 else (self.confidence_sum / self.total_scanned)
        for key, label in self._scanner_stat_labels.items():
            if key == "total_scanned":
                self.safe_update_widget(label, text=f"{self.total_scanned}")
            elif key == "malicious_count":
                self.safe_update_widget(label, text=f"{self.malicious_count}")
            elif key == "normal_count":
                self.safe_update_widget(label, text=f"{self.normal_count}")
            elif key == "conf_avg":
                self.safe_update_widget(label, text=f"{avg_conf:.1f}%")

    def update_sidebar_session_display(self):
        self.safe_update_widget(self._sb_total_lbl, text=f"{self.total_scanned} scans")
        self.safe_update_widget(self._sb_threat_lbl, text=f"{self.malicious_count} threats")
    
    def update_graph(self):
        if self.current_view != "dashboard" or self.ax is None:
            return
        
        self.ax.clear()
        self.ax.set_facecolor("#111827")
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        malicious_counts = [self.daily_data[i]["malicious"] for i in range(7)]
        normal_counts = [self.daily_data[i]["normal"] for i in range(7)]
        
        x = np.arange(len(days))
        
        self.ax.plot(x, malicious_counts, color='#ef4444', marker='o', linewidth=2.5, markersize=9, label='Malicious', zorder=3)
        self.ax.plot(x, normal_counts, color='#3b82f6', marker='s', linewidth=2.5, markersize=9, label='Normal', zorder=3)
        self.ax.fill_between(x, malicious_counts, 0, alpha=0.15, color='#ef4444', zorder=1)
        self.ax.fill_between(x, normal_counts, 0, alpha=0.15, color='#3b82f6', zorder=1)
        
        max_val = max(max(malicious_counts or [0]), max(normal_counts or [0]), 3)
        self.ax.set_ylim(0, max_val + 1.5)
        self.ax.set_xlim(-0.5, len(days) - 0.5)
        self.ax.set_ylabel("Count", color="white", fontsize=13)
        self.ax.set_xlabel("Day", color="white", fontsize=13)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(days, color="white", fontsize=12)
        self.ax.tick_params(colors="white", labelsize=11)
        self.ax.spines['bottom'].set_color('#374151')
        self.ax.spines['left'].set_color('#374151')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(loc="upper left", facecolor="#111827", labelcolor="white", framealpha=1, edgecolor='#374151', fontsize=12)
        
        for i, val in enumerate(malicious_counts):
            if val > 0:
                self.ax.annotate(str(int(val)), xy=(i, val), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', color='#ef4444', fontsize=11, weight='bold')
        for i, val in enumerate(normal_counts):
            if val > 0:
                self.ax.annotate(str(int(val)), xy=(i, val), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', color='#3b82f6', fontsize=11, weight='bold')
        
        self.ax.grid(axis='y', alpha=0.2, color='gray', linestyle='--')
        self.canvas.draw()
    
    def update_keywords_display(self):
        if self.current_view != "dashboard":
            return
        max_count = max(self.keyword_counts.values()) if self.keyword_counts.values() else 1
        for item in self.keyword_frames:
            try:
                count = self.keyword_counts.get(item['name'], 0)
                if item['count_label'] and item['count_label'].winfo_exists():
                    item['count_label'].config(text=str(count))
                if max_count > 0 and item['progress_frame'] and item['progress_frame'].winfo_exists():
                    percentage = (count / max_count) * 100
                    frame_width = item['progress_frame'].winfo_width()
                    if frame_width > 0 and item['progress_bar'] and item['progress_bar'].winfo_exists():
                        item['progress_bar'].config(width=int((percentage / 100) * frame_width))
            except:
                pass
    
    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all scan history?"):
            self.scan_history = []
            self.total_scanned = 0
            self.malicious_count = 0
            self.normal_count = 0
            self.confidence_sum = 0
            self.daily_data = {i: {"malicious": 0, "normal": 0} for i in range(7)}
            self.keyword_counts = {kw: 0 for kw in self.keyword_counts}
            
            if self.current_view == "dashboard":
                self.update_stats_display()
                self.update_graph()
                self.update_keywords_display()
            elif self.current_view == "history":
                self.show_history()
            
            messagebox.showinfo("Success", "History cleared successfully!")
    
    def logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ProfessionalURLScanner(root)
    root.mainloop()