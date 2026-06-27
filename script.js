/* ─────────────────────────────────────────────────────────────────
   script.js  —  Handwriting Recognizer
   Handles: canvas drawing · undo · brush size · predict API call · UI updates
   ───────────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const canvas      = document.getElementById("drawCanvas");
  const ctx         = canvas.getContext("2d");
  const hint        = document.getElementById("canvasHint");

  const brushSlider = document.getElementById("brushSize");
  const brushVal    = document.getElementById("brushVal");

  const clearBtn    = document.getElementById("clearBtn");
  const undoBtn     = document.getElementById("undoBtn");
  const predictBtn  = document.getElementById("predictBtn");
  const retryBtn    = document.getElementById("retryBtn");

  const emptyState  = document.getElementById("emptyState");
  const loadingState= document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");
  const errorState  = document.getElementById("errorState");

  const bigDigit    = document.getElementById("bigDigit");
  const confPill    = document.getElementById("confPill");
  const confFill    = document.getElementById("confFill");
  const confPct     = document.getElementById("confPct");
  const top3List    = document.getElementById("top3List");
  const digitGrid   = document.getElementById("digitGrid");
  const errorMsg    = document.getElementById("errorMsg");

  // ── State ─────────────────────────────────────────────────────────────────
  let isDrawing   = false;
  let lastX       = 0;
  let lastY       = 0;
  let hasDrawn    = false;
  let strokeHistory = [];   // array of ImageData snapshots for undo

  // ── Canvas initialisation ─────────────────────────────────────────────────
  function initCanvas() {
    ctx.fillStyle = "#0b0d1a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    applyBrushSettings();
  }

  function applyBrushSettings() {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth   = parseInt(brushSlider.value, 10);
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
    ctx.shadowColor = "rgba(167,139,250,0.6)";
    ctx.shadowBlur  = 6;
  }

  initCanvas();

  // ── Brush size ────────────────────────────────────────────────────────────
  brushSlider.addEventListener("input", () => {
    brushVal.textContent = brushSlider.value;
    applyBrushSettings();
  });

  // ── Coordinate helper (handles HiDPI & CSS scaling) ───────────────────────
  function getPos(e) {
    const rect  = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return [
      (clientX - rect.left) * scaleX,
      (clientY - rect.top)  * scaleY
    ];
  }

  // ── Drawing ───────────────────────────────────────────────────────────────
  function startDraw(e) {
    // Save snapshot BEFORE this stroke (for undo)
    strokeHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));

    isDrawing = true;
    [lastX, lastY] = getPos(e);

    // Draw a dot at the click/touch point (handles taps)
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();

    canvas.classList.add("is-drawing");

    // Hide hint on first draw
    if (!hasDrawn) {
      hint.classList.add("hidden");
      hasDrawn = true;
    }
  }

  function draw(e) {
    if (!isDrawing) return;
    const [x, y] = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
  }

  function stopDraw() {
    isDrawing = false;
    canvas.classList.remove("is-drawing");
  }

  // Mouse events
  canvas.addEventListener("mousedown",  startDraw);
  canvas.addEventListener("mousemove",  draw);
  canvas.addEventListener("mouseup",    stopDraw);
  canvas.addEventListener("mouseleave", stopDraw);

  // Touch events (mobile)
  canvas.addEventListener("touchstart", (e) => { e.preventDefault(); startDraw(e); }, { passive: false });
  canvas.addEventListener("touchmove",  (e) => { e.preventDefault(); draw(e); },      { passive: false });
  canvas.addEventListener("touchend",   stopDraw);

  // ── Clear ─────────────────────────────────────────────────────────────────
  function clearCanvas() {
    strokeHistory = [];
    hasDrawn = false;
    hint.classList.remove("hidden");
    initCanvas();
    showPanel(emptyState);
  }
  clearBtn.addEventListener("click", clearCanvas);

  // ── Undo ──────────────────────────────────────────────────────────────────
  undoBtn.addEventListener("click", () => {
    if (strokeHistory.length === 0) return;
    const prev = strokeHistory.pop();
    ctx.putImageData(prev, 0, 0);
    if (strokeHistory.length === 0) {
      hasDrawn = false;
      hint.classList.remove("hidden");
    }
    applyBrushSettings();
  });

  // ── Panel helpers ─────────────────────────────────────────────────────────
  const allPanels = [emptyState, loadingState, resultState, errorState];

  function showPanel(panel) {
    allPanels.forEach(p => {
      if (p === panel) p.classList.remove("hidden");
      else             p.classList.add("hidden");
    });
  }

  // ── Predict ───────────────────────────────────────────────────────────────
  predictBtn.addEventListener("click", runPredict);
  retryBtn.addEventListener("click", runPredict);

  async function runPredict() {
    if (!hasDrawn) {
      canvas.style.borderColor = "var(--coral)";
      setTimeout(() => (canvas.style.borderColor = ""), 1000);
      return;
    }

    showPanel(loadingState);

    // Get canvas image as base64 PNG
    const imageData = canvas.toDataURL("image/png");

    try {
      const response = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image: imageData })
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `Server error ${response.status}`);
      }

      const data = await response.json();

      if (data.error) throw new Error(data.error);

      renderResult(data);

    } catch (err) {
      console.error("Prediction error:", err);
      errorMsg.textContent = err.message || "Unknown error. Try again.";
      showPanel(errorState);
    }
  }

  // ── Render result ─────────────────────────────────────────────────────────
  function renderResult(data) {
    // Big digit
    bigDigit.textContent  = data.prediction;
    confPill.textContent  = data.confidence + "%";
    confPct.textContent   = data.confidence + "%";

    // Confidence bar (animated via CSS transition)
    requestAnimationFrame(() => {
      confFill.style.width = data.confidence + "%";
    });

    // Top-3 list
    top3List.innerHTML = data.top3.map(item => `
      <div class="top3-row">
        <span class="top3-lbl">${item.label}</span>
        <div class="top3-bar">
          <div class="top3-bar-fill" style="width: ${item.confidence}%"></div>
        </div>
        <span class="top3-pct">${item.confidence}%</span>
      </div>
    `).join("");

    // Build full probability grid for all 10 digits
    // data.top3 only has top 3; we need all 10 — reconstruct from server response
    // The server returns top3; we build the grid and highlight the best match
    const top3Map = {};
    data.top3.forEach(d => { top3Map[d.label] = d.confidence; });

    digitGrid.innerHTML = Array.from({ length: 10 }, (_, i) => {
      const lbl    = String(i);
      const pct    = top3Map[lbl] !== undefined ? top3Map[lbl] : 0;
      const active = lbl === data.prediction ? " active" : "";
      return `
        <div class="digit-cell${active}">
          <div class="d-num">${lbl}</div>
          <div class="d-pct">${pct > 0 ? pct + "%" : "—"}</div>
        </div>
      `;
    }).join("");

    showPanel(resultState);
  }

})();
