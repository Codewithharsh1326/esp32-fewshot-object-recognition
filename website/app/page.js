"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { GlowingEffect } from "./components/glowing-effect";
import { SnappySlider } from "./components/snappy-slider";

const DEFAULT_THRESHOLD = 0.75;
const POLL_INTERVAL = 3000;
const MAX_HISTORY = 30;

// ============ Canvas Line Chart ============
function LineChart({ data, label, color, unit, height = 180 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, w, h);

    const padL = 50, padR = 16, padT = 16, padB = 32;
    const cw = w - padL - padR;
    const ch = h - padT - padB;

    const maxVal = Math.max(...data, 1) * 1.15;
    const minVal = 0;

    // Grid lines
    const gridLines = 4;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.textAlign = "right";
    for (let i = 0; i <= gridLines; i++) {
      const y = padT + (ch / gridLines) * i;
      const val = maxVal - ((maxVal - minVal) / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + cw, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(0), padL - 8, y + 4);
    }

    // X-axis labels
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    const step = Math.max(1, Math.floor(data.length / 6));
    for (let i = 0; i < data.length; i += step) {
      const x = padL + (i / Math.max(data.length - 1, 1)) * cw;
      ctx.fillText(`#${i + 1}`, x, h - 8);
    }

    if (data.length < 2) {
      // Single dot
      const x = padL + cw / 2;
      const y = padT + ch - (data[0] / maxVal) * ch;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      return;
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, padT, 0, padT + ch);
    grad.addColorStop(0, color + "30");
    grad.addColorStop(1, color + "00");

    ctx.beginPath();
    ctx.moveTo(padL, padT + ch);
    data.forEach((val, i) => {
      const x = padL + (i / (data.length - 1)) * cw;
      const y = padT + ch - (val / maxVal) * ch;
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(padL + cw, padT + ch);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = padL + (i / (data.length - 1)) * cw;
      const y = padT + ch - (val / maxVal) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Dots
    data.forEach((val, i) => {
      const x = padL + (i / (data.length - 1)) * cw;
      const y = padT + ch - (val / maxVal) * ch;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Latest value label
    if (data.length > 0) {
      const last = data[data.length - 1];
      const x = padL + cw;
      const y = padT + ch - (last / maxVal) * ch;
      ctx.font = "bold 13px 'JetBrains Mono', monospace";
      ctx.fillStyle = color;
      ctx.textAlign = "right";
      ctx.fillText(`${last.toFixed(0)}${unit}`, x - 4, y - 10);
    }
  }, [data, color, unit]);

  return (
    <div className="chart-container">
      <div className="chart-header">
        <span className="chart-label">{label}</span>
        <span className="chart-latest" style={{ color }}>
          {data.length > 0 ? `${data[data.length - 1].toFixed(0)}${unit}` : "—"}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: `${height}px`, display: "block" }}
      />
    </div>
  );
}

// ============ Embedding Chart ============
function EmbeddingChart({ refs, test, height = 240 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 16, padT = 20, padB = 30;
    const cw = w - padL - padR;
    const ch = h - padT - padB;

    // Range: standard MobileNetV2 embeddings often fall in [-1, 1] or similar.
    // We'll auto-scale or fix range. -1 to 1 is safe for normalized embeddings.
    const minVal = -1.0;
    const maxVal = 1.0;
    const range = maxVal - minVal;

    // Draw grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.textAlign = "right";

    // Y-axis grid
    for (let i = 0; i <= 4; i++) {
      const y = padT + (ch / 4) * i;
      const val = maxVal - (range / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + cw, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), padL - 8, y + 3);
    }

    // Zero line
    const y0 = padT + ch - ((0 - minVal) / range) * ch;
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.beginPath();
    ctx.moveTo(padL, y0);
    ctx.lineTo(padL + cw, y0);
    ctx.stroke();

    // Zero line highlight
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.beginPath();
    ctx.moveTo(padL, y0);
    ctx.lineTo(padL + cw, y0);
    ctx.stroke();

    const drawBars = (data, color, widthScale = 1, opacity = 1, isOutline = false) => {
      if (!data || data.length === 0) return;

      const barW = cw / data.length;
      const gap = barW > 3 ? 1 : 0;
      const finalW = Math.max(1, barW - gap) * widthScale;

      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.globalAlpha = opacity;

      data.forEach((val, i) => {
        const x = padL + i * barW + (barW - finalW) / 2;
        const yTop = padT + ch - ((val - minVal) / range) * ch;
        const hBar = Math.abs(yTop - y0);
        const yStart = val >= 0 ? yTop : y0;

        if (isOutline) {
          ctx.lineWidth = 1;
          ctx.strokeRect(x, yStart, finalW, hBar);
        } else {
          ctx.fillRect(x, yStart, finalW, hBar);
        }
      });
      ctx.globalAlpha = 1.0;
    };

    // Draw References (Cyan/Blue bars, faint)
    if (refs && refs.length) {
      refs.forEach((ref) => {
        drawBars(ref, "#00d4ff", 0.8, 0.25);
      });
    }

    // Draw Test (Purple, slightly narrower, solid or outlined)
    if (test && test.length > 0) {
      drawBars(test, "#d946ef", 0.6, 0.6);
      drawBars(test, "#ffffff", 0.6, 0.4, true);
    }

    // X-axis label
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.textAlign = "center";
    ctx.fillText("Embedding Dimensions (0-127)", padL + cw / 2, h - 8);

    // Legend
    ctx.textAlign = "left";
    ctx.fillStyle = "#00d4ff";
    ctx.fillText("References", w - 120, 14);
    ctx.fillStyle = "#d946ef";
    ctx.fillText("Test Input", w - 120, 26);

  }, [refs, test, height]);

  return <canvas ref={canvasRef} style={{ width: "100%", height }} />;
}

// ============ Similarity Gauge (FIXED — no overlap) ============
function SimilarityGauge({ value, label, colorClass, size = 140 }) {
  const r = (size - 20) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (Math.max(0, Math.min(1, value)) * circ);
  const center = size / 2;

  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle className="gauge-bg" cx={center} cy={center} r={r} />
        <circle
          className={`gauge-fill ${colorClass}`}
          cx={center} cy={center} r={r}
          strokeDasharray={circ}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-value" style={{ color: `var(--accent-${colorClass})`, fontSize: size > 120 ? 24 : 18 }}>
          {(value * 100).toFixed(1)}%
        </span>
        <span className="gauge-label">{label}</span>
      </div>
    </div>
  );
}

// ============ Glow Card Wrapper ============
function GlowCard({ children, className = "" }) {
  return (
    <div className={`glow-card ${className}`}>
      <GlowingEffect
        spread={40}
        glow={true}
        disabled={false}
        proximity={64}
        inactiveZone={0.01}
        borderWidth={2}
      />
      {children}
    </div>
  );
}

// ============ Metric Card ============
function MetricCard({ icon, iconColor, title, subtitle, value, unit, colorClass, barPercent, barColor }) {
  return (
    <GlowCard>
      <div className="card">
        <div className="card-header">
          <div className={`card-icon ${iconColor}`}>{icon}</div>
          <div>
            <div className="card-title">{title}</div>
            {subtitle && <div className="card-subtitle">{subtitle}</div>}
          </div>
        </div>
        <div className={`metric-value ${colorClass}`}>{value}</div>
        <div className="metric-unit">{unit}</div>
        {barPercent !== undefined && (
          <div className="metric-bar">
            <div className={`metric-bar-fill ${barColor || colorClass}`}
              style={{ width: `${Math.min(barPercent, 100)}%` }} />
          </div>
        )}
      </div>
    </GlowCard>
  );
}

// ============ Toast ============
function Toast({ message, type, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3000);
    return () => clearTimeout(t);
  }, [onClose]);
  return <div className={`toast ${type}`}>{message}</div>;
}

// ============ Image Canvas (RGB888 Base64) ============
function ImageCanvas({ base64, width = 96, height = 96, scale = 2 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!base64 || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');

    // Decode Base64 to binary string
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Create ImageData (RGB888 -> RGBA)
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let i = 0; i < width * height; i++) {
      // Source is 3 bytes (RGB)
      // Dest is 4 bytes (RGBA)
      const srcIdx = i * 3;
      const dstIdx = i * 4;
      if (srcIdx + 2 < len) {
        data[dstIdx] = bytes[srcIdx];     // R
        data[dstIdx + 1] = bytes[srcIdx + 1]; // G
        data[dstIdx + 2] = bytes[srcIdx + 2]; // B
        data[dstIdx + 3] = 255;           // Alpha
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [base64, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: width * scale,
        height: height * scale,
        borderRadius: 8,
        border: '1px solid rgba(255,255,255,0.2)',
        imageRendering: 'pixelated'
      }}
    />
  );
}

// ============ Interactive History Chart ============
function HistoryChart({ history, height = 180 }) {
  const canvasRef = useRef(null);
  const [hoverData, setHoverData] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    const padL = 35, padR = 10, padT = 10, padB = 20;
    const cw = w - padL - padR;
    const ch = h - padT - padB;

    // Y-Axis Range: [-1, 1]
    const minVal = -1.0;
    const maxVal = 1.0;
    const range = maxVal - minVal;

    const getY = (sim) => padT + ch - ((sim - minVal) / range) * ch;
    const y0 = getY(0);

    // Draws Axis Labels & Grid
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    // Y-Axis Labels
    [1.0, 0.5, 0.0, -0.5, -1.0].forEach(val => {
      const y = getY(val);
      ctx.fillText(`${val * 100}%`, padL - 6, y);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.strokeStyle = val === 0 ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // X-Axis Labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const step = Math.ceil(history.length / 6);
    history.forEach((_, i) => {
      if (i % step === 0 || i === history.length - 1) {
        const x = padL + (i / (history.length - 1 || 1)) * cw;
        ctx.fillText(`${i + 1}`, x, h - padB + 4);
      }
    });

    // Gradient fill
    const grad = ctx.createLinearGradient(0, padT, 0, padT + ch);
    grad.addColorStop(0, "rgba(0, 212, 255, 0.2)");
    grad.addColorStop(1, "rgba(0, 212, 255, 0)");

    // Path
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = padL + (i / (history.length - 1 || 1)) * cw;
      const y = getY(h.similarity);
      if (i === 0) ctx.moveTo(x, y0); // Optional: area fill logic
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    // ctx.lineTo(padL + cw, padT + ch);
    // ctx.closePath();
    // ctx.fillStyle = grad;
    // ctx.fill();

    // Line
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = padL + (i / (history.length - 1 || 1)) * cw;
      const y = getY(h.similarity);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Dots
    history.forEach((h, i) => {
      const x = padL + (i / (history.length - 1 || 1)) * cw;
      const y = getY(h.similarity);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = h.similarity >= h.localThreshold ? "#10b981" : "#00d4ff";
      ctx.fill();
    });

  }, [history, height]);

  const handleMouseMove = (e) => {
    if (history.length === 0) return;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const w = rect.width;
    const idx = Math.min(Math.max(Math.round((x / w) * (history.length - 1)), 0), history.length - 1);
    setHoverData(history[idx]);
  };

  const MetricItem = ({ label, value, color }) => (
    <div style={{
      background: "rgba(255,255,255,0.05)",
      borderRadius: "8px",
      padding: "8px 12px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minWidth: "70px"
    }}>
      <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: "0.5px" }}>{label}</span>
      <span style={{ fontSize: "13px", fontWeight: "bold", color: color || "white", marginTop: "2px" }}>{value}</span>
    </div>
  );

  return (
    <div
      className="chart-container"
      style={{ position: "relative", cursor: "crosshair", overflow: "visible" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverData(null)}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: `${height}px`, display: "block" }} />

      {hoverData && (
        <div style={{
          position: "absolute",
          bottom: "0",
          left: "0",
          right: "0",
          background: "rgba(10, 10, 15, 0.90)",
          backdropFilter: "blur(4px)",
          borderTop: "1px solid rgba(255,255,255,0.1)",
          padding: "8px",
          display: "flex",
          gap: "4px",
          justifyContent: "space-between",
          alignItems: "center",
          zIndex: 20,
          pointerEvents: "none"
        }}>
          <MetricItem label="Sim" value={`${(hoverData.similarity * 100).toFixed(1)}%`} color={hoverData.similarity >= hoverData.localThreshold ? "#10b981" : "#ff4d4d"} />
          <MetricItem label="Thresh" value={`${(hoverData.localThreshold * 100).toFixed(0)}%`} color="#ffffff" />
          {/* <MetricItem label="Dist" value={`${((1 - hoverData.similarity) * 100).toFixed(1)}%`} color="#d946ef" /> */}
          <MetricItem label="Cap" value={`${hoverData.capture_ms}ms`} color="#fbbf24" />
          <MetricItem label="Inf" value={`${hoverData.inference_ms}ms`} color="#00d4ff" />
          <MetricItem label="Total" value={`${hoverData.total_ms}ms`} color="#ec4899" />
        </div>
      )}
    </div>
  );
}

const MAX_HISTORY_LEN = 30;

export default function Dashboard() {
  // Connection
  const [espIp, setEspIp] = useState("");
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connError, setConnError] = useState("");

  // Metrics
  const [metrics, setMetrics] = useState(null);

  // Recognition
  const [numRefs, setNumRefs] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [refEmbeddings, setRefEmbeddings] = useState([]);
  const [testEmbedding, setTestEmbedding] = useState([]);

  // New History Object State
  const [history, setHistory] = useState([]);
  const [refImages, setRefImages] = useState([]);
  const [testImage, setTestImage] = useState(null);

  // Legacy History (for small charts)
  const [inferenceHistory, setInferenceHistory] = useState([]);
  const [captureHistory, setCaptureHistory] = useState([]);
  const [similarityHistory, setSimilarityHistory] = useState([]);

  // Toast
  const [toast, setToast] = useState(null);
  const pollRef = useRef(null);
  const baseUrl = espIp ? `http://${espIp}` : "";

  // ---- API ----
  const apiCall = useCallback(async (path, method = "GET") => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const res = await fetch(`${baseUrl}${path}`, { method, signal: controller.signal });
      clearTimeout(timeout);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) { clearTimeout(timeout); throw e; }
  }, [baseUrl]);

  const showToast = (message, type = "success") => setToast({ message, type });

  // ---- Connect ----
  const handleConnect = async () => {
    if (!espIp.trim()) return;
    setConnecting(true); setConnError(""); setConnected(false);
    try {
      const data = await apiCall("/status");
      setNumRefs(data.num_references || 0);
      setConnected(true);
      showToast(`Connected to ESP32 at ${espIp}`);
      fetchMetrics();
    } catch (e) {
      setConnError(`Cannot connect to ${espIp}`);
      showToast(`Connection failed: ${e.message}`, "error");
    } finally { setConnecting(false); }
  };

  const handleDisconnect = () => {
    setConnected(false); setMetrics(null); setNumRefs(0);
    setTestResult(null); setInferenceHistory([]); setCaptureHistory([]); setSimilarityHistory([]);
    if (pollRef.current) clearInterval(pollRef.current);
    showToast("Disconnected");
  };

  // ---- Metrics polling ----
  const fetchMetrics = useCallback(async () => {
    if (!baseUrl) return;
    try {
      const data = await apiCall("/metrics");
      setMetrics(data);
      setNumRefs(data.num_references || 0);
    } catch { /* silent */ }
  }, [apiCall, baseUrl]);

  useEffect(() => {
    if (connected) {
      fetchMetrics();
      pollRef.current = setInterval(fetchMetrics, POLL_INTERVAL);
      return () => clearInterval(pollRef.current);
    }
  }, [connected, fetchMetrics]);

  // ---- Capture ----
  const handleCapture = async () => {
    setCapturing(true);
    try {
      const data = await apiCall("/capture", "POST");
      if (data.success) {
        setNumRefs(data.num_references);
        showToast(`Reference ${data.num_references} captured! (${data.inference_ms}ms)`);
        // Add to history
        if (data.inference_ms) {
          setInferenceHistory(prev => [...prev.slice(-MAX_HISTORY + 1), data.inference_ms]);
        }
        if (data.capture_ms) {
          setCaptureHistory(prev => [...prev.slice(-MAX_HISTORY + 1), data.capture_ms]);
        }
        if (data.embedding) {
          setRefEmbeddings(prev => [...prev, data.embedding]);
        }
        if (data.image) setRefImages(prev => [...prev, data.image]);
        fetchMetrics();
      } else { showToast(data.message || "Capture failed", "error"); }
    } catch (e) { showToast(`Capture error: ${e.message}`, "error"); }
    finally { setCapturing(false); }
  };

  // ---- Test ----
  const handleTest = async () => {
    setTesting(true); setTestResult(null);
    try {
      const data = await apiCall("/test", "POST");
      if (data.success) {
        const isMatch = data.similarity >= threshold;
        setTestResult({ ...data, match: isMatch, localThreshold: threshold });
        // Add to history
        if (data.inference_ms) {
          setInferenceHistory(prev => [...prev.slice(-MAX_HISTORY + 1), data.inference_ms]);
        }
        if (data.capture_ms) {
          setCaptureHistory(prev => [...prev.slice(-MAX_HISTORY + 1), data.capture_ms]);
        }
        if (data.embedding) {
          setTestEmbedding(data.embedding);
        }
        if (data.image) setTestImage(data.image);
        if (data.ref_embeddings) {
          setRefEmbeddings(data.ref_embeddings);
        }
        setSimilarityHistory(prev => [...prev.slice(-MAX_HISTORY + 1), data.similarity * 100]);

        // Update new history object
        setHistory(prev => {
          const newItem = {
            similarity: data.similarity,
            localThreshold: threshold,
            capture_ms: data.capture_ms,
            inference_ms: data.inference_ms,
            total_ms: data.total_ms || (data.capture_ms + data.inference_ms),
            timestamp: Date.now()
          };
          return [...prev.slice(-MAX_HISTORY_LEN + 1), newItem];
        });

        fetchMetrics();
      } else { showToast(data.message || "Test failed", "error"); }
    } catch (e) { showToast(`Test error: ${e.message}`, "error"); }
    finally { setTesting(false); }
  };

  // ---- Reset ----
  const handleReset = async () => {
    try {
      await apiCall("/reset", "POST");
      setNumRefs(0); setTestResult(null); setSimilarityHistory([]);
      setRefEmbeddings([]); setTestEmbedding([]);
      setHistory([]); // Clear history
      setRefImages([]); setTestImage(null);
      showToast("References cleared"); fetchMetrics();
    } catch (e) { showToast(`Reset error: ${e.message}`, "error"); }
  };

  // ---- Computed ----
  const psramUsedPercent = metrics ? ((metrics.psram_total - metrics.psram_free) / metrics.psram_total * 100) : 0;

  const formatUptime = (sec) => {
    if (!sec) return "0s";
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  const getSimColor = (val) => val >= threshold ? "green" : val >= threshold * 0.8 ? "amber" : "red";

  return (
    <main className="dashboard">
      {/* ============ HEADER ============ */}
      <header className="header">
        <h1>Object Recognition Dashboard</h1>
        <p>MobileNetV2 α0.35 INT8 • 128-dim Embeddings • Real-time Inference</p>
      </header>

      {/* ============ CONNECTION ============ */}
      <section className={`connection-panel ${connected ? "connected" : connError ? "error" : ""}`}>
        <div className="connection-row">
          <label>🔗 ESP32 IP</label>
          <input className="ip-input" type="text" placeholder="192.168.1.xxx"
            value={espIp} onChange={(e) => setEspIp(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !connected && handleConnect()}
            disabled={connected} />
          {!connected ? (
            <button className="btn btn-primary" onClick={handleConnect} disabled={connecting || !espIp.trim()}>
              {connecting ? "⏳ Connecting..." : "⚡ Connect"}
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleDisconnect}>⛔ Disconnect</button>
          )}
        </div>
        <div className="connection-status">
          <div className={`status-dot ${connected ? "online" : connError ? "error" : "offline"}`} />
          <span style={{ color: connected ? "var(--accent-green)" : connError ? "var(--accent-red)" : "var(--text-muted)" }}>
            {connected ? `Connected to ${espIp}` : connError || "Not connected"}
          </span>
        </div>
      </section>

      {connected && (
        <>
          {/* ============ 1. OBJECT RECOGNITION (FIRST) ============ */}
          <div className="section-header">
            <h2>🎯 Object Recognition</h2>
            <div className="line" />
          </div>

          <div className="recognition-panel">
            {/* Ref Images are now in Step 1 */}

            {/* Threshold */}
            <div className="threshold-section">
              <SnappySlider
                values={[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]}
                defaultValue={DEFAULT_THRESHOLD}
                value={threshold}
                onChange={setThreshold}
                min={0}
                max={1}
                step={0.01}
                snapping={true}
                snappingThreshold={0.015}
                label="Match Threshold"
              />
            </div>

            <div className="grid-2" style={{ marginBottom: 0 }}>
              {/* Step 1 */}
              <div className="step-section">
                <div className="step-label">Step 1 — Capture Reference</div>
                <div className="ref-images-row" style={{ display: "flex", gap: "12px", marginBottom: "20px", marginTop: "25px", justifyContent: "center" }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} style={{
                      width: 115, height: 115,
                      background: "rgba(0,0,0,0.3)",
                      borderRadius: 16,
                      border: refImages[i] ? "1px solid #10b981" : "1px dashed rgba(255,255,255,0.1)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      overflow: "hidden"
                    }}>
                      {refImages[i] ? (
                        <ImageCanvas base64={refImages[i]} width={96} height={96} scale={115 / 96} />
                      ) : (
                        <span style={{ fontSize: 28, opacity: 0.1, fontWeight: "bold" }}>{i + 1}</span>
                      )}
                    </div>
                  ))}
                </div>
                <div className="ref-counter" style={{ justifyContent: "center", marginBottom: 8 }}>
                  <span className="ref-count-text">{numRefs}/3 references</span>
                </div>
                <button className="btn btn-primary" style={{ width: "100%" }}
                  onClick={handleCapture} disabled={capturing || numRefs >= 3}>
                  {capturing ? "⏳ Capturing..." : "📸 Capture Reference"}
                </button>
              </div>

              {/* Step 2 */}
              <div className="step-section">
                <div className="step-label">Step 2 — Test Object</div>
                <div style={{ marginBottom: 16, display: "flex", justifyContent: "center", minHeight: 80, alignItems: "center" }}>
                  {testImage ? (
                    <div style={{ padding: 4, background: "rgba(0,0,0,0.3)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)" }}>
                      <ImageCanvas base64={testImage} width={96} height={96} scale={1.2} />
                    </div>
                  ) : (
                    <div style={{ width: 96, height: 96, background: "rgba(0,0,0,0.2)", borderRadius: 12, border: "1px dashed rgba(255,255,255,0.1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ fontSize: 32, opacity: 0.1 }}>?</span>
                    </div>
                  )}
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 16, textAlign: "center" }}>
                  Place object in front of camera.
                </p>
                <button className="btn btn-primary" style={{ width: "100%" }}
                  onClick={handleTest} disabled={testing || numRefs === 0}>
                  {testing ? "⏳ Processing..." : "🔍 Test Object"}
                </button>
              </div>
            </div>

            <div style={{ marginTop: 12, display: "flex", justifyContent: "flex-end" }}>
              <button className="btn btn-ghost" onClick={handleReset} disabled={numRefs === 0}>🔄 Reset All</button>
            </div>

            {/* Loading */}
            {testing && (
              <div className="spinner-overlay">
                <div className="spinner" />
                <span className="spinner-text">Running inference on ESP32...</span>
              </div>
            )}

            {/* Result */}
            {testResult && !testing && (
              <div className={`result-panel ${testResult.match ? "match" : "nomatch"}`} style={{ marginTop: 20 }}>
                <div className="result-icon">{testResult.match ? "✅" : "❌"}</div>
                <div className="result-text">{testResult.match ? "MATCH DETECTED" : "NO MATCH"}</div>

                {/* Gauges — bigger, properly spaced */}
                <div className="gauge-container">
                  <SimilarityGauge value={testResult.similarity} label="AVG" colorClass={getSimColor(testResult.similarity)} size={150} />
                  {testResult.max_similarity !== undefined && (
                    <SimilarityGauge value={testResult.max_similarity} label="MAX" colorClass={getSimColor(testResult.max_similarity)} size={150} />
                  )}
                  {testResult.min_similarity !== undefined && (
                    <SimilarityGauge value={testResult.min_similarity} label="MIN" colorClass={getSimColor(testResult.min_similarity)} size={150} />
                  )}
                </div>

                {/* Detail cards */}
                <div className="result-details">
                  <div className="result-detail-item">
                    <div className="result-detail-label">Cosine Similarity</div>
                    <div className="result-detail-value" style={{ color: testResult.match ? "var(--accent-green)" : "var(--accent-red)" }}>
                      {(testResult.similarity * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="result-detail-item">
                    <div className="result-detail-label">Threshold</div>
                    <div className="result-detail-value" style={{ color: "var(--accent-cyan)" }}>
                      {(testResult.localThreshold * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="result-detail-item">
                    <div className="result-detail-label">Distance</div>
                    <div className="result-detail-value" style={{ color: "var(--accent-purple)" }}>
                      {((1 - testResult.similarity) * 100).toFixed(2)}%
                    </div>
                  </div>
                  {testResult.capture_ms !== undefined && (
                    <div className="result-detail-item">
                      <div className="result-detail-label">Capture</div>
                      <div className="result-detail-value" style={{ color: "var(--accent-amber)" }}>{testResult.capture_ms} ms</div>
                    </div>
                  )}
                  {testResult.inference_ms !== undefined && (
                    <div className="result-detail-item">
                      <div className="result-detail-label">Inference</div>
                      <div className="result-detail-value" style={{ color: "var(--accent-cyan)" }}>{testResult.inference_ms} ms</div>
                    </div>
                  )}
                  {testResult.total_ms !== undefined && (
                    <div className="result-detail-item">
                      <div className="result-detail-label">Total Pipeline</div>
                      <div className="result-detail-value" style={{ color: "var(--accent-pink)" }}>{testResult.total_ms} ms</div>
                    </div>
                  )}
                </div>

                {/* Embedding Chart */}
                <div style={{ marginTop: 24, padding: "0 4px" }}>
                  <div className="section-header" style={{ marginBottom: 16 }}>
                    <h2 style={{ fontSize: "16px" }}>🧠 Convolutional Embeddings</h2>
                  </div>
                  <div style={{
                    background: "rgba(0,0,0,0.3)",
                    borderRadius: 16,
                    padding: 16,
                    border: "1px solid rgba(255,255,255,0.05)"
                  }}>
                    <EmbeddingChart refs={refEmbeddings} test={testEmbedding} height={200} />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ============ 2. CHARTS (Inference + Similarity History) ============ */}
          <div className="section-header">
            <h2>📈 Inference History</h2>
            <div className="line" />
          </div>

          <div className="grid-2">
            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon cyan">⚡</div>
                  <div>
                    <div className="card-title">Inference Time</div>
                    <div className="card-subtitle">Per-run latency (ms)</div>
                  </div>
                </div>
                {inferenceHistory.length > 0 ? (
                  <LineChart data={inferenceHistory} label="Inference Latency" color="#00d4ff" unit="ms" />
                ) : (
                  <div className="chart-empty">Run a capture or test to see data</div>
                )}
              </div>
            </GlowCard>

            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon purple">📸</div>
                  <div>
                    <div className="card-title">Capture Time</div>
                    <div className="card-subtitle">Camera capture latency (ms)</div>
                  </div>
                </div>
                {captureHistory.length > 0 ? (
                  <LineChart data={captureHistory} label="Capture Latency" color="#a855f7" unit="ms" />
                ) : (
                  <div className="chart-empty">Run a capture or test to see data</div>
                )}
              </div>
            </GlowCard>
          </div>

          {history.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <GlowCard className="" >
                <div className="card" style={{ marginBottom: 0 }}>
                  <div className="card-header">
                    <div className="card-icon green">🎯</div>
                    <div>
                      <div className="card-title">Similarity Score History</div>
                      <div className="card-subtitle">Detailed metrics per run (Hover for details)</div>
                    </div>
                  </div>
                  <HistoryChart history={history} height={200} />
                </div>
              </GlowCard>
            </div>
          )}

          {/* ============ 3. SYSTEM TELEMETRY ============ */}
          <div className="section-header">
            <h2>📊 System Telemetry</h2>
            <div className="line" />
          </div>

          <div className="grid-4">
            <MetricCard icon="⚡" iconColor="cyan" title="Inference" subtitle="Last run"
              value={metrics?.last_inference_ms ?? "—"} unit="milliseconds" colorClass="cyan"
              barPercent={metrics ? (metrics.last_inference_ms / 2000) * 100 : 0} barColor="cyan" />
            <MetricCard icon="📸" iconColor="purple" title="Capture" subtitle="Last capture"
              value={metrics?.last_capture_ms ?? "—"} unit="milliseconds" colorClass="purple" />
            <MetricCard icon="🧠" iconColor="green" title="PSRAM" subtitle="Memory usage"
              value={metrics ? `${((metrics.psram_total - metrics.psram_free) / 1024 / 1024).toFixed(1)}` : "—"}
              unit={metrics ? `/ ${(metrics.psram_total / 1024 / 1024).toFixed(1)} MB` : "MB"}
              colorClass="green" barPercent={psramUsedPercent} barColor="green" />
            <MetricCard icon="💾" iconColor="amber" title="Heap" subtitle="Free SRAM"
              value={metrics ? `${(metrics.free_heap / 1024).toFixed(0)}` : "—"} unit="KB free" colorClass="amber" />
            <MetricCard icon="📶" iconColor="blue" title="WiFi RSSI" subtitle="Signal strength"
              value={metrics?.wifi_rssi ?? "—"} unit="dBm" colorClass="blue"
              barPercent={metrics ? Math.min(100, ((metrics.wifi_rssi + 90) / 60) * 100) : 0} barColor="cyan" />
            <MetricCard icon="🔄" iconColor="pink" title="Inferences" subtitle="Total count"
              value={metrics?.total_inferences ?? "0"} unit="total runs" colorClass="pink" />
            <MetricCard icon="⏱️" iconColor="cyan" title="Uptime" subtitle="Since boot"
              value={formatUptime(metrics?.uptime_sec)} unit="" colorClass="cyan" />
            <MetricCard icon="🖥️" iconColor="purple" title="CPU" subtitle="Clock speed"
              value={metrics?.cpu_freq_mhz ?? "—"} unit="MHz" colorClass="purple" />
          </div>

          {/* ============ 4. POWER & MODEL ============ */}
          <div className="grid-2">
            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon amber">🔋</div>
                  <div>
                    <div className="card-title">Power Consumption</div>
                    <div className="card-subtitle">Estimated based on ESP32-S3 specs</div>
                  </div>
                </div>
                <table className="power-table">
                  <thead><tr><th>Component</th><th>State</th><th>Power</th></tr></thead>
                  <tbody>
                    <tr><td>CPU (Dual-core)</td><td>240 MHz Active</td><td>~95 mA</td></tr>
                    <tr><td>WiFi</td><td>Station Mode</td><td>~120 mA</td></tr>
                    <tr><td>PSRAM</td><td>Active</td><td>~15 mA</td></tr>
                    <tr><td>Camera (OV2640)</td><td>Capturing</td><td>~40 mA</td></tr>
                    <tr><td>SPI Bus</td><td>8 MHz</td><td>~5 mA</td></tr>
                    <tr><td style={{ fontWeight: 700, color: "var(--accent-amber)" }}>Total</td><td></td>
                      <td style={{ fontWeight: 700, color: "var(--accent-amber)" }}>~275 mA @ 3.3V</td></tr>
                    <tr><td style={{ color: "var(--text-muted)" }}>Power Draw</td><td></td>
                      <td style={{ color: "var(--text-muted)" }}>~0.91 W</td></tr>
                  </tbody>
                </table>
              </div>
            </GlowCard>

            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon blue">🏗️</div>
                  <div>
                    <div className="card-title">Model Architecture</div>
                    <div className="card-subtitle">Deployed model specs</div>
                  </div>
                </div>
                <div className="specs-grid">
                  <div className="spec-item"><div className="spec-label">Architecture</div><div className="spec-value">MobileNetV2</div></div>
                  <div className="spec-item"><div className="spec-label">Width Multiplier</div><div className="spec-value">α = 0.35</div></div>
                  <div className="spec-item"><div className="spec-label">Quantization</div><div className="spec-value">INT8</div></div>
                  <div className="spec-item"><div className="spec-label">Input Size</div><div className="spec-value">96 × 96 × 3</div></div>
                  <div className="spec-item"><div className="spec-label">Embedding Dim</div><div className="spec-value">128-D L2-Norm</div></div>
                  <div className="spec-item"><div className="spec-label">Model Size</div><div className="spec-value">{metrics?.model_size_kb ? `${metrics.model_size_kb} KB` : "~771 KB"}</div></div>
                  <div className="spec-item"><div className="spec-label">Tensor Arena</div><div className="spec-value">{metrics?.arena_size_kb ? `${metrics.arena_size_kb} KB` : "2048 KB"}</div></div>
                  <div className="spec-item"><div className="spec-label">Camera</div><div className="spec-value">OV2640 160×120</div></div>
                </div>
              </div>
            </GlowCard>
          </div>

          {/* ============ 5. THROUGHPUT ============ */}
          <div className="section-header">
            <h2>🚀 Performance & Throughput</h2>
            <div className="line" />
          </div>

          <div className="grid-3">
            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon green">📈</div>
                  <div><div className="card-title">Throughput</div><div className="card-subtitle">Inference rate</div></div>
                </div>
                <div className="metric-value green">
                  {metrics?.last_inference_ms ? (1000 / (metrics.last_inference_ms + (metrics.last_capture_ms || 0))).toFixed(2) : "—"}
                </div>
                <div className="metric-unit">inferences / second</div>
              </div>
            </GlowCard>
            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon cyan">⚙️</div>
                  <div><div className="card-title">Pipeline Latency</div><div className="card-subtitle">Capture + Inference</div></div>
                </div>
                <div className="metric-value cyan">
                  {metrics ? (metrics.last_capture_ms || 0) + (metrics.last_inference_ms || 0) : "—"}
                </div>
                <div className="metric-unit">ms total pipeline</div>
              </div>
            </GlowCard>
            <GlowCard>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon amber">⚡</div>
                  <div><div className="card-title">Energy / Inference</div><div className="card-subtitle">Estimated</div></div>
                </div>
                <div className="metric-value amber">
                  {metrics?.last_inference_ms ? (0.275 * 3.3 * (metrics.last_inference_ms / 1000)).toFixed(3) : "—"}
                </div>
                <div className="metric-unit">mJ (millijoules)</div>
              </div>
            </GlowCard>
          </div>
        </>
      )}

      {/* NOT CONNECTED */}
      {!connected && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text-muted)" }}>
          <div style={{ fontSize: 64, marginBottom: 16 }}>🔌</div>
          <p style={{ fontSize: 18, fontWeight: 500, color: "var(--text-secondary)" }}>Enter your ESP32-S3 IP address above to connect</p>
          <p style={{ fontSize: 14, marginTop: 8 }}>Make sure the ESP32 is powered on and connected to the same WiFi network</p>
        </div>
      )}

      {/* FOOTER */}
      <footer className="footer">
        <p>ESP32-S3 Object Recognition System • MobileNetV2 INT8 • B.Tech Thesis Project</p>
        <p style={{ marginTop: 4 }}>Built with Next.js • Powered by TensorFlow Lite Micro</p>
      </footer>

      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </main>
  );
}
