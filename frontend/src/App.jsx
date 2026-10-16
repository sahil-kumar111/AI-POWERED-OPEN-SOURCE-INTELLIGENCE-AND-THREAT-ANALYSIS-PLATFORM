import React, { useState, useEffect } from "react";
import {
  Shield,
  AlertTriangle,
  Users,
  Flame,
  Crosshair,
  Activity,
  FileText,
  MapPin,
  Video,
  Play,
  Download,
  RefreshCw,
  Layers,
  Compass,
  CheckCircle,
  Clock,
  Check,
} from "lucide-react";
import "./App.css";

const API_BASE = "/api";

export default function App() {
  const [location, setLocation] = useState("Connaught Place, New Delhi");
  const [selectedFile, setSelectedFile] = useState(null);
  const [useSample, setUseSample] = useState(true);
  const [venueCapacity, setVenueCapacity] = useState(15);
  const [frameSkip, setFrameSkip] = useState(2);

  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [jobId, setJobId] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [healthStatus, setHealthStatus] = useState("checking");

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => {
        setHealthStatus(data.status === "healthy" ? "online" : "degraded");
      })
      .catch(() => {
        setHealthStatus("offline");
      });
  }, []);

  // Poll job status if analyzing
  useEffect(() => {
    if (!analyzing || !jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        setProgress(job.progress || 0);
        setCurrentStage(job.stage || "Processing...");

        if (job.status === "completed") {
          clearInterval(interval);
          // Fetch final results
          const resultRes = await fetch(`${API_BASE}/jobs/${jobId}/results`);
          if (resultRes.ok) {
            const data = await resultRes.json();
            setResults(data);
          }
          setAnalyzing(false);
        } else if (job.status === "failed") {
          clearInterval(interval);
          setAnalyzing(false);
          setErrorMsg(job.error || "Analysis pipeline encountered an error.");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [analyzing, jobId]);

  // Handle Form Submission
  const handleStartAnalysis = async () => {
    setErrorMsg(null);
    if (!location.trim()) {
      setErrorMsg("Please enter a surveillance location.");
      return;
    }
    if (!useSample && !selectedFile) {
      setErrorMsg("Please select a video file or enable the bundled scenario video.");
      return;
    }

    setAnalyzing(true);
    setProgress(5);
    setCurrentStage("Submitting intelligence analysis request...");

    try {
      let endpoint = `${API_BASE}/analyze`;
      let formData = new FormData();
      formData.append("location", location.trim());
      formData.append("venue_capacity", venueCapacity);
      formData.append("frame_skip", frameSkip);

      if (useSample) {
        endpoint = `${API_BASE}/sample-analyze`;
      } else if (selectedFile) {
        // Validate extension
        const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf(".")).toLowerCase();
        if (![".mp4", ".avi", ".mov", ".mkv", ".webm"].includes(ext)) {
          throw new Error(`Unsupported video format (${ext}). Supported: MP4, AVI, MOV, MKV.`);
        }
        formData.append("video", selectedFile);
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${response.status}`);
      }

      const resData = await response.json();
      setJobId(resData.job_id);
      setProgress(10);
      setCurrentStage("Analysis task registered. Running pipeline...");
    } catch (err) {
      setAnalyzing(false);
      setErrorMsg(err.message || "Failed to initiate analysis.");
    }
  };

  const overview = results?.event_overview || {};
  const ati = overview.aegis_threat_index ?? 0;
  const level = overview.threat_level || "LOW";
  const dominant = results?.threat_detection?.dominant_threat_category || "none";
  const pAnalysis = results?.person_and_crowd_analysis || {};
  const rankedThreats = results?.threat_detection?.ranked_threat_categories || [];
  const viz = results?.heatmaps_and_visualizations || {};
  const fusion = results?.evidence_fusion || {};
  const report = results?.final_intelligence_report || {};
  const sevenQ = report.seven_questions || {};

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <Shield size={32} color="#38bdf8" />
          <div>
            <div className="brand-title">AEGIS THREAT INTELLIGENCE</div>
            <div className="brand-subtitle">Multimodal Defense & Real-Time Open-Source Threat Analysis</div>
          </div>
        </div>

        <div className="header-meta">
          <div className="status-pill">
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: healthStatus === "online" ? "#22c55e" : healthStatus === "degraded" ? "#eab308" : "#ef4444",
                display: "inline-block",
              }}
            />
            BACKEND: {healthStatus.toUpperCase()}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Input Panel */}
        <div className="input-panel">
          <div className="panel-title">
            <Compass size={20} color="#38bdf8" />
            Surveillance Feeds & Target Zone
          </div>

          <div className="input-grid">
            <div className="form-group">
              <label className="form-label">Event Location / Search Query</label>
              <input
                type="text"
                className="form-input"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. Connaught Place, New Delhi"
                disabled={analyzing}
              />
              <div className="preset-pills">
                {["Connaught Place, New Delhi", "Colaba, Mumbai", "MG Road, Bengaluru"].map((p) => (
                  <button
                    key={p}
                    type="button"
                    className="preset-btn"
                    onClick={() => setLocation(p)}
                    disabled={analyzing}
                  >
                    {p.split(",")[0]}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Video Stream Source</label>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <button
                  type="button"
                  className="preset-btn"
                  style={{
                    flex: 1,
                    padding: "0.55rem",
                    background: useSample ? "rgba(56, 189, 248, 0.25)" : "transparent",
                    borderColor: useSample ? "#38bdf8" : "var(--border-color)",
                  }}
                  onClick={() => setUseSample(true)}
                  disabled={analyzing}
                >
                  📁 Verified CCTV Sample (Armed Incident)
                </button>
                <button
                  type="button"
                  className="preset-btn"
                  style={{
                    flex: 1,
                    padding: "0.55rem",
                    background: !useSample ? "rgba(56, 189, 248, 0.25)" : "transparent",
                    borderColor: !useSample ? "#38bdf8" : "var(--border-color)",
                  }}
                  onClick={() => setUseSample(false)}
                  disabled={analyzing}
                >
                  📤 Upload Video
                </button>
              </div>
              {!useSample && (
                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv"
                  onChange={(e) => setSelectedFile(e.target.files[0])}
                  disabled={analyzing}
                  style={{ marginTop: "0.4rem", fontSize: "0.8rem" }}
                />
              )}
            </div>

            <div className="form-group">
              <label className="form-label">Venue Crowd Capacity</label>
              <input
                type="number"
                min="1"
                max="500"
                className="form-input"
                value={venueCapacity}
                onChange={(e) => setVenueCapacity(parseInt(e.target.value) || 15)}
                disabled={analyzing}
              />
            </div>

            <button
              type="button"
              className="analyze-button"
              onClick={handleStartAnalysis}
              disabled={analyzing}
            >
              {analyzing ? (
                <>
                  <RefreshCw size={18} className="spin" />
                  ANALYZING ({progress}%)
                </>
              ) : (
                <>
                  <AlertTriangle size={18} />
                  ANALYZE VIDEO & THREATS
                </>
              )}
            </button>
          </div>

          {errorMsg && (
            <div
              style={{
                marginTop: "1rem",
                padding: "0.75rem 1rem",
                borderRadius: "6px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid #ef4444",
                color: "#ef4444",
                fontSize: "0.85rem",
              }}
            >
              ⚠️ {errorMsg}
            </div>
          )}
        </div>

        {/* Live Progress Bar when Analyzing */}
        {analyzing && (
          <div className="progress-panel">
            <div className="progress-header">
              <span>{currentStage || "Executing Multimodal AI Pipeline..."}</span>
              <span style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>{progress}%</span>
            </div>
            <div className="progress-bar-bg">
              <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {/* Dashboard Results Section */}
        {results && (
          <div>
            {/* Metric Ribbon */}
            <div className="metric-ribbon">
              <div className="metric-card">
                <span className="metric-card-title">AEGIS Threat Index (ATI)</span>
                <span className={`metric-card-value badge-${level}`}>{ati} / 100</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Official Status: <strong>{level}</strong>
                </span>
              </div>

              <div className="metric-card">
                <span className="metric-card-title">Dominant Threat</span>
                <span className="metric-card-value" style={{ color: "var(--accent-orange)" }}>
                  {dominant.toUpperCase()}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Classification: {overview.threat_classification}
                </span>
              </div>

              <div className="metric-card">
                <span className="metric-card-title">Unique Tracked People</span>
                <span className="metric-card-value" style={{ color: "var(--accent-cyan)" }}>
                  {pAnalysis.unique_people_count ?? 0}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  ByteTrack Association Active
                </span>
              </div>

              <div className="metric-card">
                <span className="metric-card-title">Total Detections</span>
                <span className="metric-card-value">
                  {pAnalysis.total_person_detections ?? 0}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Across {viz.frame_timeline?.length ?? 90} Analyzed Frames
                </span>
              </div>

              <div className="metric-card">
                <span className="metric-card-title">Execution Duration</span>
                <span className="metric-card-value" style={{ color: "var(--accent-green)" }}>
                  {results.execution_time_seconds}s
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Multimodal Pipeline
                </span>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="tabs-nav">
              {[
                { id: "overview", label: "Event Overview", icon: Activity },
                { id: "people", label: "Person & Crowd Tracking", icon: Users },
                { id: "threats", label: "Threat Detection", icon: Crosshair },
                { id: "visuals", label: "Heatmaps & Visualizations", icon: Layers },
                { id: "fusion", label: "Evidence Fusion & XAI", icon: Shield },
                { id: "report", label: "Intelligence Report", icon: FileText },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    <Icon size={16} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* TAB PANELS */}
            {activeTab === "overview" && (
              <div className="tab-panel">
                <h3 style={{ marginBottom: "1rem" }}>Surveillance Event Summary</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                  <div>
                    <p style={{ margin: "0.5rem 0" }}>
                      <strong>Video Name:</strong> {overview.video_name}
                    </p>
                    <p style={{ margin: "0.5rem 0" }}>
                      <strong>Target Location:</strong> {overview.location}
                    </p>
                    <p style={{ margin: "0.5rem 0" }}>
                      <strong>Coordinates:</strong> Latitude {overview.coordinates?.latitude}, Longitude {overview.coordinates?.longitude}
                    </p>
                    <p style={{ margin: "0.5rem 0" }}>
                      <strong>Threat Detected:</strong>{" "}
                      <span className={overview.threat_detected ? "badge-HIGH" : "badge-LOW"}>
                        {overview.threat_detected ? "YES - THREAT IDENTIFIED" : "NO THREATS DETECTED"}
                      </span>
                    </p>
                    <p style={{ margin: "0.5rem 0" }}>
                      <strong>Official Threat Level:</strong>{" "}
                      <span className={`badge-tag-${level}`} style={{ padding: "2px 8px", borderRadius: "4px" }}>
                        {level}
                      </span>
                    </p>
                  </div>

                  <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "6px" }}>
                    <h4 style={{ color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>ATI Threat Index Calculation Rationale</h4>
                    <p style={{ fontSize: "0.85rem", lineHeight: 1.6 }}>{overview.calculation_explanation}</p>
                    <div style={{ marginTop: "1rem", fontSize: "0.75rem", color: "var(--text-dim)" }}>
                      ⚠️ <em>Analytical Estimate Notice:</em> The AEGIS Threat Index (ATI) represents a deterministic multimodal heuristic assessment. Model confidence is not a calibrated real-world probability.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "people" && (
              <div className="tab-panel">
                <h3 style={{ marginBottom: "1rem" }}>Person Detection & Crowd Density Analysis</h3>
                <div
                  style={{
                    background: "rgba(56, 189, 248, 0.1)",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    padding: "1rem",
                    borderRadius: "6px",
                    marginBottom: "1.5rem",
                  }}
                >
                  <p style={{ fontSize: "0.9rem", color: "var(--accent-cyan)" }}>
                    💡 <strong>Identity Association & Tracking:</strong> {pAnalysis.detections_vs_unique_explanation}
                  </p>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
                  <div className="metric-card">
                    <span className="metric-card-title">Unique Tracked People</span>
                    <span className="metric-card-value">{pAnalysis.unique_people_count ?? 0}</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-card-title">Total Person Detections</span>
                    <span className="metric-card-value">{pAnalysis.total_person_detections ?? 0}</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-card-title">Peak Concurrent Persons</span>
                    <span className="metric-card-value">{pAnalysis.peak_concurrent_persons ?? 0}</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-card-title">Crowd Capacity Limit</span>
                    <span className="metric-card-value">{pAnalysis.venue_capacity ?? 15}</span>
                  </div>
                </div>

                {/* Person Count Over Time SVG Chart */}
                <h4>Frame-by-Frame Person Density vs Venue Capacity</h4>
                <div style={{ background: "var(--bg-secondary)", padding: "1.5rem", borderRadius: "8px", marginTop: "0.75rem" }}>
                  {viz.frame_timeline && viz.frame_timeline.length > 0 ? (
                    <svg viewBox="0 0 800 200" style={{ width: "100%", height: "200px" }}>
                      <line x1="40" y1="170" x2="780" y2="170" stroke="#334155" strokeWidth="1" />
                      <line x1="40" y1="30" x2="40" y2="170" stroke="#334155" strokeWidth="1" />

                      {/* Threshold Line */}
                      <line
                        x1="40"
                        y1={170 - (Math.min(pAnalysis.venue_capacity || 15, 20) / 20) * 130}
                        x2="780"
                        y2={170 - (Math.min(pAnalysis.venue_capacity || 15, 20) / 20) * 130}
                        stroke="#ef4444"
                        strokeDasharray="4"
                        strokeWidth="1.5"
                      />
                      <text
                        x="650"
                        y={165 - (Math.min(pAnalysis.venue_capacity || 15, 20) / 20) * 130}
                        fill="#ef4444"
                        fontSize="10"
                      >
                        Capacity Limit ({pAnalysis.venue_capacity})
                      </text>

                      {/* Data Line */}
                      <polyline
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="2.5"
                        points={viz.frame_timeline
                          .map((item, idx) => {
                            const x = 40 + (idx / Math.max(viz.frame_timeline.length - 1, 1)) * 740;
                            const y = 170 - (Math.min(item.person_count, 20) / 20) * 130;
                            return `${x},${y}`;
                          })
                          .join(" ")}
                      />
                      <text x="40" y="190" fill="#94a3b8" fontSize="10">
                        0.0s
                      </text>
                      <text x="750" y="190" fill="#94a3b8" fontSize="10">
                        {viz.frame_timeline[viz.frame_timeline.length - 1]?.timestamp_sec}s
                      </text>
                    </svg>
                  ) : (
                    <p style={{ color: "var(--text-muted)" }}>No frame timeline data recorded.</p>
                  )}
                </div>
              </div>
            )}

            {activeTab === "threats" && (
              <div className="tab-panel">
                <h3 style={{ marginBottom: "1rem" }}>Supported Threat Categories Breakdown</h3>
                <div className="threat-grid">
                  {rankedThreats.map((threat) => (
                    <div
                      key={threat.category}
                      className={`threat-card ${threat.status === "DETECTED" ? "detected" : ""}`}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontWeight: 700, fontSize: "1rem" }}>{threat.label}</span>
                        <span className={`badge-tag-${threat.severity}`} style={{ padding: "2px 8px", borderRadius: "4px", fontSize: "0.75rem" }}>
                          {threat.status}
                        </span>
                      </div>
                      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                        Confidence: <strong>{(threat.confidence * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                        Detections: <strong>{threat.detection_count}</strong> ({threat.frame_percentage}% of frames)
                      </div>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-dim)", marginTop: "0.4rem" }}>
                        {threat.explanation}
                      </p>
                    </div>
                  ))}
                </div>

                <h4>Ranked Threat Categories by Severity & Evidence</h4>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Category</th>
                      <th>Status</th>
                      <th>Max Confidence</th>
                      <th>Detections</th>
                      <th>Frame %</th>
                      <th>Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rankedThreats.map((t) => (
                      <tr key={t.rank}>
                        <td>#{t.rank}</td>
                        <td style={{ fontWeight: 600 }}>{t.label}</td>
                        <td>
                          <span className={t.status === "DETECTED" ? "badge-HIGH" : "badge-LOW"}>
                            {t.status}
                          </span>
                        </td>
                        <td>{(t.confidence * 100).toFixed(1)}%</td>
                        <td>{t.detection_count}</td>
                        <td>{t.frame_percentage}%</td>
                        <td>
                          <span className={`badge-${t.severity}`}>{t.severity}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === "visuals" && (
              <div className="tab-panel">
                <h3 style={{ marginBottom: "1.25rem" }}>Data-Backed Heatmaps & Visual Evidence</h3>
                <div className="viz-row">
                  <div className="viz-box">
                    <h4>🗺️ Spatial Threat Activation Map (EigenCAM)</h4>
                    <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                      Class-agnostic feature activation overlay on highest-confidence threat frame.
                    </p>
                    {jobId ? (
                      <img
                        src={`${API_BASE}/jobs/${jobId}/artifacts/threat_highlight.jpg`}
                        alt="EigenCAM Heatmap"
                        className="viz-img"
                        onError={(e) => {
                          e.target.src = `${API_BASE}/jobs/fallback/artifacts/threat_highlight.jpg`;
                        }}
                      />
                    ) : (
                      <img src="/output/threat_highlight.jpg" alt="EigenCAM Heatmap" className="viz-img" />
                    )}
                  </div>

                  <div className="viz-box">
                    <h4>🎥 Annotated Surveillance Video Preview</h4>
                    <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                      YOLOv11 detections with bounding boxes, confidence, and ByteTrack track IDs.
                    </p>
                    {viz.annotated_video_artifact && (
                      <video
                        controls
                        style={{ width: "100%", borderRadius: "6px", marginTop: "0.5rem" }}
                        src={`${API_BASE}/jobs/${jobId || "default"}/artifacts/${viz.annotated_video_artifact}`}
                      />
                    )}
                  </div>
                </div>

                <h4>⏱️ Frame-wise Threat Activity Over Time</h4>
                <div style={{ background: "var(--bg-secondary)", padding: "1.5rem", borderRadius: "8px", marginTop: "0.75rem" }}>
                  {viz.frame_timeline && viz.frame_timeline.length > 0 ? (
                    <svg viewBox="0 0 800 160" style={{ width: "100%", height: "160px" }}>
                      <line x1="40" y1="130" x2="780" y2="130" stroke="#334155" strokeWidth="1" />
                      <line x1="40" y1="20" x2="40" y2="130" stroke="#334155" strokeWidth="1" />

                      {/* Threat activity polyline */}
                      <polyline
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="2"
                        points={viz.frame_timeline
                          .map((item, idx) => {
                            const x = 40 + (idx / Math.max(viz.frame_timeline.length - 1, 1)) * 740;
                            const tVal = (item.threat_counts?.weapon || 0) + (item.threat_counts?.fire || 0);
                            const y = 130 - Math.min(tVal, 5) * 20;
                            return `${x},${y}`;
                          })
                          .join(" ")}
                      />
                      <text x="40" y="150" fill="#94a3b8" fontSize="10">
                        0.0s
                      </text>
                      <text x="750" y="150" fill="#94a3b8" fontSize="10">
                        {viz.frame_timeline[viz.frame_timeline.length - 1]?.timestamp_sec}s
                      </text>
                    </svg>
                  ) : (
                    <p style={{ color: "var(--text-muted)" }}>No threat activity timeline recorded.</p>
                  )}
                </div>
              </div>
            )}

            {activeTab === "fusion" && (
              <div className="tab-panel">
                <h3 style={{ marginBottom: "1rem" }}>Multimodal Evidence Fusion & SHAP Attribution</h3>
                <h4>Feature Attribution Breakdown (Percentage Contribution to Score)</h4>
                <div style={{ margin: "1rem 0" }}>
                  {fusion.feature_attribution &&
                    fusion.feature_attribution.map((fa, i) => (
                      <div key={i} style={{ marginBottom: "0.75rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "0.2rem" }}>
                          <span>
                            <strong>{fa.factor.toUpperCase()}</strong> ({fa.modality})
                          </span>
                          <span style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>{fa.percentage_importance}%</span>
                        </div>
                        <div style={{ background: "var(--bg-secondary)", borderRadius: "4px", height: "8px", overflow: "hidden" }}>
                          <div
                            style={{
                              background: "linear-gradient(90deg, #38bdf8, #818cf8)",
                              height: "100%",
                              width: `${fa.percentage_importance}%`,
                            }}
                          />
                        </div>
                        <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                          {fa.evidence_reason}
                        </p>
                      </div>
                    ))}
                </div>

                <h4 style={{ marginTop: "1.5rem" }}>Tactical Reasoning</h4>
                <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "6px", fontSize: "0.85rem", lineHeight: 1.6 }}>
                  {report.tactical_reasoning}
                </div>

                <h4 style={{ marginTop: "1.5rem" }}>Actionable Operational Directives</h4>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem", marginTop: "0.75rem" }}>
                  <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "6px" }}>
                    <h5 style={{ color: "var(--accent-red)", marginBottom: "0.5rem" }}>⏱️ 0 - 5 Min (Immediate)</h5>
                    <ul style={{ paddingLeft: "1.2rem", fontSize: "0.8rem", lineHeight: 1.5 }}>
                      {report.recommendations?.immediate_directives_0_to_5_min?.map((act, i) => (
                        <li key={i}>{act}</li>
                      ))}
                    </ul>
                  </div>

                  <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "6px" }}>
                    <h5 style={{ color: "var(--accent-orange)", marginBottom: "0.5rem" }}>⏱️ 5 - 30 Min (Containment)</h5>
                    <ul style={{ paddingLeft: "1.2rem", fontSize: "0.8rem", lineHeight: 1.5 }}>
                      {report.recommendations?.area_management_5_to_30_min?.map((act, i) => (
                        <li key={i}>{act}</li>
                      ))}
                    </ul>
                  </div>

                  <div style={{ background: "var(--bg-secondary)", padding: "1rem", borderRadius: "6px" }}>
                    <h5 style={{ color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>🛡️ Investigation</h5>
                    <ul style={{ paddingLeft: "1.2rem", fontSize: "0.8rem", lineHeight: 1.5 }}>
                      {report.recommendations?.intelligence_and_investigation?.map((act, i) => (
                        <li key={i}>{act}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "report" && (
              <div className="tab-panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                  <h3>Final Executive Intelligence Briefing</h3>
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <a
                      href={`${API_BASE}/jobs/${jobId || "default"}/artifacts/daily_intelligence_report.md`}
                      download="daily_intelligence_report.md"
                      className="preset-btn"
                      style={{ padding: "0.5rem 0.8rem", textDecoration: "none" }}
                    >
                      <Download size={14} style={{ marginRight: 4 }} /> Download Report (.md)
                    </a>
                    <a
                      href={`${API_BASE}/jobs/${jobId || "default"}/artifacts/gis_layer.geojson`}
                      download="gis_layer.geojson"
                      className="preset-btn"
                      style={{ padding: "0.5rem 0.8rem", textDecoration: "none" }}
                    >
                      <Download size={14} style={{ marginRight: 4 }} /> Download GIS (.geojson)
                    </a>
                  </div>
                </div>

                <div className="questions-list">
                  {[
                    { q: "1. What is happening in the video?", a: sevenQ.q1_what_is_happening },
                    { q: "2. How many people are visible or reliably tracked?", a: sevenQ.q2_how_many_people },
                    { q: "3. Which threats were detected?", a: sevenQ.q3_which_threats_detected },
                    { q: "4. Which threat category is dominant?", a: sevenQ.q4_dominant_threat },
                    { q: "5. What is the overall threat severity and why?", a: sevenQ.q5_overall_threat_severity_and_why },
                    { q: "6. What evidence supports the conclusion?", a: sevenQ.q6_evidence_supporting_conclusion },
                    { q: "7. What additional information or limitations affect the result?", a: sevenQ.q7_limitations_and_additional_info },
                  ].map((item, idx) => (
                    <div key={idx} className="question-card">
                      <div className="question-title">{item.q}</div>
                      <div className="question-answer">{item.a}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        AEGIS Open-Source Intelligence & Threat Analysis Platform • Video-First Multi-Modal Architecture • Version 2.0.0
      </footer>
    </div>
  );
}
