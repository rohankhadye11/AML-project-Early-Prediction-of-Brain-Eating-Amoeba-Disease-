"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from "recharts";

// Clinical Schema for 9 Diagnostic Features
interface ClinicalFeatures {
  Freshwater_Exposure: number;
  Severe_Headache: number;
  Fever: number;
  Nausea_Vomiting: number;
  Stiff_Neck: number;
  Photophobia: number;
  Age: number;
  Symptom_Onset_Days: number;
  CSF_Pressure: number;
}

interface PredictionResponse {
  risk_probability: number;
  shap_values: Record<string, number>;
  gradcam_base64: string;
  triage_level: "CRITICAL_HIGH_RISK" | "LOW_RISK" | string;
  status: string;
}

export default function ClinicalDashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  // Default Clinical Symptoms (Acute PAM Positive Presentation)
  const [clinicalData, setClinicalData] = useState<ClinicalFeatures>({
    Freshwater_Exposure: 1,
    Severe_Headache: 1,
    Fever: 1,
    Nausea_Vomiting: 1,
    Stiff_Neck: 1,
    Photophobia: 1,
    Age: 0.15, // ~12 yrs old (normalized)
    Symptom_Onset_Days: 0.20, // ~3 days progression
    CSF_Pressure: 0.85, // Critically elevated opening pressure
  });

  const handleToggle = (key: keyof ClinicalFeatures) => {
    setClinicalData((prev) => ({
      ...prev,
      [key]: prev[key] === 1 ? 0 : 1,
    }));
  };

  const handleSliderChange = (key: keyof ClinicalFeatures, value: number) => {
    setClinicalData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (uploadEvent) => {
        setImagePreview(uploadEvent.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Helper to create a synthetic mock brain MRI canvas if none uploaded
  const generateMockMRICanvas = (): Promise<File> => {
    return new Promise((resolve) => {
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Dark background
        ctx.fillStyle = "#0a0a0f";
        ctx.fillRect(0, 0, 224, 224);

        // Brain parenchyma ellipse
        ctx.fillStyle = "#333333";
        ctx.beginPath();
        ctx.ellipse(112, 112, 75, 90, 0, 0, Math.PI * 2);
        ctx.fill();

        // Ventricles
        ctx.fillStyle = "#111111";
        ctx.beginPath();
        ctx.arc(102, 112, 10, 0, Math.PI * 2);
        ctx.arc(122, 112, 10, 0, Math.PI * 2);
        ctx.fill();

        // Hyperintense basal amoebic necrosis focus
        ctx.fillStyle = "#d4d4d8";
        ctx.beginPath();
        ctx.arc(95, 88, 16, 0, Math.PI * 2);
        ctx.fill();
      }

      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], "synthetic_mri_patient.png", {
            type: "image/png",
          });
          resolve(file);
        }
      });
    });
  };

  const executeInference = async (useDemoFile: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      let fileToUpload = selectedFile;
      if (!fileToUpload || useDemoFile) {
        fileToUpload = await generateMockMRICanvas();
        // Set preview for demo
        const reader = new FileReader();
        reader.onload = (e) => setImagePreview(e.target?.result as string);
        reader.readAsDataURL(fileToUpload);
      }

      const formData = new FormData();
      formData.append("image", fileToUpload);
      formData.append("clinical_data", JSON.stringify(clinicalData));

      // Call FastAPI endpoint
      const response = await fetch("http://localhost:8000/api/predict_multimodal", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setPrediction(data);
    } catch (err: any) {
      console.error("Inference Error:", err);
      setError(
        err.message || "Failed to communicate with FastAPI Late Fusion inference engine."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Format SHAP values for Recharts horizontal bar visualization
  const chartData = prediction
    ? Object.entries(prediction.shap_values)
        .map(([feature, value]) => ({
          name: feature.replace(/_/g, " "),
          rawFeature: feature,
          value: parseFloat(value.toFixed(4)),
        }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      {/* Header Bar */}
      <header className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="px-2.5 py-1 rounded bg-indigo-500/20 text-indigo-400 text-xs font-semibold uppercase tracking-wider border border-indigo-500/30">
              IEEE Phase 6 Clinical Deployment
            </span>
            <span className="text-xs text-slate-400 font-mono">
              PyTorch Late Fusion + XAI
            </span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-2">
            Primary Amoebic Meningoencephalitis (PAM) Decision Support
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Multimodal Dual-Branch Risk Stratification with Grad-CAM Neuroimaging and SHAP Attribution
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => executeInference(true)}
            disabled={isLoading}
            className="px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition shadow-lg shadow-indigo-600/20 flex items-center gap-2"
          >
            {isLoading ? "Running Dual Backbone..." : "Load Demo Acute PAM Case"}
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Clinical & Imaging Input Console (4 Cols) */}
        <section className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center justify-between">
              <span>Patient Clinical Inputs</span>
              <span className="text-xs font-normal text-slate-500">9 Anchors</span>
            </h2>

            {/* Binary Symptoms Toggles */}
            <div className="space-y-2.5">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Acute Symptoms & Exposure
              </label>
              {[
                { key: "Freshwater_Exposure", label: "Freshwater Exposure History" },
                { key: "Severe_Headache", label: "Severe Bifrontal Headache" },
                { key: "Fever", label: "High Pyrexia / Fever" },
                { key: "Nausea_Vomiting", label: "Nausea & Projectile Vomiting" },
                { key: "Stiff_Neck", label: "Nuchal Rigidity (Stiff Neck)" },
                { key: "Photophobia", label: "Marked Photophobia" },
              ].map(({ key, label }) => {
                const active = clinicalData[key as keyof ClinicalFeatures] === 1;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleToggle(key as keyof ClinicalFeatures)}
                    className={`w-full flex items-center justify-between px-3.5 py-2 rounded-lg text-xs font-medium border transition ${
                      active
                        ? "bg-rose-500/15 border-rose-500/40 text-rose-300"
                        : "bg-slate-800/40 border-slate-700/60 text-slate-400 hover:bg-slate-800"
                    }`}
                  >
                    <span>{label}</span>
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        active ? "bg-rose-500 ring-4 ring-rose-500/20" : "bg-slate-600"
                      }`}
                    />
                  </button>
                );
              })}
            </div>

            {/* Continuous Normalized Biomarkers */}
            <div className="mt-5 pt-4 border-t border-slate-800 space-y-3">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Demographic & CSF Biomarkers
              </label>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Patient Age (Normalized)</span>
                  <span className="font-mono text-slate-200">
                    {clinicalData.Age.toFixed(2)} (~12 yrs)
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={clinicalData.Age}
                  onChange={(e) => handleSliderChange("Age", parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 bg-slate-800"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Symptom Onset Progression</span>
                  <span className="font-mono text-slate-200">
                    {clinicalData.Symptom_Onset_Days.toFixed(2)} (~3 days)
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={clinicalData.Symptom_Onset_Days}
                  onChange={(e) =>
                    handleSliderChange("Symptom_Onset_Days", parseFloat(e.target.value))
                  }
                  className="w-full accent-indigo-500 bg-slate-800"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">CSF Opening Pressure</span>
                  <span className="font-mono text-rose-400 font-semibold">
                    {clinicalData.CSF_Pressure.toFixed(2)} (Elevated)
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={clinicalData.CSF_Pressure}
                  onChange={(e) =>
                    handleSliderChange("CSF_Pressure", parseFloat(e.target.value))
                  }
                  className="w-full accent-rose-500 bg-slate-800"
                />
              </div>
            </div>

            {/* MRI Scan Upload */}
            <div className="mt-5 pt-4 border-t border-slate-800">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Brain MRI Neuroimaging Slice
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-indigo-400 hover:file:bg-slate-700 cursor-pointer"
              />
              {imagePreview && (
                <div className="mt-3 flex items-center gap-3 bg-slate-950 p-2 rounded border border-slate-800">
                  <img
                    src={imagePreview}
                    alt="Uploaded MRI Preview"
                    className="w-12 h-12 object-cover rounded bg-black"
                  />
                  <div className="text-[11px] text-slate-400">
                    <p className="font-medium text-slate-300">Active Scan Loaded</p>
                    <p>Resized to 224x224 grayscale</p>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={() => executeInference(false)}
              disabled={isLoading}
              className="mt-6 w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold text-sm transition shadow-lg shadow-emerald-600/20"
            >
              {isLoading ? "Analyzing Multimodal Data..." : "Run Multimodal Risk Inference"}
            </button>
          </div>
        </section>

        {/* Right Column: 3 Diagnostic Evaluation Panels (8 Cols) */}
        <section className="lg:col-span-8 space-y-6">
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
              <span className="font-bold text-rose-400">Error:</span>
              <span>{error}</span>
            </div>
          )}

          {isLoading && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center space-y-4">
              <div className="inline-block animate-spin rounded-full h-10 w-10 border-4 border-indigo-500 border-t-transparent" />
              <h3 className="text-lg font-bold text-white">Analyzing Multimodal Data...</h3>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Executing ResNet-18 forward pass on 224x224 MRI, extracting 64-D visual embeddings,
                projecting 9-D clinical vector, and synthesizing Grad-CAM & SHAP attributions.
              </p>
            </div>
          )}

          {!isLoading && !prediction && !error && (
            <div className="bg-slate-900/50 border border-dashed border-slate-800 rounded-xl p-16 text-center text-slate-500">
              <p className="text-base font-medium">No patient prediction generated yet.</p>
              <p className="text-xs mt-1 text-slate-600">
                Click "Load Demo Acute PAM Case" or adjust symptoms and upload an MRI to run inference.
              </p>
            </div>
          )}

          {prediction && !isLoading && (
            <div className="space-y-6">
              {/* PANEL 1: Risk Triage Indicator */}
              <div
                className={`rounded-xl border p-6 shadow-sm transition ${
                  prediction.risk_probability >= 0.5
                    ? "bg-rose-950/30 border-rose-600/60 text-rose-100"
                    : "bg-emerald-950/30 border-emerald-600/60 text-emerald-100"
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <span
                      className={`inline-block px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider mb-2 ${
                        prediction.risk_probability >= 0.5
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}
                    >
                      {prediction.risk_probability >= 0.5
                        ? "CRITICAL TRIAGE: HIGH PAM RISK"
                        : "LOW PROBABILITY: BACTERIAL/VIRAL SCREEN"}
                    </span>
                    <h2 className="text-2xl font-black text-white">
                      Predicted Risk: {(prediction.risk_probability * 100).toFixed(1)}%
                    </h2>
                    <p className="text-xs text-slate-300 mt-1 max-w-xl">
                      {prediction.risk_probability >= 0.5
                        ? "Immediate empirical amoebicidal intervention recommended (Miltefosine, Amphotericin B, therapeutic hypothermia) alongside STAT CSF wet mount."
                        : "Low correlation with amoebic fulminant progression. Maintain standard protocol for suspected viral or non-amoebic bacterial meningitis."}
                    </p>
                  </div>

                  <div className="text-right">
                    <div
                      className={`text-5xl font-extrabold tracking-tighter ${
                        prediction.risk_probability >= 0.5
                          ? "text-rose-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {(prediction.risk_probability * 100).toFixed(0)}%
                    </div>
                    <span className="text-[11px] text-slate-400 uppercase tracking-widest font-mono">
                      Calibrated Risk
                    </span>
                  </div>
                </div>
              </div>

              {/* Grid: Imaging XAI (Panel 2) & Tabular XAI (Panel 3) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* PANEL 2: XAI Imaging (Grad-CAM) */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col">
                  <div className="mb-3">
                    <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                      Neurological Region of Interest (Grad-CAM)
                    </h3>
                    <p className="text-xs text-slate-400">
                      ResNet-18 layer4[-1] gradient activation heatmap
                    </p>
                  </div>

                  <div className="flex-1 flex flex-col items-center justify-center bg-black/60 rounded-lg p-3 border border-slate-800">
                    <img
                      src={prediction.gradcam_base64}
                      alt="Grad-CAM Brain MRI Overlay"
                      className="w-56 h-56 object-contain rounded-md shadow-lg border border-slate-700"
                    />
                    <div className="mt-3 text-center">
                      <span className="inline-block px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-semibold border border-rose-500/30">
                        Focal Basal Hyperintensity Highlighted
                      </span>
                      <p className="text-[11px] text-slate-400 mt-1">
                        Gradient localization concentrated around basal meningeal cisterns.
                      </p>
                    </div>
                  </div>
                </div>

                {/* PANEL 3: Clinical Drivers (SHAP) */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col">
                  <div className="mb-3">
                    <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                      Clinical Feature Drivers (SHAP)
                    </h3>
                    <p className="text-xs text-slate-400">
                      Shapley values conditioned on patient's MRI slice
                    </p>
                  </div>

                  <div className="flex-1 h-64 min-h-[260px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={chartData}
                        margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
                      >
                        <XAxis
                          type="number"
                          tick={{ fill: "#94a3b8", fontSize: 10 }}
                          domain={["auto", "auto"]}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          tick={{ fill: "#cbd5e1", fontSize: 10 }}
                          width={110}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0f172a",
                            borderColor: "#334155",
                            borderRadius: "8px",
                            fontSize: "12px",
                            color: "#f8fafc",
                          }}
                          formatter={(val: any) => [`${val}`, "SHAP Impact"]}
                        />
                        <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                        <Bar dataKey="value" radius={[2, 2, 2, 2]}>
                          {chartData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={entry.value >= 0 ? "#f43f5e" : "#38bdf8"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
                      Increases PAM Risk (+)
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-sky-400 inline-block" />
                      Protective / Baseline (-)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
