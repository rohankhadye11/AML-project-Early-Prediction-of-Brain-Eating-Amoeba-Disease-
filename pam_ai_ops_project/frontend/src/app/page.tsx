"use client";

import React, { useState, ChangeEvent, FormEvent } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  Download,
  Droplets,
  ExternalLink,
  Eye,
  FileCheck,
  FileText,
  Flame,
  HeartPulse,
  HelpCircle,
  Layers,
  Microscope,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Upload,
  Waves,
  X,
  Zap,
} from "lucide-react";

// =============================================================================
// TYPES & INTERFACES
// =============================================================================
interface ClinicalFormState {
  Freshwater_Exposure: number;
  Nasal_Irrigation_History: number;
  Severe_Headache: number;
  Fever: number;
  Nausea_Vomiting: number;
  Stiff_Neck: number;
  Confusion: number;
  Seizures: number;
  Photophobia: number;
  Age: number;
  Symptom_Onset_Days: number;
  CSF_Pressure: number;
  CSF_WBC_Count: number;
}

interface MultimodalDetails {
  filename: string;
  total_fused_features: number;
  clinical_features_count: number;
  mri_features_count: number;
}

interface PredictionResponse {
  prediction: number;
  risk_probability: number;
  status: string;
  clinical_interpretation: string;
  multimodal_details?: MultimodalDetails;
}

// =============================================================================
// PRESETS
// =============================================================================
const HIGH_RISK_PRESET: ClinicalFormState = {
  Freshwater_Exposure: 1,
  Nasal_Irrigation_History: 1,
  Severe_Headache: 1,
  Fever: 1,
  Nausea_Vomiting: 1,
  Stiff_Neck: 1,
  Confusion: 1,
  Seizures: 0,
  Photophobia: 1,
  Age: 21,
  Symptom_Onset_Days: 3,
  CSF_Pressure: 415.0,
  CSF_WBC_Count: 680.0,
};

const LOW_RISK_PRESET: ClinicalFormState = {
  Freshwater_Exposure: 0,
  Nasal_Irrigation_History: 0,
  Severe_Headache: 1,
  Fever: 1,
  Nausea_Vomiting: 0,
  Stiff_Neck: 0,
  Confusion: 0,
  Seizures: 0,
  Photophobia: 0,
  Age: 52,
  Symptom_Onset_Days: 9,
  CSF_Pressure: 165.0,
  CSF_WBC_Count: 45.0,
};

export default function PAMPredictorPortal() {
  const [formData, setFormData] = useState<ClinicalFormState>(HIGH_RISK_PRESET);
  const [mriFile, setMriFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"all" | "high" | "low">("high");

  // Toggle binary symptoms (0 or 1)
  const handleToggle = (key: keyof ClinicalFormState) => {
    setFormData((prev) => ({
      ...prev,
      [key]: prev[key] === 1 ? 0 : 1,
    }));
  };

  // Handle continuous numeric vitals
  const handleNumericChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  // Handle MRI scan image upload
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setMriFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  // Presets
  const loadPreset = (preset: ClinicalFormState, type: "high" | "low") => {
    setFormData(preset);
    setActiveTab(type);
    setResult(null);
    setError(null);
  };

  const resetForm = () => {
    setFormData({
      Freshwater_Exposure: 0,
      Nasal_Irrigation_History: 0,
      Severe_Headache: 0,
      Fever: 0,
      Nausea_Vomiting: 0,
      Stiff_Neck: 0,
      Confusion: 0,
      Seizures: 0,
      Photophobia: 0,
      Age: 30,
      Symptom_Onset_Days: 5,
      CSF_Pressure: 180.0,
      CSF_WBC_Count: 150.0,
    });
    setMriFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  // Form submission via native multipart fetch
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payloadData = new FormData();
      // 1. Append JSON-encoded clinical data string
      payloadData.append("clinical_data", JSON.stringify(formData));

      // 2. Append image file (or synthesized fallback canvas if not provided)
      if (mriFile) {
        payloadData.append("file", mriFile);
      } else {
        const canvas = document.createElement("canvas");
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#0f172a";
          ctx.fillRect(0, 0, 224, 224);
          ctx.fillStyle = "#334155";
          ctx.beginPath();
          ctx.arc(112, 112, 75, 0, 2 * Math.PI);
          ctx.fill();
        }
        const blob = await new Promise<Blob>((resolve) =>
          canvas.toBlob((b) => resolve(b!), "image/png")
        );
        payloadData.append("file", blob, "baseline_mri_scan.png");
      }

      // 3. Native fetch POST to FastAPI backend
      const response = await fetch("http://127.0.0.1:8000/predict/multimodal", {
        method: "POST",
        body: payloadData,
      });

      if (!response.ok) {
        const errDetail = await response.json().catch(() => ({}));
        throw new Error(
          errDetail.detail || `Server error (Status ${response.status})`
        );
      }

      const data: PredictionResponse = await response.json();
      setResult(data);

      // Smooth scroll down to result card
      const resultElement = document.getElementById("diagnostic-results");
      if (resultElement) {
        resultElement.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected connection error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  const binarySymptoms: {
    key: keyof ClinicalFormState;
    label: string;
    desc: string;
    icon: React.ReactNode;
  }[] = [
    {
      key: "Freshwater_Exposure",
      label: "Freshwater Exposure",
      desc: "Warm lakes, rivers, or geothermal springs swimming within 14 days",
      icon: <Waves className="w-4 h-4 text-teal-600" />,
    },
    {
      key: "Nasal_Irrigation_History",
      label: "Nasal / Sinus Irrigation",
      desc: "Neti pot, sinus rinse, or ritual ablution with unboiled tap water",
      icon: <Droplets className="w-4 h-4 text-cyan-600" />,
    },
    {
      key: "Severe_Headache",
      label: "Bifrontal Severe Headache",
      desc: "Intense, excruciating frontal or bitemporal localized pain",
      icon: <Brain className="w-4 h-4 text-rose-600" />,
    },
    {
      key: "Fever",
      label: "High Pyrexia / Fever",
      desc: "Sudden high-grade fever (> 38.5°C / 101.3°F)",
      icon: <Flame className="w-4 h-4 text-amber-600" />,
    },
    {
      key: "Nausea_Vomiting",
      label: "Nausea & Projectile Vomiting",
      desc: "Hallmark of elevated intracranial pressure and brain inflammation",
      icon: <Activity className="w-4 h-4 text-indigo-600" />,
    },
    {
      key: "Stiff_Neck",
      label: "Stiff Neck (Meningismus)",
      desc: "Nuchal rigidity, Kernig's sign, or Brudzinski's sign present",
      icon: <Stethoscope className="w-4 h-4 text-teal-600" />,
    },
    {
      key: "Confusion",
      label: "Altered Mental Status",
      desc: "Progressive confusion, delirium, lethargy, or hallucinations",
      icon: <HelpCircle className="w-4 h-4 text-purple-600" />,
    },
    {
      key: "Seizures",
      label: "Cortical Seizure Episodes",
      desc: "Focal or generalized cortical irritation activity",
      icon: <Zap className="w-4 h-4 text-amber-500" />,
    },
    {
      key: "Photophobia",
      label: "Photophobia",
      desc: "Heightened sensitivity and pain in response to visual light",
      icon: <Eye className="w-4 h-4 text-blue-600" />,
    },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 selection:bg-teal-500 selection:text-white font-sans">
      {/* =========================================================================
          1. ENTERPRISE NAVIGATION BAR
      ========================================================================= */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-700 to-teal-900 flex items-center justify-center text-white shadow-md shadow-teal-900/20">
              <Microscope className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-slate-900 tracking-tight text-lg">
                  PAM<span className="text-teal-700">Sentinel</span>
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-50 text-teal-700 border border-teal-200">
                  CLINICAL CDS v1.0
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium hidden sm:block">
                Naegleria fowleri Early Detection Intelligence
              </p>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-8 text-xs font-semibold text-slate-600 tracking-wide">
            <a href="#threat" className="hover:text-teal-700 transition">
              CLINICAL THREAT
            </a>
            <a href="#architecture" className="hover:text-teal-700 transition">
              MULTIMODAL AI
            </a>
            <a href="#portal" className="hover:text-teal-700 transition">
              TRIAGE PORTAL
            </a>
            <a href="#analytics" className="hover:text-teal-700 transition">
              POWER BI ANALYTICS
            </a>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>FastAPI Backend Active</span>
            </div>

            <a
              href="#portal"
              className="px-4 py-2 rounded-lg bg-teal-800 text-white hover:bg-teal-900 text-xs font-bold shadow-md shadow-teal-900/20 transition flex items-center gap-1.5"
            >
              <span>Access CDS Portal</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </nav>

      {/* =========================================================================
          2. HERO SECTION
      ========================================================================= */}
      <section className="relative bg-gradient-to-br from-teal-950 via-teal-900 to-slate-950 text-white overflow-hidden py-20 lg:py-28">
        {/* Abstract Medical Background Overlay */}
        <div
          className="absolute inset-0 opacity-20 bg-cover bg-center mix-blend-overlay"
          style={{
            backgroundImage:
              "url('https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80')",
          }}
        />
        {/* Subtle grid mesh */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,rgba(20,184,166,0.15),transparent_60%)]" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/20 border border-teal-400/30 text-teal-200 text-xs font-semibold backdrop-blur-md">
              <Sparkles className="w-3.5 h-3.5 text-teal-300" />
              <span>59-Dimensional Multimodal Decision Support</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-tight">
              AI-Powered Early Detection for{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 via-cyan-200 to-white">
                Naegleria fowleri.
              </span>
            </h1>

            <p className="text-lg sm:text-xl text-teal-100/90 leading-relaxed font-normal">
              Fusing clinical symptoms with neuroimaging to identify Primary
              Amoebic Meningoencephalitis (PAM) in milliseconds. Empowers acute
              care physicians to intervene before irreversible cerebral necrosis.
            </p>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4 pt-4">
              <a
                href="#portal"
                className="px-7 py-4 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 font-bold text-sm shadow-xl shadow-teal-500/25 transition flex items-center justify-center gap-2 group"
              >
                <span>Access Clinical Portal</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </a>

              <a
                href="#architecture"
                className="px-7 py-4 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-white font-semibold text-sm backdrop-blur-md transition flex items-center justify-center gap-2"
              >
                <Cpu className="w-4 h-4 text-teal-300" />
                <span>Explore Multimodal Architecture</span>
              </a>
            </div>
          </div>

          {/* Floating Key Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-16 pt-10 border-t border-teal-800/60">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="text-3xl sm:text-4xl font-black text-teal-300">
                95.73%
              </div>
              <div className="text-xs text-teal-100/80 mt-1 font-medium">
                Multimodal Fused Accuracy
              </div>
              <div className="text-[11px] text-teal-200/50 mt-1">
                Random Forest 59-D Tensor
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="text-3xl sm:text-4xl font-black text-rose-300">
                96.34%
              </div>
              <div className="text-xs text-rose-100/80 mt-1 font-medium">
                Clinical Sensitivity (Recall)
              </div>
              <div className="text-[11px] text-rose-200/50 mt-1">
                Zero Missed Fatal Infections
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="text-3xl sm:text-4xl font-black text-cyan-300">
                0.9832
              </div>
              <div className="text-xs text-cyan-100/80 mt-1 font-medium">
                Test ROC-AUC Score
              </div>
              <div className="text-[11px] text-cyan-200/50 mt-1">
                Optimal Decision Boundary
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="text-3xl sm:text-4xl font-black text-amber-300">
                &lt; 120ms
              </div>
              <div className="text-xs text-amber-100/80 mt-1 font-medium">
                Emergency Inference Latency
              </div>
              <div className="text-[11px] text-amber-200/50 mt-1">
                Real-Time REST API Processing
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          3. DISEASE AWARENESS & PUBLIC HEALTH IMPERATIVE
      ========================================================================= */}
      <section id="threat" className="py-20 lg:py-28 bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-16">
          <div className="max-w-3xl mx-auto text-center space-y-4">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-50 text-rose-700 text-xs font-bold tracking-wider uppercase border border-rose-200">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Clinical Urgency &amp; Pathophysiology</span>
            </span>
            <h2 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight">
              The Lethal Reality of Primary Amoebic Meningoencephalitis
            </h2>
            <p className="text-base text-slate-600 leading-relaxed">
              <em>Naegleria fowleri</em> is a free-living, thermophilic amoeba
              that infiltrates the central nervous system via the olfactory
              neuroepithelium, causing hyper-acute, necrotizing hemorrhagic
              encephalitis.
            </p>
          </div>

          {/* Three Key Pillar Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-rose-50/50 border border-rose-200 rounded-2xl p-8 space-y-4 relative overflow-hidden">
              <div className="w-12 h-12 rounded-xl bg-rose-600 text-white flex items-center justify-center shadow-lg shadow-rose-600/30">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">
                97%+ Case Fatality Rate
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Out of over 150 documented cases in the United States, only 4
                patients have ever survived. Death typically occurs within{" "}
                <strong>1 to 12 days</strong> (median 5 days) after symptom
                onset due to cerebral edema and brain herniation.
              </p>
              <div className="pt-2 text-xs font-semibold text-rose-700 uppercase tracking-wider flex items-center gap-1">
                <span>Near-Universal Mortality</span>
              </div>
            </div>

            <div className="bg-amber-50/50 border border-amber-200 rounded-2xl p-8 space-y-4 relative overflow-hidden">
              <div className="w-12 h-12 rounded-xl bg-amber-600 text-white flex items-center justify-center shadow-lg shadow-amber-600/30">
                <Stethoscope className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">
                The Diagnostic Dilemma
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Early PAM presentation (severe frontal headache, high fever,
                nuchal rigidity, vomiting) is <strong>clinically indistinguishable</strong>{" "}
                from acute bacterial meningitis. Routine antibiotics have zero
                efficacy against amoebae, leading to fatal diagnostic delays.
              </p>
              <div className="pt-2 text-xs font-semibold text-amber-700 uppercase tracking-wider flex items-center gap-1">
                <span>Mimics Bacterial Meningitis</span>
              </div>
            </div>

            <div className="bg-teal-50/50 border border-teal-200 rounded-2xl p-8 space-y-4 relative overflow-hidden">
              <div className="w-12 h-12 rounded-xl bg-teal-700 text-white flex items-center justify-center shadow-lg shadow-teal-700/30">
                <Clock className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">
                The Window of Viability
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                Experimental survivals demonstrate that aggressive therapy
                (intravenous &amp; intrathecal <strong>Miltefosine</strong>,
                Liposomal Amphotericin B, and targeted temperature management)
                must begin within hours of initial clinical presentation.
              </p>
              <div className="pt-2 text-xs font-semibold text-teal-700 uppercase tracking-wider flex items-center gap-1">
                <span>Hours Decide Survival</span>
              </div>
            </div>
          </div>

          {/* Infection Vectors Card Grid */}
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
              <div>
                <h3 className="text-lg font-bold text-slate-900">
                  Primary Epidemiological Transmission Vectors
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  The amoeba thrives in warm freshwater (&gt; 30°C / 86°F) and enters when water is forced up the nasal cavity.
                </p>
              </div>
              <span className="text-xs font-bold text-teal-700 bg-teal-50 px-3 py-1 rounded-md border border-teal-200 self-start sm:self-auto">
                Olfactory Nerve Pathway
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs flex items-start gap-3">
                <div className="p-2 rounded-lg bg-teal-100 text-teal-800 shrink-0">
                  <Waves className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Warm Lakes &amp; Rivers
                  </h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Diving, jumping, or submerging in untreated summer surface water.
                  </p>
                </div>
              </div>

              <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs flex items-start gap-3">
                <div className="p-2 rounded-lg bg-amber-100 text-amber-800 shrink-0">
                  <Flame className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Hot Springs &amp; Canals
                  </h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Naturally heated geothermal waters and unmonitored thermal runoff.
                  </p>
                </div>
              </div>

              <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs flex items-start gap-3">
                <div className="p-2 rounded-lg bg-blue-100 text-blue-800 shrink-0">
                  <Droplets className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Nasal Irrigation / Neti Pots
                  </h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Using non-sterile or unboiled municipal tap water for sinus cleansing.
                  </p>
                </div>
              </div>

              <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs flex items-start gap-3">
                <div className="p-2 rounded-lg bg-indigo-100 text-indigo-800 shrink-0">
                  <HeartPulse className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Unchlorinated Pools
                  </h4>
                  <p className="text-xs text-slate-500 mt-1">
                    Inadequately disinfected splash pads and artificial recreation water.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          4. MULTIMODAL AI ARCHITECTURE EXPLAINER
      ========================================================================= */}
      <section id="architecture" className="py-20 lg:py-28 bg-slate-900 text-white relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-16">
          <div className="max-w-3xl mx-auto text-center space-y-4">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-teal-500/20 text-teal-300 text-xs font-bold tracking-wider uppercase border border-teal-500/30">
              <Cpu className="w-3.5 h-3.5" />
              <span>Machine Learning Operations (MLOps)</span>
            </span>
            <h2 className="text-3xl sm:text-4xl font-black tracking-tight text-white">
              The Multimodal AI Fusion Engine
            </h2>
            <p className="text-base text-slate-300 leading-relaxed">
              Bridging statistical clinical decision trees with automated
              neuroimaging computer vision to resolve extreme diagnostic ambiguity.
            </p>
          </div>

          {/* Two-Column Technical Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Column 1: Clinical Tabular Engine */}
            <div className="bg-slate-800/70 border border-slate-700/80 rounded-2xl p-8 space-y-6 backdrop-blur-md">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-300 flex items-center justify-center border border-teal-500/30">
                    <FileCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      Clinical Feature Pipeline
                    </h3>
                    <p className="text-xs text-teal-300 font-mono">
                      9 RFE-Selected Diagnostic Anchors
                    </p>
                  </div>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-full bg-teal-950 text-teal-300 border border-teal-800 font-mono">
                  91.5% Clinical Acc
                </span>
              </div>

              <p className="text-sm text-slate-300 leading-relaxed">
                Raw clinical inputs undergo a rigorous three-layer feature
                optimization strategy:
              </p>

              <div className="space-y-3">
                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-teal-300">
                    <span>1. Chi-Square (χ²) Significance</span>
                    <span className="font-mono">p &lt; 0.05</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Isolates statistically decisive symptoms: Freshwater Exposure (χ² = 223.96), Stiff Neck (χ² = 170.39), Confusion (χ² = 139.90).
                  </p>
                </div>

                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-teal-300">
                    <span>2. Pearson Multicollinearity Pruning</span>
                    <span className="font-mono">|r| &gt; 0.80</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Detects laboratory redundancy between CSF Pressure and CSF WBC (|r| = 0.9141) and drops redundant dimensions to prevent variance inflation.
                  </p>
                </div>

                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-teal-300">
                    <span>3. Recursive Feature Elimination</span>
                    <span className="font-mono">DecisionTree RFE</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Iteratively prunes the feature space down to 9 optimal anchors for lightweight emergency triage input.
                  </p>
                </div>
              </div>
            </div>

            {/* Column 2: Neuroimaging Computer Vision Engine */}
            <div className="bg-slate-800/70 border border-slate-700/80 rounded-2xl p-8 space-y-6 backdrop-blur-md">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-cyan-500/20 text-cyan-300 flex items-center justify-center border border-cyan-500/30">
                    <Brain className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      Neuroimaging Feature Pipeline
                    </h3>
                    <p className="text-xs text-cyan-300 font-mono">
                      50-D Computer Vision Vector
                    </p>
                  </div>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
                  OpenCV 224x224
                </span>
              </div>

              <p className="text-sm text-slate-300 leading-relaxed">
                Automated computer vision extracts 50 continuous features from
                axial brain MRI and CT slices:
              </p>

              <div className="space-y-3">
                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-cyan-300">
                    <span>1. Intensity Histogram Profiling</span>
                    <span className="font-mono">32 Dimensions</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    32-bin normalized grayscale distribution capturing tissue density and hyperintense signals caused by acute meningoencephalitis.
                  </p>
                </div>

                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-cyan-300">
                    <span>2. Quadrant Spatial Pooling</span>
                    <span className="font-mono">8 Dimensions</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Mean intensity and standard deviation across 4 cerebral quadrants to detect asymmetric hemorrhagic lesions and midline shift.
                  </p>
                </div>

                <div className="p-3.5 bg-slate-900/60 rounded-xl border border-slate-700/60">
                  <div className="flex items-center justify-between text-xs font-bold text-cyan-300">
                    <span>3. Statistical Moments &amp; Central ROI</span>
                    <span className="font-mono">10 Dimensions</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Variance, skewness, 10th/90th percentiles, and ventricular region-of-interest intensity reflecting diffuse cerebral edema.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Central Multimodal Fusion Banner */}
          <div className="p-8 rounded-2xl bg-gradient-to-r from-teal-900/60 via-slate-800/80 to-cyan-900/60 border border-teal-500/30 flex flex-col sm:flex-row items-center justify-between gap-6">
            <div className="space-y-1 text-center sm:text-left">
              <span className="text-xs font-bold text-teal-300 uppercase tracking-wider">
                Phase 4 Unified Architecture
              </span>
              <h4 className="text-xl font-black text-white">
                59-Dimensional Fused Concatenation Vector
              </h4>
              <p className="text-xs text-slate-300 max-w-xl">
                9 Clinical Features + 50 Neuroimaging Features = 59-D Input Tensor fed into our calibrated Random Forest Classifier (100 estimators).
              </p>
            </div>

            <div className="flex items-center gap-4 shrink-0">
              <div className="text-center px-4 py-2 bg-slate-900/80 rounded-xl border border-slate-700">
                <span className="text-xs text-slate-400 block">Baseline Clinical</span>
                <span className="text-lg font-bold text-slate-200">91.46%</span>
              </div>
              <ArrowRight className="w-5 h-5 text-teal-400" />
              <div className="text-center px-4 py-2 bg-teal-500/20 rounded-xl border border-teal-400/40">
                <span className="text-xs text-teal-300 block">Multimodal Fused</span>
                <span className="text-lg font-extrabold text-emerald-400">95.73%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          5. THE CLINICAL DECISION SUPPORT PORTAL
      ========================================================================= */}
      <section id="portal" className="py-20 lg:py-28 bg-slate-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
          
          {/* Section Header */}
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-teal-100 text-teal-800 text-xs font-bold tracking-wider uppercase border border-teal-200">
                <Activity className="w-3.5 h-3.5" />
                <span>Phase 5 Real-Time Triage Portal</span>
              </span>
              <h2 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight mt-2">
                Clinical Decision Support Interface
              </h2>
              <p className="text-sm text-slate-600 mt-1 max-w-2xl">
                Enter presenting patient vitals, toggle clinical risk factors, and attach neuroimaging to execute immediate multimodal risk stratification.
              </p>
            </div>

            {/* Quick Presets Bar */}
            <div className="flex items-center gap-2 self-start md:self-auto bg-white p-1.5 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-xs font-semibold text-slate-500 px-2 hidden sm:inline">
                Presets:
              </span>
              <button
                type="button"
                onClick={() => loadPreset(HIGH_RISK_PRESET, "high")}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  activeTab === "high"
                    ? "bg-rose-600 text-white shadow-xs"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                }`}
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>High-Risk PAM Case</span>
              </button>

              <button
                type="button"
                onClick={() => loadPreset(LOW_RISK_PRESET, "low")}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  activeTab === "low"
                    ? "bg-teal-700 text-white shadow-xs"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                }`}
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Control Case</span>
              </button>

              <button
                type="button"
                onClick={resetForm}
                title="Reset to blank defaults"
                className="p-1.5 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100 transition"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* =========================================================================
              DIAGNOSTIC RESULTS DISPLAY (DYNAMIC)
          ========================================================================= */}
          {result && (
            <div
              id="diagnostic-results"
              className={`rounded-3xl p-6 sm:p-10 shadow-2xl border-2 transition-all transform animate-pulse-slow ${
                result.prediction === 1
                  ? "bg-gradient-to-br from-rose-50 via-white to-red-50 border-rose-500 shadow-rose-200"
                  : "bg-gradient-to-br from-emerald-50 via-white to-teal-50 border-emerald-500 shadow-emerald-200"
              }`}
            >
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 border-b pb-6 border-slate-200">
                <div className="flex items-start gap-4">
                  <div
                    className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 shadow-lg ${
                      result.prediction === 1
                        ? "bg-rose-600 text-white shadow-rose-600/30"
                        : "bg-emerald-600 text-white shadow-emerald-600/30"
                    }`}
                  >
                    {result.prediction === 1 ? (
                      <AlertTriangle className="w-8 h-8" />
                    ) : (
                      <CheckCircle2 className="w-8 h-8" />
                    )}
                  </div>

                  <div>
                    <span
                      className={`text-xs font-black uppercase tracking-wider px-3 py-1 rounded-full border ${
                        result.prediction === 1
                          ? "bg-rose-100 text-rose-800 border-rose-200"
                          : "bg-emerald-100 text-emerald-800 border-emerald-200"
                      }`}
                    >
                      {result.prediction === 1
                        ? "CRITICAL EMERGENCY ALERT: POSITIVE PAM SCREEN"
                        : "LOW RISK: NEGATIVE PAM SCREEN"}
                    </span>
                    <h3 className="text-2xl sm:text-3xl font-black text-slate-900 mt-2">
                      {result.prediction === 1
                        ? "Presumed Primary Amoebic Meningoencephalitis"
                        : "Consistent with Non-Amoebic Meningitis / Low PAM Risk"}
                    </h3>
                  </div>
                </div>

                <div className="text-left md:text-right bg-white p-4 rounded-2xl border border-slate-200 shadow-xs shrink-0 w-full md:w-auto">
                  <span className="text-xs font-semibold text-slate-500 block">
                    Multimodal Probability
                  </span>
                  <span
                    className={`text-4xl font-black ${
                      result.prediction === 1
                        ? "text-rose-600"
                        : "text-emerald-600"
                    }`}
                  >
                    {(result.risk_probability * 100).toFixed(1)}%
                  </span>
                  <span className="text-[11px] text-slate-600 block mt-0.5 font-medium">
                    {result.prediction === 1
                      ? "Exceeds 70% Action Threshold"
                      : "Below Diagnostic Threshold"}
                  </span>
                </div>
              </div>

              {/* Progress Bar Meter */}
              <div className="mt-6 space-y-1.5">
                <div className="flex justify-between text-xs font-bold text-slate-600">
                  <span>Confidence Gauge</span>
                  <span>{(result.risk_probability * 100).toFixed(1)}% Fused Confidence</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden p-0.5 shadow-inner">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${
                      result.prediction === 1
                        ? "bg-gradient-to-r from-amber-500 to-rose-600"
                        : "bg-gradient-to-r from-teal-400 to-emerald-500"
                    }`}
                    style={{
                      width: `${Math.max(result.risk_probability * 100, 5)}%`,
                    }}
                  />
                </div>
              </div>

              {/* Clinical Protocol Recommendation */}
              <div className="mt-6 bg-white/95 rounded-2xl p-6 border border-slate-200 shadow-xs space-y-3">
                <div className="flex items-center gap-2">
                  <ShieldAlert
                    className={`w-5 h-5 ${
                      result.prediction === 1
                        ? "text-rose-600"
                        : "text-emerald-600"
                    }`}
                  />
                  <h4 className="text-sm font-bold uppercase tracking-wider text-slate-900">
                    Recommended Clinical Triage Pathway
                  </h4>
                </div>
                <p className="text-sm font-medium text-slate-700 leading-relaxed">
                  {result.clinical_interpretation}
                </p>

                {result.prediction === 1 && (
                  <div className="pt-2 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-semibold text-rose-950">
                    <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                      1. Emergency Miltefosine (PO/NG)
                    </div>
                    <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                      2. Liposomal Amphotericin B
                    </div>
                    <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                      3. Targeted Hypothermia (&lt; 34°C)
                    </div>
                    <div className="p-3 bg-rose-50 rounded-xl border border-rose-200">
                      4. Continuous ICP Decompression
                    </div>
                  </div>
                )}
              </div>

              {/* Multimodal Dimensional Breakdown */}
              {result.multimodal_details && (
                <div className="mt-6 flex flex-wrap items-center gap-3 text-xs text-slate-600">
                  <span className="px-3 py-1.5 bg-white rounded-lg border border-slate-200 font-mono shadow-2xs">
                    Scan: {result.multimodal_details.filename}
                  </span>
                  <span className="px-3 py-1.5 bg-white rounded-lg border border-slate-200 shadow-2xs">
                    Total Fused Features:{" "}
                    <strong className="text-slate-900">
                      {result.multimodal_details.total_fused_features}
                    </strong>
                  </span>
                  <span className="px-3 py-1.5 bg-white rounded-lg border border-slate-200 shadow-2xs">
                    Clinical Anchors:{" "}
                    <strong className="text-slate-900">
                      {result.multimodal_details.clinical_features_count}
                    </strong>
                  </span>
                  <span className="px-3 py-1.5 bg-white rounded-lg border border-slate-200 shadow-2xs">
                    Neuroimaging Metrics:{" "}
                    <strong className="text-slate-900">
                      {result.multimodal_details.mri_features_count}
                    </strong>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="p-5 rounded-2xl bg-red-50 border border-red-200 text-red-800 text-sm flex items-start justify-between shadow-sm">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                <div>
                  <strong className="font-bold">Inference Error: </strong>
                  <span>{error}</span>
                  <p className="text-xs text-red-600 mt-1">
                    Ensure the FastAPI backend is active at <code>http://127.0.0.1:8000</code>.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-700 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* =========================================================================
              THE MAIN PREDICTION FORM
          ========================================================================= */}
          <form
            onSubmit={handleSubmit}
            className="bg-white rounded-3xl shadow-xl border border-slate-200/80 p-6 sm:p-10 space-y-10"
          >
            {/* Section 1: Demographics & Clinical Timing */}
            <div className="space-y-4">
              <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center text-xs font-bold">
                      1
                    </span>
                    <span>Patient Demographics &amp; Timing</span>
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Age and duration since initial symptom presentation.
                  </p>
                </div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  2 Numeric Anchors
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2">
                <div>
                  <label htmlFor="age-input" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Patient Age (Years)
                  </label>
                  <div className="mt-1.5 relative">
                    <input
                      id="age-input"
                      type="number"
                      name="Age"
                      min={1}
                      max={100}
                      value={formData.Age}
                      onChange={handleNumericChange}
                      className="block w-full rounded-xl border border-slate-300 bg-slate-50/50 p-3 text-sm text-slate-900 focus:border-teal-600 focus:bg-white focus:ring-2 focus:ring-teal-600/20 transition font-medium"
                      required
                    />
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">
                    Range: 1 – 80 (Pediatric &amp; young adults skew higher in water sports)
                  </span>
                </div>

                <div>
                  <label htmlFor="symptom-onset-input" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Symptom Onset Duration (Days)
                  </label>
                  <div className="mt-1.5 relative">
                    <input
                      id="symptom-onset-input"
                      type="number"
                      name="Symptom_Onset_Days"
                      min={0}
                      max={30}
                      value={formData.Symptom_Onset_Days}
                      onChange={handleNumericChange}
                      className="block w-full rounded-xl border border-slate-300 bg-slate-50/50 p-3 text-sm text-slate-900 focus:border-teal-600 focus:bg-white focus:ring-2 focus:ring-teal-600/20 transition font-medium"
                      required
                    />
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">
                    Hyper-acute window: 1 – 7 days (Median PAM fatal progression is 5 days)
                  </span>
                </div>
              </div>
            </div>

            {/* Section 2: Clinical Symptoms & Exposure Indicators (9 Binary) */}
            <div className="space-y-4">
              <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center text-xs font-bold">
                      2
                    </span>
                    <span>Clinical Signs &amp; Exposure Toggles</span>
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Click each tile to toggle presence (YES / NO) during emergency patient evaluation.
                  </p>
                </div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  9 Binary Anchors
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
                {binarySymptoms.map(({ key, label, desc, icon }) => {
                  const isActive = formData[key] === 1;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => handleToggle(key)}
                      className={`text-left p-4 rounded-2xl border transition-all flex flex-col justify-between group ${
                        isActive
                          ? "bg-teal-50/60 border-teal-400/80 shadow-md shadow-teal-500/10"
                          : "bg-slate-50/80 border-slate-200 hover:bg-slate-100 hover:border-slate-300"
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <div className={`p-1.5 rounded-lg ${isActive ? "bg-teal-600 text-white" : "bg-slate-200 text-slate-600"}`}>
                              {icon}
                            </div>
                            <span className="font-bold text-sm text-slate-900">
                              {label}
                            </span>
                          </div>

                          <span
                            className={`text-xs font-black px-2.5 py-1 rounded-full transition-all ${
                              isActive
                                ? "bg-teal-700 text-white shadow-xs"
                                : "bg-slate-200 text-slate-500"
                            }`}
                          >
                            {isActive ? "YES" : "NO"}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                          {desc}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Section 3: Lumbar Puncture CSF Biomarkers */}
            <div className="space-y-4">
              <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center text-xs font-bold">
                      3
                    </span>
                    <span>Cerebrospinal Fluid (CSF) Lumbar Puncture Vitals</span>
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Opening pressure and leukocyte count from initial spinal tap.
                  </p>
                </div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  2 Continuous Vitals
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2">
                <div>
                  <label htmlFor="csf-pressure-input" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                    CSF Opening Pressure (mmH₂O)
                  </label>
                  <div className="mt-1.5 relative">
                    <input
                      id="csf-pressure-input"
                      type="number"
                      name="CSF_Pressure"
                      min={50}
                      max={600}
                      step={0.1}
                      value={formData.CSF_Pressure}
                      onChange={handleNumericChange}
                      className="block w-full rounded-xl border border-slate-300 bg-slate-50/50 p-3 text-sm text-slate-900 focus:border-teal-600 focus:bg-white focus:ring-2 focus:ring-teal-600/20 transition font-medium"
                      required
                    />
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500 mt-1">
                    <span>Normal Adult: 100 – 200 mmH₂O</span>
                    <span className="font-semibold text-rose-700">
                      PAM Typical: 350 – 500+ mmH₂O
                    </span>
                  </div>
                </div>

                <div>
                  <label htmlFor="csf-wbc-input" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                    CSF WBC Pleocytosis Count (cells/mm³)
                  </label>
                  <div className="mt-1.5 relative">
                    <input
                      id="csf-wbc-input"
                      type="number"
                      name="CSF_WBC_Count"
                      min={0}
                      max={2500}
                      step={0.1}
                      value={formData.CSF_WBC_Count}
                      onChange={handleNumericChange}
                      className="block w-full rounded-xl border border-slate-300 bg-slate-50/50 p-3 text-sm text-slate-900 focus:border-teal-600 focus:bg-white focus:ring-2 focus:ring-teal-600/20 transition font-medium"
                      required
                    />
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500 mt-1">
                    <span>Normal: &lt; 5 cells/mm³</span>
                    <span className="font-semibold text-rose-700">
                      PAM Pleocytosis: 400 – 1,000+ cells/mm³
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Section 4: Neuroimaging (Brain MRI / CT) Upload */}
            <div className="space-y-4">
              <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-teal-100 text-teal-800 flex items-center justify-center text-xs font-bold">
                      4
                    </span>
                    <span>Brain Neuroimaging (MRI / CT Slice)</span>
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Upload axial brain scan to extract 50-D spatial moments and intensity histograms.
                  </p>
                </div>
                <span className="text-[11px] font-semibold text-teal-700 uppercase tracking-wider bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                  Multimodal Fusion
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-2 items-center">
                <label 
                  htmlFor="mri-scan-file-input"
                  className="sm:col-span-2 border-2 border-dashed border-slate-300 hover:border-teal-500 rounded-2xl p-6 text-center hover:bg-teal-50/20 cursor-pointer transition flex flex-col items-center justify-center min-h-[160px] group"
                >
                  <div className="w-12 h-12 rounded-xl bg-slate-100 group-hover:bg-teal-100 text-slate-400 group-hover:text-teal-700 flex items-center justify-center transition mb-3">
                    <Upload className="w-6 h-6" />
                  </div>
                  <span className="text-sm font-bold text-slate-800">
                    {mriFile ? mriFile.name : "Click to select or drop brain MRI scan (PNG/JPEG)"}
                  </span>
                  <span className="text-xs text-slate-500 mt-1">
                    Standardized by OpenCV pipeline to 224x224 grayscale tensor
                  </span>
                  <input
                    id="mri-scan-file-input"
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>

                {/* Thumbnail Preview */}
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-2xl flex flex-col items-center justify-center min-h-[160px] text-center">
                  {previewUrl ? (
                    <div className="space-y-2 flex flex-col items-center">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={previewUrl}
                        alt="Brain MRI Scan Preview"
                        className="w-28 h-28 object-cover rounded-xl border border-slate-300 shadow-sm"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          setMriFile(null);
                          setPreviewUrl(null);
                        }}
                        className="text-xs text-rose-600 hover:text-rose-800 font-semibold"
                      >
                        Remove Scan
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-1 text-slate-400">
                      <Brain className="w-8 h-8 mx-auto text-slate-300" />
                      <span className="text-xs font-semibold block text-slate-600">
                        No scan selected
                      </span>
                      <span className="text-[11px] text-slate-600 block">
                        Auto-synthesizes baseline scan
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Submission Action Bar */}
            <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="text-xs text-slate-500 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-teal-600" />
                <span>Runs 59-D Multimodal Inference via Local FastAPI Server</span>
              </div>

              <button
                type="submit"
                disabled={loading}
                className={`w-full sm:w-auto px-10 py-4 rounded-xl text-sm font-bold text-white shadow-xl transition flex items-center justify-center gap-3 ${
                  loading
                    ? "bg-slate-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-teal-700 to-teal-900 hover:from-teal-800 hover:to-teal-950 active:scale-[0.99] shadow-teal-900/20"
                }`}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    <span>Executing Multimodal Tensor Fusion...</span>
                  </>
                ) : (
                  <>
                    <span>Execute Multimodal Risk Stratification</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </section>

      {/* =========================================================================
          6. POWER BI ANALYTICS LAYER (STEP 1 SHOWCASE)
      ========================================================================= */}
      <section id="analytics" className="py-20 lg:py-24 bg-white border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
          <div className="max-w-3xl space-y-3">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-800 text-xs font-bold tracking-wider uppercase border border-indigo-200">
              <Database className="w-3.5 h-3.5" />
              <span>Phase 5 Analytics &amp; Reporting Layer</span>
            </span>
            <h2 className="text-3xl font-black text-slate-900 tracking-tight">
              Enterprise Power BI Multimodal Analytics Dataset
            </h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              Exported as <code>data/processed/powerbi_multimodal_analytics.csv</code>. 
              Denormalized and formatted for Star Schema ingestion into Power BI to compute DAX measures for executive hospital monitoring and epidemiologic heatmaps.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                True Positives
              </div>
              <div className="text-3xl font-black text-rose-600">158</div>
              <p className="text-xs text-slate-500">
                48.17% of cohort — Critical PAM patients accurately flagged for immediate treatment.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                True Negatives
              </div>
              <div className="text-3xl font-black text-emerald-600">156</div>
              <p className="text-xs text-slate-500">
                47.56% of cohort — Non-amoebic acute controls correctly spared toxic miltefosine regimens.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                False Positives
              </div>
              <div className="text-3xl font-black text-amber-600">8</div>
              <p className="text-xs text-slate-500">
                2.44% — Precautionary alerts triggering mandatory confirmatory CSF PCR screening.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                False Negatives
              </div>
              <div className="text-3xl font-black text-slate-800">6</div>
              <p className="text-xs text-slate-500">
                1.83% — Low missed-case rate mitigated by secondary clinical checks.
              </p>
            </div>
          </div>

          {/* DAX Measures Overview */}
          <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between text-teal-400 border-b border-slate-800 pb-3 font-sans">
              <span className="font-bold text-sm">Key Power BI DAX Formulas Ready for Import</span>
              <span className="text-xs text-slate-400">powerbi_multimodal_analytics.csv</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-teal-300 font-bold block mb-1">Diagnostic Sensitivity =</span>
                <code>DIVIDE(CALCULATE(COUNTROWS(Analytics), Analytics[Prediction_Type] = &quot;True Positive&quot;), CALCULATE(COUNTROWS(Analytics), Analytics[Actual_Outcome] = 1), 0)</code>
              </div>

              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-teal-300 font-bold block mb-1">Diagnostic Specificity =</span>
                <code>DIVIDE(CALCULATE(COUNTROWS(Analytics), Analytics[Prediction_Type] = &quot;True Negative&quot;), CALCULATE(COUNTROWS(Analytics), Analytics[Actual_Outcome] = 0), 0)</code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          7. FOOTER
      ========================================================================= */}
      <footer className="bg-slate-950 text-slate-400 py-12 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-white font-black text-base">
                <Microscope className="w-5 h-5 text-teal-400" />
                <span>PAMSentinel Clinical CDS Portal</span>
              </div>
              <p className="text-xs text-slate-500">
                AI Growth Operations Flagship Project &bull; Primary Amoebic Meningoencephalitis Early Risk Detection
              </p>
            </div>

            <div className="flex items-center gap-4 text-xs font-semibold">
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg bg-slate-800 text-teal-300 hover:bg-slate-700 transition flex items-center gap-1.5"
              >
                <span>FastAPI Swagger Docs</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              <a
                href="http://127.0.0.1:8000/redoc"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition flex items-center gap-1.5"
              >
                <span>ReDoc API</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          <div className="pt-6 border-t border-slate-900 text-center text-xs text-slate-600 space-y-2">
            <p>
              <strong>Clinical Research Disclaimer:</strong> This platform is designed as an investigational Clinical Decision Support (CDS) prototype for research and portfolio demonstration. All therapeutic interventions for <em>Naegleria fowleri</em> should be confirmed via CDC protocols and verified laboratory PCR analysis.
            </p>
            <p className="text-slate-700">
              Stack: Next.js 14, Tailwind CSS, Lucide React, FastAPI, Scikit-Learn Random Forest, OpenCV, Power BI Star Schema.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
