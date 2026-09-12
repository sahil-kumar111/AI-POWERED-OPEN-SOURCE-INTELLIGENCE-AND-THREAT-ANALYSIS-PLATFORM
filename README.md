# AI-Powered Open-Source Intelligence and Threat Analysis Platform

An AI-driven multimodal intelligence platform that combines **video intelligence, open-source intelligence (OSINT), natural language processing, geospatial analysis, threat fusion, and explainable AI** to identify, correlate, and assess potential security threats.

The platform is designed as a modular and security-focused architecture where individual intelligence modules independently process their respective data sources and contribute structured intelligence to a centralized **Threat Fusion and Analysis Engine**.

---

## 🚀 Overview

Modern security monitoring systems generate massive amounts of heterogeneous data from:

- Surveillance and camera feeds
- Weapon and person detection systems
- Fire and smoke detection
- Crowd monitoring
- News and publicly available information
- Social and open-source reports
- Geospatial information
- Textual intelligence

Analyzing these sources independently can result in fragmented intelligence.

This project addresses that problem by building a unified multimodal intelligence pipeline:

```text
                    ┌──────────────────────────┐
                    │     DATA SOURCES         │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
      ┌──────────────────┐              ┌──────────────────┐
      │ VIDEO INTELLIGENCE│              │ OSINT INTELLIGENCE│
      └─────────┬────────┘              └─────────┬────────┘
                │                                 │
                ▼                                 ▼
      ┌──────────────────┐              ┌──────────────────┐
      │ Weapon Detection │              │ News / RSS       │
      │ Person Detection │              │ Public Sources   │
      │ Fire Detection   │              │ Event Reports    │
      │ Smoke Detection  │              └─────────┬────────┘
      │ Crowd Counting   │                        │
      │ Object Tracking  │                        ▼
      └─────────┬────────┘              ┌──────────────────┐
                │                       │ NLP Intelligence │
                │                       │ Entity Extraction│
                │                       │ Sentiment        │
                │                       │ Threat Detection │
                │                       └─────────┬────────┘
                │                                 │
                └───────────────┬─────────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │    THREAT FUSION ENGINE  │
                    │                          │
                    │ Multimodal Correlation   │
                    │ Event Correlation        │
                    │ Confidence Aggregation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   GEOSPATIAL ANALYSIS    │
                    │                          │
                    │ Location Intelligence    │
                    │ Hotspot Detection        │
                    │ Event Mapping             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  EXPLAINABLE AI LAYER    │
                    │                          │
                    │ SHAP / Feature Importance│
                    │ Evidence Attribution     │
                    │ Decision Explanation     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      THREAT INDEX        │
                    │                          │
                    │ Risk Score               │
                    │ Severity                 │
                    │ Confidence               │
                    │ Supporting Evidence      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     SECURE FASTAPI       │
                    │          API             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       DASHBOARD          │
                    │                          │
                    │ Threat Monitoring        │
                    │ Maps & Hotspots          │
                    │ Video Intelligence       │
                    │ Explanations             │
                    └──────────────────────────┘
