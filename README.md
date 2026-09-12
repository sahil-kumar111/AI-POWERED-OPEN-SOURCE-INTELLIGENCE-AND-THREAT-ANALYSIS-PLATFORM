# AI-Powered Open-Source Intelligence and Threat Analysis Platform

An AI-driven multimodal intelligence platform that combines **video intelligence, open-source intelligence (OSINT), natural language processing (NLP), geospatial analysis, threat fusion, and explainable AI** to identify, correlate, and assess potential security threats.

The platform follows a modular and security-focused architecture where individual intelligence modules process their respective data sources and contribute structured intelligence to a centralized **Threat Fusion and Analysis Engine**.

---

## 🚀 Overview

Modern security monitoring systems generate large amounts of heterogeneous data from:

- Surveillance and camera feeds
- Weapon detection systems
- Person detection systems
- Fire and smoke detection
- Crowd monitoring
- Object tracking
- News and publicly available information
- Open-source incident reports
- Geospatial information
- Textual intelligence

Analyzing these sources independently can result in fragmented intelligence.

This platform addresses this problem through a unified multimodal intelligence pipeline.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────────┐
                         │       DATA SOURCES      │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          ┌───────────────────┐               ┌───────────────────┐
          │ VIDEO INTELLIGENCE│               │ OSINT INTELLIGENCE│
          └─────────┬─────────┘               └─────────┬─────────┘
                    │                                   │
       ┌────────────┼────────────┐                      ▼
       │            │            │             ┌───────────────────┐
       ▼            ▼            ▼             │ OSINT Collection  │
 ┌──────────┐ ┌──────────┐ ┌────────────┐      │ News / RSS / Web  │
 │ Weapon   │ │ Person   │ │ Fire/Smoke │      └─────────┬─────────┘
 │Detection │ │Detection │ │ Detection  │                │
 │  YOLO    │ │  YOLO    │ │    YOLO    │                ▼
 └────┬─────┘ └────┬─────┘ └────────────┘      ┌───────────────────┐
      │            │                           │  NLP Intelligence │
      └──────┬─────┘                           │                  │
             │                                 │ • Entity Extract │
             ▼                                 │ • Sentiment       │
     ┌──────────────────┐                      │ • Threat Classify │
     │    ByteTrack     │                      └─────────┬─────────┘
     │ Multi-Object     │                                │
     │    Tracking      │                                │
     │                  │                                │
     │ • Object IDs     │                                │
     │ • Track History  │                                │
     │ • Movement       │                                │
     └────────┬─────────┘                                │
              │                                          │
              ▼                                          │
     ┌──────────────────┐                                │
     │ Crowd Counting   │                                │
     │ & Scene Analysis │                                │
     └────────┬─────────┘                                │
              │                                          │
              └──────────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     THREAT FUSION        │
                    │         ENGINE            │
                    │                          │
                    │ • Event Correlation      │
                    │ • Multimodal Fusion      │
                    │ • Confidence Aggregation │
                    │ • Temporal Correlation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   GEOSPATIAL ANALYSIS    │
                    │                          │
                    │ • Event Mapping          │
                    │ • Spatial Correlation    │
                    │ • Hotspot Detection      │
                    │ • Location Intelligence  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    EXPLAINABLE AI        │
                    │      DECISION LAYER      │
                    │                          │
                    │ • SHAP                   │
                    │ • Feature Importance     │
                    │ • Evidence Attribution   │
                    │ • Decision Explanation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       THREAT INDEX       │
                    │                          │
                    │ • Risk Score              │
                    │ • Threat Level            │
                    │ • Confidence              │
                    │ • Severity                │
                    │ • Evidence                │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       SECURE API         │
                    │        FastAPI            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        DASHBOARD         │
                    │                          │
                    │ • Threat Monitoring      │
                    │ • Video Intelligence     │
                    │ • Geospatial Maps        │
                    │ • Threat Explanations    │
                    └──────────────────────────┘
