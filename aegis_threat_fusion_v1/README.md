# AEGIS Threat Fusion

This package is integrated with the existing project by `run_aegis_fusion.py`. Read [`../PHASE4_INTEGRATION_GUIDE.md`](../PHASE4_INTEGRATION_GUIDE.md) for the current report schemas, safe PowerShell commands, timestamp/location requirements, publication-time behavior and test commands.

The engine is rule-based, not a calibrated probability. It preserves configured weights and uses a strict `> 0.60` threshold for weapon/fire/smoke. It consumes saved JSON only; no model inference or embedding generation is invoked by Threat Fusion.

Run its unit/integration tests from this directory's parent:

```powershell
python -m unittest discover -s threat_fusion_v1/tests -v
```
