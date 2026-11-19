# Known limitations

This initial checked-in implementation establishes the database design, secure authentication, synthetic data generator, core univariate PSI/KS/JSD calculations, normalisation, DIS variants and decision engine. The following requested parts are not yet implemented: dataset/model/batch/label REST routers; reference-profile persistence; ADWIN state persistence; SHAP and sensitivity services; complete monitoring orchestration; retraining endpoints; Streamlit multi-page interface; experiment runner; seed/demo/query scripts; and integration tests. Consequently, it is not yet the complete end-to-end project specified in the supplied brief.

Even after completion, use is limited to simulated tabular binary classification; SHAP is attribution rather than causality, univariate detectors can miss multivariate drift, and proposed DIS thresholds need calibration.
