CAFOAD revision code and public fused datasets
================================================

This repository contains the code and public fused data released with the
revised CAFOAD manuscript. The Python files are uploaded without changing the
experiment implementation. Configure the data and result paths described below
before running them on another computer.

Five principal code files
-------------------------

1. combinenew.py
   Reconstructs the cyber-physical datasets using the common, label-blind
   200-ms causal-grid fusion rule. The corresponding raw source files are
   required when regenerating the fused CSV files.

2. CAFOAD_major_revision.py
   Implements CAFOAD, the matched baseline models, train-only preprocessing,
   grouped training and inference, robustness evaluation, and runtime
   analyses. The CAFOAD configuration uses a 64-dimensional representation,
   four attention heads, and two causal Transformer layers.

3. multiclass_grouped_cv_runner.py
   Entry point for the leakage-audited, class-complete grouped five-fold
   multiclass evaluation. Each retained event block is used as test data
   exactly once, and preprocessing and model selection remain fold local.

4. igcps_multiclass_online_runner.py
   Implements the manuscript's validation-calibrated three-signal guarded
   prequential adaptation on the six-class IGCPS stream. It predicts and logs
   each batch before any optional update, and writes the protocol, batch,
   classwise, and contamination audits.

5. run_grouped_decision_fusion.py
   Entry point for the grouped five-fold decision-fusion comparison: naive
   Bayes (NB), behavior knowledge space (BKS), majority voting, and weighted
   majority voting. It can append the TV-DBN and CAFOAD reference rows when the
   main-run and efficiency result CSV files are supplied.

Required auxiliary code
-----------------------

multiclass_eventwise_runner.py
   Provides the shared dataset mapping, original-label conversion,
   causal-window loader, and class-weighted training support imported by
   multiclass_grouped_cv_runner.py. Keep it in the same directory as the
   grouped five-fold runner.

compute_efficiency_metrics.py
   Reconstructs the fold-0 train-only feature schema and generates the matched
   batch-one model FLOPs and latency audit. Its output
   TMC_multiclass_results/efficiency_metrics.csv can be passed to
   run_grouped_decision_fusion.py through --efficiency when producing the
   combined decision-fusion table.

Two public fused datasets
-------------------------

1. wdt_final_unified.csv
   Final causally fused Water Distribution Testbed (WDT) data used in the
   revised experiments.

2. ics_flow_final_unified.csv
   Final causally fused ICS-Flow data used in the revised experiments.

IGCPS availability
------------------

The fused IGCPS file is not publicly uploaded because the underlying IGCPS
records are proprietary assets of Shenzhen Gas Corporation and require
corporate approval for release. Researchers with authorized access can use
combinenew.py and the same evaluation runners to reproduce the IGCPS analysis.

Path configuration
------------------

The uploaded code preserves the paths used for the reported experiments.
Before running it on another computer, configure the following mappings:

1. In CAFOAD_major_revision.py, set DEFAULT_DATASETS to the locations of the
   fused CSV files. Its current defaults use the causal_grid_fusion subfolder.

2. In multiclass_eventwise_runner.py, set DATASETS to the same fused CSV
   locations. multiclass_grouped_cv_runner.py and
   igcps_multiclass_online_runner.py import this mapping.

3. In run_grouped_decision_fusion.py, set DEFAULT_DATASETS to the local fused
   CSV locations. The uploaded file retains the absolute paths of the original
   experiment computer.

4. If run_grouped_decision_fusion.py is used to create the combined table with
   TV-DBN and CAFOAD, pass the main grouped result file through --overall and
   the model-efficiency file through --efficiency. Without those two reference
   files, use the script functions for the four decision-fusion methods or
   generate the required result files first.

The two public CSV files may either be copied into the path expected by the
scripts or referenced by updating the mappings above. No change to the model,
fusion rule, fold construction, or training protocol is required.

Example commands
----------------

After configuring the paths, run the grouped multiclass experiment on the two
included public datasets:

  python multiclass_grouped_cv_runner.py --output grouped_cv_results --datasets WDT,ICS-Flow

Run the grouped decision-fusion comparison on the public datasets:

  python run_grouped_decision_fusion.py --output decision_fusion_results --datasets WDT,ICS-Flow --overall PATH_TO_OVERALL_RESULTS.csv --efficiency PATH_TO_EFFICIENCY_METRICS.csv

To run the complete three-dataset protocol, place an authorized
igcps_final_unified.csv at the configured IGCPS path and include IGCPS in the
--datasets argument.

With authorized IGCPS data configured in multiclass_eventwise_runner.py, run
the six-class guarded online audit:

  python igcps_multiclass_online_runner.py --output online_audit_results --seed 13 --epochs 30

Core dependencies
-----------------

Python 3, NumPy, pandas, SciPy, scikit-learn, PyTorch, and psutil. Reproducing
the model-efficiency audit additionally requires THOP.

Reproducibility scope
---------------------

The public CSV files are the final causally fused WDT and ICS-Flow files used
in the revised experiments. The grouped evaluation preserves original
multiclass labels, constructs homogeneous causal event blocks, applies purge
gaps between adjacent blocks, and fits preprocessing and feature selection
using training data only. The IGCPS access restriction is a data-licensing
constraint and is not replaced by synthetic or interpolated public data.
