# Self-Supervised Contrastive Learning for Individual and Collective Outlier Detection in Semantic Trajectories

This repository contains the extended code and results for the paper "Transferable Unsupervised Outlier Detection Framework for Human Semantic Trajectories" presented at ACM SIGSPATIAL 2024 and candidate for the Best Paper Award.

## Authors

- Xingzu Zhan, Shenzhen University, China
- Hossein Amiri, Emory University, USA
- Zheng Zhang, Emory University, USA
- Dazhou Yu, Emory University, USA
- Yuntong Hu, Emory University, USA
- Honghang Chen, Shenzhen University, China
- Liang Zhao, Emory University, USA
- Andreas Züfle, Emory University, USA

## Abstract

Semantic trajectories enrich spatiotemporal data with textual semantics such as trip purposes and activities at points of interest, enabling the discovery of outlier behaviors relevant to public health, social security, and urban planning. Existing detectors often depend on hand crafted rules and rarely fuse space, time, and text in a transferable way. We propose TOD4Traj, an unsupervised and transfer ready framework for human semantic trajectories that detects anomalies at both the individual and collective levels. TOD4Traj first unifies modalities through spatiotemporal to semantic contrastive alignment so heterogeneous signals share a common representation. It then learns regular mobility via temporal contrastive learning that aligns the same weekday and separates different weekdays. Deviations are scored by combining a consistency score against a user’s own history with a population score based on a small set of learned centroids that capture low rank mobility regularities. For group analysis, we introduce a population level contrastive objective and a collective outlier score that compare a date’s population embedding with historical weekday prototypes, revealing collective abnormality. Across simulated cities and a GeoLife derived dataset, TOD4Traj consistently outperforms strong baselines on both tasks and shows robust zero shot transfer across cities, demonstrating effectiveness, robustness, and adaptability for outlier detection in human semantic trajectories.

## Repository Structure

- `src/`: Contains the source code for TOD4Traj framework. split into `individual` and `collective` for respective outlier detection tasks.
- `results/`: Includes experimental results and evaluation metrics. Split into `individual` and `collective`.
