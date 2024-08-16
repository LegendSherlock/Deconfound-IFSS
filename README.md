# Deconfound Semantic Shift and Incompleteness in Incremental Few-shot Semantic Segmentation
This repository contains the code for the paper:
<br>
**Deconfound Semantic Shift and Incompleteness in Incremental Few-shot Semantic Segmentation**<br>
Anonymous authors
<br>
AAAI 2025

<p align='center'>
  <img src='images/Fig4.png' width="800px">
</p>

### Abstract

Incremental few-shot semantic segmentation (IFSS) expands segmentation capacity of the trained model to segment new-class images with few samples. However, semantic meanings may shift from background to object class or vice versa during incremental learning. Moreover, new-class samples often lack representative attribute features when the new class greatly differs from the pre-learned old class. In this paper, we propose a causal framework to discuss the cause of semantic shift and incompleteness in IFSS, and we deconfound the revealed causal effects from two aspects. First, we propose a Causal Intervention Module (CIM) to resist semantic shift. CIM progressively and adaptively updates prototypes of old class, and removes the confounder in an intervention manner. Second, a Prototype Refinement Module (PRM) is proposed to complete the missing semantics. In PRM, knowledge gained from the episode learning scheme assists in fusing features of new-class and old-class prototypes. Experiments on both PASCAL-VOC 2012 and ADE20k benchmarks demonstrate the outstanding performance of our method.

## Notes
2024/8/15. The usage will be updated soon.


## Acknowledgments

This code is based on the implementations of [**MiB**](https://github.com/fcdl94/MiB) and [**Prototype Completion**](https://github.com/zhangbq-research/Prototype_Completion_for_FSL).