## Knowledge Distillation for Multimodal Visuomotor Policies

A study of the effectiveness of knowledge distillation for compressing VLA‑like manipulator control models.

--------

## Overview

The paper investigates the use of knowledge distillation to compress a multimodal visuomotor policy that takes as input a scene image and the coordinates of a target point. The teacher is a ResNet-18 with an MLP branch for the target (~11M parameters); the students are compact CNNs (~30k parameters), both multimodal and unimodal.

Experiments show that pre‑training is critically important for a multimodal learner (MAE improves by 2.5 times), while a unimodal learner can approach a multimodal one through distillation. This opens up the possibility of deploying compact policies without explicitly passing the target as input.


--------

