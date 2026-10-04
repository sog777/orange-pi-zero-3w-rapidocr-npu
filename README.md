# Getting RapidOCR Running on the Orange Pi Zero 3W NPU

## Porting PP-OCRv6 to the Allwinner A733 / Vivante VIP9000

### Abstract

This project began with a simple goal: run RapidOCR on an Orange Pi Zero 3W using its neural processing unit.

Achieving reliable OCR required substantially more than converting an ONNX model. The work involved static model conversion, graph rewrites, quantization engineering, isolated arithmetic tests, simulator-versus-hardware comparisons, repeated execution tests, and full-frame accuracy improvements.

The resulting production system uses the Vivante VIP9000 for text detection and eligible recognition attempts. A canonical ONNX Runtime CPU recognizer checks every recognition crop and supplies the final text and confidence scores. CPU execution also handles wide lines, orientation classification, image processing, and post-processing.

Further experiments attempted to eliminate CPU recognition. Some reduced CPU consumption substantially, but they introduced accuracy regressions and increased latency. Those changes were rejected.

The final result is a practical, NPU-supported, CPU-verified RapidOCR implementation integrated into Visual AI. This paper describes the successful changes, rejected approaches, measured results, and remaining limitations.

## 1. Hardware and Toolchain

Our development board, named Luna, was an Orange Pi Zero 3W using:

- Allwinner A733 SoC.
- Two Cortex-A76 and six Cortex-A55 CPU cores.
- Vivante VIP9000 NPU.
- Debian 12 Bookworm, AArch64.
- Linux kernel `6.6.98-sun60iw2`.
- NPU device interface `/dev/vipcore`.

The conversion environment used ACUITY 6.30.22 and VivanteIDE 5.11.0. The board’s VIPLite runtime identified itself as `2.0.3.2-AW-2024-08-30`.

The conversion target was `VIP9000NANODI_PLUS_PID0X1000003B`.

The working conversion and execution path was:

**ONNX → ACUITY → quantization/export → NBG → VIPLite → physical NPU execution**

Compilation and simulation ran in an existing Docker environment on the development Mac. Physical validation ran on Luna.

Other A733 development work documents the same VIP9000, VIPLite, chip identifier, and device interface [1, 2]. This supports the general applicability of the approach, although it does not establish that our OCR results transfer unchanged to every A733 board or software image.

## 2. Why Installing RapidOCR Was Not Enough

RapidOCR can execute ONNX models through ONNX Runtime on ARM CPUs. Using the A733 NPU requires a separate conversion and execution path.

Several model structures either failed conversion or executed incorrectly after apparently successful compilation. A model could import, quantize, compile into an NBG, and execute without crashing, yet still produce incorrect OCR.

Successful compilation was therefore treated as an intermediate milestone. Our validation chain compared the original ONNX reference, converted floating-point graph, quantized/exported graph, Vivante simulator, physical VIP9000 execution, and finally CTC decoding and complete OCR output.

These comparisons helped distinguish preprocessing mistakes, graph-conversion errors, quantization losses, execution problems, and decoding errors.

## 3. OCR Pipeline and Production Architecture

The OCR pipeline consists of image preprocessing, detection, text-region extraction, orientation handling, recognition, CTC decoding, and reading order.

The production system divides this work as follows:

| Processing stage | Production execution |
|---|---|
| Text-detection neural network | NPU |
| Eligible fixed-width recognition attempts | NPU |
| Independent recognition checking | CPU |
| Final recognized text and confidence scores | Canonical CPU recognizer |
| Genuinely wide text lines | CPU recognition |
| Angle classification | CPU |
| Resizing, normalization, geometry, CTC decoding, and reading order | CPU |

A crucial detail is that CPU verification runs on every recognition crop, including crops where NPU and CPU agree. “CPU corrections” count disagreements corrected by that verification; they do not represent the total number of CPU recognition calls.

The system is accurately described as NPU-supported, CPU-verified OCR. It does not eliminate the CPU recognition workload.

We did not establish a general speed advantage over an equivalent CPU-only implementation using the complete final accuracy profile. The measurements establish working NPU execution and document reliability checks, accuracy improvements, and better overall results than the full-NPU alternatives we tested.

## 4. Converting the Detector

The detector was converted to a static input of `[1,3,736,736]`. Its validated tensor metadata included:

| Tensor | Shape | Type | Scale | Zero point |
|---|---|---|---:|---:|
| Input | `[1,3,736,736]` | UINT8 | 0.0078431377 | 127 |
| Output | `[1,1,736,736]` | UINT8 | 0.0039214897 | 0 |

Correct preprocessing was essential. Channel ordering, normalization, tensor layout, scale, and zero point all had to match the compiled model.

An early physical detector benchmark measured approximately **27.7 ms per inference**. Repeated execution testing included a 100-run pass.

This measurement describes detector inference under the tested configuration. It excludes the rest of the OCR pipeline and should not be interpreted as full-frame OCR latency.

## 5. Converting the Recognizer

The original recognizer supported dynamic width, `[N,3,48,W]`. The deployed NPU recognizer uses an input of `[1,3,48,320]` and output of `[1,40,18710]`.

The output contains 40 CTC time steps over an 18,710-class vocabulary.

Before debugging physical NPU execution, we checked the converted floating-point model against the original ONNX reference. One conversion-stage comparison matched all **40/40 argmax positions**, with very small numerical differences.

This demonstrated that the import stage could preserve the reference prediction. It did not establish correctness after quantization and hardware execution.

## 6. Graph Rewrites

Several targeted graph rewrites were required.

### Stride-2 convolutions

Problematic convolution structures were rewritten as **stride-1 convolution → 1×1 MaxPool with stride 2**.

The 1×1 pooling operation performs spatial sampling. It is not conventional 2×2 max pooling. Padding and output geometry must be preserved carefully.

### Projection convolutions

Some projection operations were expressed through **Reshape → Multiply → ReduceSum → Reshape**. This avoided problematic convolution implementations while reproducing the intended projection.

Investigated projections included structures corresponding to 96→24, 192→48, and 48→192 channels. Replacing problematic convolutions with FullyConnected operations did not reliably solve the issue.

### Conv40 output splitting

A later projection with weights shaped `[384,768,1,1]` was split into two 192-output-channel convolutions. The research graph also used a Pad/Add reconstruction variant.

These changes addressed a specific conversion/execution problem; they are not a general recommendation to split every large convolution.

### LayerNorm rewrites

LayerNorm required several changes, including replacing `Pow(x,2)` with `Mul(x,x)`, rewriting targeted affine/subtraction structures, and replacing normalization Divide operations with reciprocal-square-root multiplication.

The general lesson was that mathematically equivalent graphs could behave very differently in the Vivante execution path.

## 7. The LayerNorm Divide Breakthrough

A companion investigation isolated a serious first-LayerNorm arithmetic problem.

The original normalization used **centered / sqrt(variance + epsilon)**. The replacement used **centered × Pow(variance + epsilon, −0.5)**.

The companion investigation reported that:

- Original node213 `VSI_NN_OP_DIVIDE` produced incorrect numerical results.
- Replacement node212 `VSI_NN_OP_POW` matched expected quantized arithmetic exactly.
- Replacement node213 `VSI_NN_OP_MULTIPLY` also matched expected quantized arithmetic exactly.
- Both replacement checks reported 100% exact agreement and maximum error zero.

The merged graph and generated artifacts were inspected. The supplied findings record these isolation results, but the original isolation dumps and checker scripts were not all included in that review. Those particular tests are therefore attributed to the companion investigation rather than described as independently repeated.

The production recognizer ultimately used the Divide bypass in all five LayerNorm blocks.

### Remaining precision sensitivity

The companion investigation also found saturation around node211 Add.159: seven of 40 values clipped in its tested UINT8 configuration.

Changing that generated-C output to FLOAT16 improved the decoded sequence and recovered expected trailing tokens **3795, 2737, 4645**. It did not recover the complete expected sequence.

This FLOAT16 modification was a generated-C experiment, not a change already embedded in the cumulative ONNX model or deployed production recognizer.

The working signed-INT8 graph still contains a quantized node211 output. Signed INT8 alone does not expand its real-valued range, and we did not establish the same seven-value clipping count on the final production model.

## 8. Quantization Was Part of Model Engineering

Quantization was treated as part of the model, rather than a final optimization switch.

The deployed recognizer metadata specifies:

| Property | Production value |
|---|---|
| Input shape | `[1,3,48,320]` |
| Input type | Signed INT8 |
| Input scale | 0.0078431377 |
| Input zero point | −1 |
| Input calibration range | Approximately −1 to 1 |
| CNN weight quantization | Per-channel INT8 |
| Output type | FLOAT16 |

Earlier UINT8 configurations used different scales and zero points. Those values should not be used as instructions for the current production model.

Input-range correction, graph repairs, and per-channel quantization contributed to improving the unchecked 18-line page result from **9/18 to 14/18**. Quantization changes were evaluated against reference inference because they could affect both arithmetic and final recognition.

## 9. The Large Output Classifier

The recognizer includes a large classification operation:

`[40,120] × [120,18710] → [40,18710]`

One experiment split the 18,710 output classes into eighteen chunks of 1,024 classes and one final chunk of 278 classes. The pieces were reconstructed into the original output.

CPU comparison preserved all 40 argmax positions with effectively negligible numerical differences. Nevertheless, particular Vivante executions still produced incorrect results in some chunks. Reducing a problematic chunk to 256 outputs did not automatically repair it.

This showed why CPU algebraic equivalence and successful compilation must both be followed by physical execution checks.

## 10. Physical Validation and Benchmark Scope

A validated reference recognizer output matched the Vivante simulator byte-for-byte. The compared FLOAT16 tensor contained **1,496,800 bytes**, corresponding to `1 × 40 × 18710 × 2 bytes`.

This was parity for a particular model configuration and fixture. It was not proof of parity across arbitrary images or proof that simulator output always matches canonical CPU recognition.

The reference recognizer also passed 100 consecutive runs with the expected CTC sequence and text: **绿洲仕格维花园公寓**.

The saved repeated-run benchmark reported a median recognizer call of **13.36 ms**, with a median NPU execution portion of **11.76 ms**.

An earlier combined OCR sample measured approximately **120.3 ms**. That result predates CPU verification and must not be presented as current verified-path latency.

Five later CPU-verified reference-line runs took approximately **195–204 ms**, excluding engine construction.

For the 18-line page, five CPU-verified runs took approximately **1.64–1.66 seconds**. Each run produced:

- 18 NPU recognition attempts.
- 18 CPU recognition checks.
- 14 agreements.
- Four CPU corrections.
- Final text, scores, and boxes matching the canonical CPU reference.

CPU-reference agreement is a reproducibility result. It does not establish perfect transcription of the visible image.

## 11. Improving Real Full-Frame OCR Accuracy

The original detector configuration missed substantial text in high-resolution camera frames, particularly small print on paper.

We subsequently developed and deployed an accuracy profile using higher-resolution detection preprocessing, overlapping detector tiles, blending of tile detection maps, recognition from the original-resolution frame, PAPER-specific missing-line recovery, targeted recognition retries, and improved geometric reading order.

For large frames, detection used a resized whole frame with a long side of **2,080 pixels**, **736-pixel tiles**, and a **672-pixel stride**.

The complete camera frame remained available. Internal detector tiles and recognized text-region crops were OCR processing steps; they did not remove part of the full frame supplied to the application's separate image-answering path.

Additional PAPER recovery reused cached detector maps, avoiding unnecessary repeated NPU detection calls. Candidate regions and recognition retries were constrained using geometry, confidence, and consistency checks. Confidence was treated as supporting evidence, not proof that the recognized text was correct.

### Original six-frame results

The following results use reference-word recall and three-run median processing times:

| Frame | Original recall | Improved recall | Original time | Improved time |
|---|---:|---:|---:|---:|
| PAPER 1 | 32.79% | 97.81% | 1.81 s | 5.14 s |
| PAPER 2 | 19.23% | 99.04% | 1.00 s | 17.63 s |
| PAPER 3 | 9.54% | 100% | 0.84 s | 8.00 s |
| SCREEN 1 | 42.11% | 100% | 1.48 s | 3.27 s |
| SCREEN 2 | 37.70% | 100% | 1.44 s | 3.22 s |
| SCREEN 3 | 0% | 100% | 0.70 s | 3.61 s |

These improvements deliberately accepted additional processing time because accuracy was the priority during this phase.

The final PAPER 2 refinement improved recall from 98.56% to 99.04%, with median timing effectively unchanged at approximately 17.6 seconds. The other five original frames retained identical text, and the additional regression outputs remained unchanged relative to the preceding accuracy candidate.

Word recall does not measure every punctuation, capitalization, line-layout, or substitution error. A result of 100% word recall should not be described as perfect OCR.

## 12. Broader Regression Validation

The regression material included six original PAPER/SCREEN frames, 66 additional distinct frames after duplicate removal, and two small reference fixtures. The additional frames comprised eight PAPER and 58 SCREEN frames.

This produced **72 reference-scored frames** and **74 engine comparisons** when the small fixtures were included.

For the additional material, production weighted word recall was approximately **96.29% for PAPER** and **88.25% for SCREEN**.

These sessions included difficult captures and diagnostic variants. They were not a randomized benchmark, and multiple frames could represent related scenes. Consequently, the results support measured improvements on this material, not a universal claim of 98–100% recognition accuracy.

Validation retained extracted text, scores, boxes, timing, NPU calls, CPU checks, disagreements, and recovery outcomes.

## 13. Why Full-NPU Recognition Was Not Promoted

The next phase attempted to remove CPU recognition while preserving the improved production baseline.

A separate experimental implementation prevented hidden CPU recognition fallbacks. This mattered because an older “NPU” mode still used CPU recognition for genuinely wide lines.

Experiments included sliding recognition windows and CTC stitching, mixed-precision sequence processing, wider recognition models, confidence-gate adjustments, overlap and fusion changes, and NPU angle classifiers.

### Full-NPU comparison

Across the original six frames:

| Frame | Production recall | Selected full-NPU candidate |
|---|---:|---:|
| PAPER 1 | 97.81% | 95.63% |
| PAPER 2 | 99.04% | 98.56% |
| PAPER 3 | 100% | 98.36% |
| SCREEN 1 | 100% | 100% |
| SCREEN 2 | 100% | 100% |
| SCREEN 3 | 100% | 98.21% |

Across all 72 reference-scored frames, the candidate improved recall on two, regressed on 24, and tied on 46. Tied recall did not necessarily mean identical text.

Across 74 engine comparisons, only six complete text outputs matched production exactly. The diagnostic comparison retained 1,821 box-associated text, score, missing-region, or added-region differences.

### Resource trade-off

One complete comparison measured:

| Metric | Production | Full-NPU candidate |
|---|---:|---:|
| Aggregate processing time | 335.94 s | 602.48 s |
| Aggregate CPU time | 631.20 CPU-s | 141.10 CPU-s |
| NPU recognition calls | 1,206 | 5,208 |
| CPU verification calls | 1,958 | 0 |
| CPU wide-line recognition calls | 752 | 0 |

CPU time sums work across threads and can exceed elapsed time.

The full-NPU candidate reduced CPU consumption by approximately **77.65%**, but increased processing time by approximately **79.34%** and reduced accuracy.

A subsequent near-width fitting experiment reduced its aggregate time to approximately **585.51 seconds**. It remained approximately **74.29% slower** than production and introduced a recall regression. Neither configuration met the promotion criteria.

### Wider recognizer

A 640-wide recognizer preserved CPU reference predictions closely but achieved only **37/48 exact hardware crop results**, compared with **42/48** for the selected 320-wide experimental configuration. It was rejected before complete regression qualification.

### Angle classification

The best tested NPU classifier disagreed with CPU on **231 of 3,824 checks**, comprising normal and 180-degree-rotated crops. Its inference speed was promising, but the disagreement rate prevented promotion. Other tested classifier variants performed substantially worse. The CPU angle classifier was retained.

These results establish the practical limit of the implementations tested. They do not prove that future compiler fixes, different models, or different architectures cannot improve NPU coverage.

## 14. Reliability and Failure Recovery

Validation extended beyond successful inference. Tests covered repeated recognizer execution, repeated complete OCR on selected fixtures, injected NPU inference failures, invalid-image handling, engine close/recreation, recovery on subsequent valid requests, and actual VIPLite execution counters.

Production recovered from an injected recognition inference failure using canonical CPU recognition, preserving the expected final output.

The experimental full-NPU implementation surfaced the injected failure rather than silently falling back to CPU recognition, then recovered on a subsequent valid request.

These recovery tests do not mean startup can succeed without working NPU libraries, models, or driver access. Likewise, zero execution errors during a regression run do not establish recognition accuracy or universal reliability.

## 15. Integration into Visual AI

The OCR implementation was integrated into Visual AI through a persistent OCR worker and bridge. Keeping the OCR engine alive avoids rebuilding models between sequential frames. Calls remain sequential because detector preprocessing stores per-image geometry.

The integration retained the application's existing paper/screen routing and image-processing behavior. Deployment checks included OCR operation, camera capture, and bounded application startup.

Camera initialization required separate diagnostic and recovery work. Later bounded capture checks passed, but an earlier cold-boot check failed. These records should not be represented as proof that every possible cold-boot camera failure was eliminated.

The retained records also do not establish a complete, unrestricted interactive test of every application scenario. The OCR validation results and the broader application’s deployment status remain distinct.

## 16. Checkpoints and Research Records

Known-good checkpoints were preserved before risky graph and deployment changes. As the work progressed, later verified states superseded earlier development milestones.

Cleanup removed retired generated binaries, duplicate compiler outputs, and unnecessary copies while retaining important source graphs, rewrite scripts, quantization information, regression data, and validation reports.

The retained research material includes model provenance, conversion records, runtime and dependency versions, reference fixtures, numerical comparisons, OCR comparisons, and recovery findings. These records make it possible to distinguish final production choices from abandoned experiments and to interpret each measurement in its original context.

## 17. Recommended Method for Other Developers

The most useful development sequence was to establish a known-good CPU reference, confirm basic NPU execution, freeze dynamic dimensions where required, and validate preprocessing independently.

Floating-point equivalence should precede quantization experiments. Suspicious operations should be isolated and graph rewrites tested numerically. Intermediate tensors are as useful as final text when identifying where an error first appears.

Simulator checks should be followed by physical execution, repeated testing, and failure recovery. Complete OCR latency and CPU consumption should be measured separately from individual model inference. Hidden CPU fallbacks must be identified when evaluating an NPU-only claim.

Broader visible-text regression material is essential before promoting a change. Preserve the production baseline while experimenting, and promote changes only when their overall accuracy, reliability, and performance justify them.

Treat the compiler as a target architecture with specific constraints. A successful ONNX conversion does not establish correct execution.

## 18. Performance and Validation Summary

| Measurement | Result and scope |
|---|---|
| Detector input | `[1,3,736,736]` |
| Early detector inference | Approximately 27.7 ms |
| Recognizer input | `[1,3,48,320]` |
| Recognizer output | `[1,40,18710]` |
| Reference recognizer repeated test | 100/100 correct for the reference fixture |
| Median reference recognizer call | 13.36 ms |
| Median NPU portion | 11.76 ms |
| Simulator/physical comparison | Byte-exact for validated reference output |
| Early unchecked combined sample | Approximately 120.3 ms |
| CPU-verified reference line | Approximately 195–204 ms |
| CPU-verified 18-line page | Approximately 1.64–1.66 s |
| Final page agreement with CPU reference | 18/18 |
| Page NPU/CPU agreements | 14/18 |
| Page CPU corrections | 4/18 |
| Original six-frame improved recall | 97.81–100% |
| Additional PAPER weighted recall | 96.29% |
| Additional SCREEN weighted recall | 88.25% |
| Full-NPU migration | Rejected because of accuracy and latency regressions |

These measurements describe specific models, fixtures, preprocessing profiles, and execution paths. They are not universal Orange Pi benchmarks.

## Conclusion

RapidOCR can use the Orange Pi Zero 3W’s A733/VIP9000 NPU successfully, but reliable PP-OCRv6 execution required substantial graph, quantization, runtime, and validation work.

Important breakthroughs included targeted convolution rewrites, LayerNorm Divide bypasses, per-channel INT8 quantization, physical-versus-simulator parity, CPU verification, and higher-resolution full-frame detection.

The deployed result remains hybrid. Detection and eligible recognition attempts run on the NPU, while canonical CPU recognition protects final accuracy.

Attempts to eliminate CPU recognition reduced CPU consumption but failed the combined accuracy and latency requirements. Those changes were not promoted.

The project demonstrates a practical Visual AI OCR implementation and a documented engineering method. Its results are bounded by the tested models, fixtures, and software stack; they do not establish perfect recognition or universal hardware correctness.

## References and Supporting Records

1. [A733 NPU driver project](https://github.com/petayyyy/a733_npu_driver/blob/main/README.md): platform, VIPLite interface, and conversion workflow.
2. [A733 deployment notes](https://github.com/petayyyy/a733_npu_driver/blob/main/docs/07-porting-radxa-to-orangepi.md): Radxa-to-Orange-Pi porting.

Project evidence retained with the implementation includes:

- `selftest_result.json`
- `verified_validation.json`
- `pipeline_validation.json`
- `verification/simulator_parity.json`
- `ASSISTANT_LAYERNORM_FINDINGS_2026-10-03.txt`
- `assistant-layernorm-comparison.md`
- `OCR_ACCURACY_DEPLOYMENT_REPORT.md`
- `FULL_NPU_EXPERIMENT_REPORT.md`
- `NPU_MIGRATION_FOLLOWUP.md`

The external references support platform and toolchain context. The OCR measurements and experiment conclusions come from the project’s own validation records, as summarized in the implementation review used to prepare this paper.
