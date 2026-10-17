# RapidOCR on the Orange Pi Zero 3W NPU

# Public source supplement 16 October 2026

This edition excludes 312 vendor-generated C/header files with unverified
redistribution terms and both compiled NBG model files (detector.nb and
recognizer.nb). The proprietary vip_lite.h remains excluded. Obtain an authorized
SDK and generate required outputs locally; commands referencing excluded outputs
describe conditional build/validation steps, not an immediately runnable release.

Original camera/test images remain private. Commands referencing them are kept.
Numeric parity tensors are retained; they are not the original image files.
No target-board execution or fresh-install qualification was performed for this
publication revision. Production code and private backups were not touched.

The source manifest describes shipped readable files; excluded artifacts are
listed separately with their historical hashes. COMMAND_RECORD contains retained
instructions, comments and fragments, not a complete chronological execution log.
Some historical references and checksum evidence name artifacts no longer shipped.
PACKAGE_SHA256SUMS is the authoritative checksum inventory for this edition.

No blanket reuse license has been selected for project-authored code. Preserve
the included third-party license notices. Exclusion is not a legal clearance
opinion about the remaining package.


Complete engineering record and public source supplement

Allwinner A733 and Vivante VIP9000  |  Visual AI  |  5 October 2026

We made PP-OCRv6 detection and fixed-width recognition execute on the physical NPU, improved full-frame OCR accuracy, and integrated the verified path into Visual AI. Production keeps canonical CPU recognition because the evaluated NPU-only alternatives reduced accuracy and increased latency.

This manual includes the engineering chronology, public reproduction commands, complete listings of the core implementation, measurement results, failure findings and a source inventory. The companion archive contains the retained source and command supplement with the vendor-output exclusions described below. It is part of this deliverable, not a link to a private project folder.

Retained command records and reproduction recipes are distinct. A saved shell script or log command is evidence of a preserved instruction, not proof that every line ran in that order. A complete chronological terminal transcript was not retained. Commands that were not recorded cannot be supplied honestly as historical fact.

## Contents

Platform and dependencies

Chronology from initial conversion to final production

Graph transformation and precision findings

Runtime architecture and accuracy policy

Full frame accuracy development

Regression and performance results

NPU migration experiments and stopping decision

Visual AI integration and operating services

Camera recovery and backup scope

Public installation and validation recipes

Source and command supplement

Complete core source listings

## Platform and dependencies

| Component | Tested configuration |
| --- | --- |
| Board | Orange Pi Zero 3W |
| SoC and CPU | A733 with 2 Cortex A76 and 6 Cortex A55 |
| Accelerator | Vivante VIP9000 |
| OS | Orange Pi Debian 12 Bookworm AArch64 |
| Kernel | 6.6.98-sun60iw2 |
| Device | /dev/vipcore |
| ACUITY | 6.30.22 |
| VivanteIDE | 5.11.0 |
| VIPLite userspace | 2.0.3.2-AW-2024-08-30 |
| NPU driver | 2.0.3.4-AW-2025-10-27 |
| RapidOCR | 3.9.2 |
| ONNX Runtime for OCR | 1.30.0 |
| NumPy and OpenCV | 2.4.6 and 5.0.0.93 |
| Compiler target | VIP9000NANODI_PLUS_PID0X1000003B |

The working chain is ONNX to ACUITY to Vivante compilation to NBG to VIPLite to /dev/vipcore. A custom C bridge and Python session adapters connect RapidOCR to VIPLite. This is not a built-in ONNX Runtime Vivante execution provider and does not use RKNN.

The compiler ran in an existing x86-64 Docker environment; execution and final validation ran on the physical AArch64 board. The ARM64 shared library cannot execute directly on macOS. A Python dependency installation does not install the vendor kernel driver or userspace libraries.

## Chronology from initial conversion to final production

### Initial inspection

Inspected supplied files, toolchain, models and board access. Initial board access failures were resolved after power and SSH availability were restored. Simulator work and physical tests are different evidence sources.

### Detector execution

Converted a static 736 by 736 detector and validated tensor packing and the physical VIPLite path.

### Recognizer conversion

Fixed recognizer width at 320, kept the dynamic-width canonical CPU model and vocabulary as reference, and separated floating-point conversion from quantized execution failures.

### Graph surgery

Tested stride sampling, SE projections, Conv40 reconstruction, LayerNorm square and affine substitutions, and output-classifier splitting. Rejected successful compilations that produced wrong numerical results.

### LayerNorm investigation

Proved incorrect Divide arithmetic in the first tested LayerNorm; validated reciprocal-square-root Pow and Multiply against actual quantized inputs. Observed saturation and tested node211 FLOAT16.

### Physical recognizer milestone

Applied five LayerNorm bypasses and selected INT8 input range. Reference recognizer passed 100 runs; simulator and physical raw output matched on the fixture. Earlier page agreement increased from 9 to 14 of 18 lines.

### CPU checked production

Checked every recognition crop with canonical CPU probabilities. Final earlier page output matched all 18 canonical lines, scores and boxes on five repeats; injected NPU inference failures recovered.

### Visual AI deployment

Connected the OCR adapter and persistent worker, preserving answering, PAPER and SCREEN routing, knowledge, speech, camera behavior and answer combination.

### Resident answering service

Moved the CPU offline answering server into a lingering systemd user service. Prevented duplicate processes and per-launch model reloads; verified boot and lifecycle behavior.

### Full frame accuracy

Benchmarked six supplied camera frames, introduced tiled detection and full-resolution recognition, and refined PAPER 2 recovery. Validated 66 additional unique captures and deployed the improvement after user acceptance of imperfect cases.

### Strict NPU only work

Isolated an NPU-only recognizer with windowed wide lines and no CPU neural recognition fallback. Tested precision, overlaps, width fitting, larger width and angle classification against protected production.

### Stopping and preservation

Rejected migration that was slower or less accurate. Removed redundant/generated experimental artifacts while retaining unique research and results. Updated the same verified full project backup on board and Mac.

This is an engineering-stage chronology derived from retained records. It is not a fabricated timestamped transcript of every terminal interaction. The source archive preserves scripts, logs and measurement records with their own provenance.

## Graph transformation and precision findings

### Static shapes

| Tensor | Shape | Format and quantization |
| --- | --- | --- |
| Detector input | 1 x 3 x 736 x 736 | UINT8 scale 0.007843137718737125 zero point 127 |
| Detector output | 1 x 1 x 736 x 736 | UINT8 scale 0.00392148969694972 zero point 0 |
| Recognizer input | 1 x 3 x 48 x 320 | INT8 scale 0.007843137718737125 zero point -1 |
| Recognizer output | 1 x 40 x 18710 | FLOAT16 |

The recognizer source accepts variable width. The compiled model uses 320 pixels. Historical UINT8 recognizer scale approximately 0.0062129949 and zero point 161 describe an earlier experiment, not final production.

### Stride sampling

Two recognizer stride-2 convolutions became stride-1 convolutions followed by 1 by 1 MaxPool at stride 2. Kernel 1 pooling performs sampling rather than neighborhood maximum selection. The rewrite asserts the expected replacement count.

### SE projections

Ten bias-free projections on spatially reduced [1,C,1,1] tensors became Reshape, broadcast Multiply, ReduceSum and Reshape. This is not a generic substitution for arbitrary spatial convolutions. Channel pairs include 96 and 24, 192 and 48, and 384 and 96.

### Conv40 reconstruction

A [384,768,1,1] projection was split into two output groups. The initial script concatenated the results; the cumulative graph later uses Pad and Add reconstruction. The initial split script alone is not the final transformation.

### LayerNorm square and affine

Five square Pow operations became Mul(x,x). Affine subtraction restructuring is preserved in cumulative ONNX and imported compiler graphs.

### Division bypass

The tested node213 Divide implementation was numerically wrong. Pow(variance+epsilon,-0.5) followed by Multiply replaced division. Each isolated replacement matched its expected quantized arithmetic exactly, with maximum error zero. This validates local arithmetic, not full-model CPU equivalence.

### Node211 saturation

Add.159 UINT8 output clipped 7 of 40 values at 255, with recorded scale 0.0015841168351471424 and zero point 0. A generated-C FLOAT16 patch improved CTC but did not solve all recognition errors. This patch was not automatically baked into the ONNX.

### All five LayerNorm blocks

The first bypass was present in the imported model. build_allbypass.sh changes four remaining named sqrt and real_div pairs in ACUITY JSON, quantizes per-channel INT8 and fixes the normalized input range to [-1,1].

### Large output classifier experiment

A [40,120] by [120,18710] multiplication was split into 18 groups of 1024 and one group of 278. CPU predictions stayed equivalent but physical results differed across chunks. Smaller 256-output chunks did not reliably fix the tested arithmetic. This approach was rejected.

### Precision boundaries

FLOAT16 sequence-tail, prefix, variance and input-range variants were probed. Some FLOAT16 stem to quantized layer boundaries failed export; all-FLOAT16 and other successful exports did not establish a better production model.

```
First LayerNorm experiment
Before: [5032,2241,7038,74,11935]
Node211 FLOAT16: [5032,55,2241,7038,11935,3795,2737,4645]
Expected: [11194,7920,2241,7038,11183,11935,3795,2737,4645]
```

The retained four ONNX rewrite scripts do not independently reproduce every cumulative edit. Some Pad/Add, affine and division changes were made in intermediate/generated graphs. The saved production compiler model, weights and quantization support re-export; a single fully automated original-ONNX-to-final transformation script was not retained.

## Runtime architecture and accuracy policy

The C bridge queries actual tensor properties, checks size and supported type, allocates/maps buffers, flushes input, runs VIPLite, invalidates output and copies results. Reference counting and cleanup release shared runtime resources. Python serializes native calls with a lock; the complete OCR engine must be used sequentially because preprocessing stores per-image geometry.

Quantization uses the serialized scale with FLOAT64 intermediate arithmetic before rounding and clipping. NCHW shape, RGB/BGR handling, padding and normalization must match the calibration metadata. Output dequantization follows output metadata.

Production computes canonical CPU recognition for every crop first. It attempts NPU recognition where supported, compares decoded CTC tokens, records agreement or correction, and always returns canonical CPU probabilities. Confidence agreement is not sufficient to skip verification. Wide-line counters are a subset of canonical CPU checks, not extra checks to add to them.

Widths 321 to 384 are fitted to 320 in the existing runtime. Genuinely wide/unsupported crops use CPU in the legacy npu-labelled mode. That label is therefore not proof of CPU-free recognition. The separate NPU-only experiment explicitly removes the CPU recognizer/checker and fallback.

An NPU inference exception after successful startup can recover through already-computed canonical output. CPU errors propagate. Model or VIPLite startup failures still prevent normal construction. Canonical CPU agreement establishes reference consistency, not correctness against visible ground truth or recovery of text never detected.

## Full frame accuracy development

A single 736-pixel detector view missed much small print on high-resolution camera frames. The deployed large-image profile uses a 2080-pixel full-frame canvas, 736-pixel tiles, step 672 and weighted edge blending. Recognition reads full-resolution source pixels. The original full frame supplied to OpenAI remains unchanged; internal detector tiles and text-line regions are standard OCR operations.

Large-image box threshold is 0.35, segmentation threshold 0.30, PAPER unclip 1.0 and SCREEN unclip 1.6. The first pass can add non-overlapping detections from an undilated mask. A long PAPER line of at least 50 characters below 0.90 confidence can trigger recovery. That pass reuses cached detector maps, disables dilation and raises candidate count to 20000.

Geometry-based missing-line trials require at least eight words, score at least 0.98 and agreeing word prefixes across at least two trials, without inserting reference text. Selected long low-confidence recognition regions retry heights 0.7, 0.6 and 0.5 with downward offset 0.1; acceptance requires confidence gain over 0.005. Reading-order correction uses deskewed geometry, not answer labels.

The final PAPER 2 offset refinement recovered maintain instead of maintaine. PAPER 1 tighter box experiments damaged recall and were rejected. No demonstrated 100-percent, no-time-increase solution was retained for PAPER 1 or PAPER 2. The accepted production improvement prioritizes accuracy and leaves remaining errors explicit.

## Regression and performance results

| Frame | Before recall | Deployed recall | Before and after seconds |
| --- | --- | --- | --- |
| paper_frame_1.jpg | 32.79% | 97.81% | 1.81 / 5.14 |
| paper_frame_2.jpg | 19.23% | 99.04% | 1.00 / 17.63 |
| paper_frame_3.jpg | 9.54% | 100.00% | 0.84 / 8.00 |
| screen_frame_1.jpg | 42.11% | 100.00% | 1.48 / 3.27 |
| screen_frame_2.jpg | 37.70% | 100.00% | 1.44 / 3.22 |
| screen_frame_3.jpg | 0.00% | 100.00% | 0.70 / 3.61 |

PAPER 2 progressed from 19.23 percent to about 69.23 percent, then 98.56 percent and finally 99.04 percent. Its remaining temperature-symbol error is 99.9 degrees F rendered as 99.9 apostrophe F, with punctuation omissions. The last three-run medians were 17.65 and 17.63 seconds, within measurement noise. PAPER 1 remains 97.81 percent recall, with lower ordered accuracy where answer rows reorder.

Word recall counts normalized reference-word occurrences and does not prove exact spelling, punctuation, line layout, absence of extra text or correct order. Ground truth preserves visible source typos. Ordered WER and strict character metrics were also retained. SCREEN scores cover question/options; full outputs retain headers and peripheral detections. Six tuning images do not establish universal accuracy.

| Additional group | Coverage | Weighted recall |
| --- | --- | --- |
| PAPER | 8 unique images | 96.29% |
| Diagnostic full SCREEN | 9 captures | 92.63% |
| Raw SCREEN | 50 unique captures | 86.93% |

The supplied sessions contained 75 eligible files and nine exact duplicates: 66 unique inputs, comprising eight PAPER, fifty raw SCREEN and eight supplied SCREEN sections. The set contains correlated burst frames and overlapping question content. Supplied sections were not newly cropped by validation. Some poor captures score zero and are not excluded to inflate averages.

Remaining failures include absent question blocks and first lines, gibberish, merged words, punctuation and abbreviation errors. The PAPER recovery trigger did not fire on the new PAPER pages: missing detections provide no low-confidence line and incorrect text can have a high score. CPU checking cannot repair these detector omissions.

The final PAPER 2 refinement left all 66 additional-frame text, scores, boxes and recognition counters unchanged. Earlier preservation and fault-injection checks retained small-reference behavior, source hashes and canonical output. Brightness/resizing/JPEG variations did not establish universal 98-percent accuracy.

| Fixture or measurement | Recorded result |
| --- | --- |
| Reference recognizer | 100 of 100 runs correct |
| Wrapper and bridge medians | 13.364 ms and 11.758 ms |
| Raw output parity | 1,496,800 bytes equal to simulator/vendor reference on fixture |
| Earlier checked page | 18 of 18 canonical lines; 14 agreements and 4 corrections |
| Checked page latency | Approximately 1.64 to 1.66 s excluding engine construction |
| Earlier small-line NPU path | Approximately 120.3 ms in earlier configuration |
| Checked small-line path | Approximately 195 to 204 ms |

The bridge measurement includes native transfer/cache operations, not an isolated accelerator kernel. Byte parity and repeated correctness apply to the saved fixture. The earlier 120.3 ms result is not the later CPU-checked or high-resolution accuracy profile. Accuracy improvements take longer because more text is detected and checked. A general accuracy-preserving speedup over an identical CPU-only pipeline was not established across the 74-input set.

## NPU migration experiments and stopping decision

| Measurement on 74 inputs | Production | Selected NPU only |
| --- | --- | --- |
| Elapsed seconds | 335.943 | 602.480 |
| Median seconds | 3.940 | 6.104 |
| Process CPU seconds | 631.197 | 141.098 |
| Detector NPU calls | 629 | 629 |
| Recognizer NPU calls | 1206 | 5208 |
| CPU checks | 1958 | 0 |
| Wide CPU calls | 752 | 0 |
| Peak process RSS KiB | 573664 | 392368 |

The selected FLOAT16-tail NPU-only recognizer uses 320-pixel windows, step 256, overlap 64 and context-based output selection on the eight-pixel output grid. It improved a frozen crop probe from 38 of 48 to 42 of 48 exact CTC matches. CPU work fell about 77.65 percent, but wall time increased about 79.34 percent. All CPU preprocessing and orientation work still remains; NPU-only refers to neural recognition, not an all-hardware OCR pipeline.

| Original frame | Production recall | NPU only recall |
| --- | --- | --- |
| PAPER 1 | 97.81% | 95.63% |
| PAPER 2 | 99.04% | 98.56% |
| PAPER 3 | 100% | 98.36% |
| SCREEN 1 | 100% | 100% |
| SCREEN 2 | 100% | 100% |
| SCREEN 3 | 100% | 98.21% |

Across 72 scored images, two improved, twenty-four regressed and forty-six had equal word recall. Across all 74, complete text matched on six inputs, boxes on twenty-seven and score lists on none. The complete comparison records 1821 associated text/score/missing/added differences. Additional PAPER recall was 96.62 percent versus 96.29 production; combined 58-image SCREEN recall was 87.85 versus 88.25 percent.

A near-width fitting probe improved 42 to 43 of 48 and reduced 76 to 63 calls. Its complete set still regressed a word and took 585.507 versus 335.943 seconds, about 74.29 percent slower. A 640-pixel model preserved CPU graph argmax with maximum error below 3.2e-6, but physical exactness fell to 37 of 48; it was rejected before a full-set run.

| Angle classifier | Normal errors | Upside down errors | Total decisions |
| --- | --- | --- | --- |
| FLOAT16 | 57 | 1057 | 3824 |
| INT8 stride rewrite | 48 | 183 | 3824 |
| FLOAT16 stride rewrite | 55 | 1047 | 3824 |

The fastest classifier candidate still disagreed in 231 of 3824 decisions. CPU classification remained. These are crop decisions, with 1912 normal and 1912 rotated crops, not 3824 independent images or every retry crop. Other overlap, squeezing, confidence and precision trials did not yield a safe improvement.

Stability checks included repeated fixture/subset outputs, injected inference failures, invalid-image recovery and closed-handle reconstruction. Normal 74-input runs had zero NPU execution errors. Reliability passes did not erase accuracy or speed failures. The repeat suites cover stated subsets, not a repeated exhaustive 74-image stress qualification.

No tested additional migration satisfied accuracy, reliability and speed together. Production was not changed by the NPU-only effort. This stopping decision does not prove a hardware ceiling or preclude future compiler/model improvements.

## Visual AI integration and operating services

The adapter selects the verified engine with accuracy wrapper by default. Detection-only alignment retains its original profile. Environment controls select auto, board or CPU backend and accuracy or baseline profile. Existing PAPER/SCREEN behavior, Offline/OpenAI answer combination, knowledge, camera and speech policies were preserved.

A persistent systemd user service loads the CPU answering model at boot with user lingering, restart always, a three-second restart delay and journal logging. The launcher checks health instead of starting a second server. Saved records contain 22 passing lifecycle/boot checks and a model-loaded monotonic timestamp before the verification SSH session. The normal health gate was about 0.16 seconds. The answering model remains CPU based.

Deployment checks included OCR, offline grammar response, knowledge retrieval, OpenAI authentication/synthetic request, Piper playback and camera resolution transitions. Normal startup reached the persistent OCR worker. The bounded startup contained no visible question, so it is not evidence of a completed interactive PAPER/SCREEN-to-hybrid-answer test.

## Camera recovery and backup scope

The WN camera 1bcf:0b31 enumerated at SuperSpeed 5000M but UVC probe controls failed with -19 and initialization with -5. Delayed targeted binding to uvcvideo recovered interfaces. A cold boot also left missing configuration after -71 and -110 USB errors. Recovery restored configuration 1 for the exact SuperSpeed device before binding, without forcing USB2 or changing hardware.

After observed cold-boot recovery, 300 of 300 1280 by 720 frames passed in about 18.46 seconds at 5000M. A timer checks every fifteen seconds, avoiding an already-bound camera. The root cause of initial USB failure was not fully isolated; this is a validated recovery of observed states, not proof of a universal kernel fix.

The verified full project backup was updated on 4 October 2026 on board and Mac. It contains 27368 regular files and 316 symlinks, 7100321549 bytes, SHA256 3bd16531975b618b45ae7266a68f6d72960b2660b5a382ddd43a23eae7edc3f6. It includes the application, runtime, models, environments, answering model/server, Piper assets, recovery scripts and installed offline service. It excludes OS image, external OpenAI key, Bluetooth pairings, external audio settings and system driver/packages. Board failure does not establish memory-card failure; image the original card before overwriting it.

The public source supplement is not a replacement for that restore archive. It uses generic names and omits private data and large duplicate environments. Historical experimental binaries removed during accepted cleanup must be rebuilt before running retired experiments.

## Public installation and validation recipes

These are public reproduction recipes using user@ocr-board and /home/user/rapidocr-npu. They are not a recovered verbatim terminal transcript. Match the user account and absolute paths consistently. Versioned vendor dependencies must already be installed. Adapted public scripts have not been run on new hardware.

### Inspect the platform

```
ssh user@ocr-board
uname -a
cat /etc/os-release
lscpu
ls -l /dev/vipcore
lsusb -t
ls -l /dev/video*
systemctl --user list-unit-files
loginctl show-user user -p Linger
```

### Re export the saved graph on the compiler host

```
cd /home/user/rapidocr-npu
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/runtime/compiler:/project" ubuntu-npu:v2.0.10.2 \
  sh /project/rebuild.sh
```

### Compile and verify the board bridge

```
cd /home/user/rapidocr-npu/runtime
# Authorized SDK headers are required; vip_lite.h is not distributed
export VIPLITE_INCLUDE_DIR=/path/to/authorized/viplite/include
cc -O2 -fPIC -shared -Wall -Wextra -Werror -I"$VIPLITE_INCLUDE_DIR" -Iinclude \
  vip_bridge.c -lNBGlinker -lVIPhal -o librapidocr_vip.so
../venv/bin/python selftest.py --runs 100
../venv/bin/python validate_verified.py
../venv/bin/python validate_pipeline.py
```

### Run the standalone verified fixture

```
cd /home/user/rapidocr-npu/runtime
sh run.sh samples/text_line.png
sh run.sh samples/test_page.jpg
```

### Enable the resident answering service

```
# Install the service file before reloading and enabling it
mkdir -p "$HOME/.config/systemd/user"
install -m 0644 /home/user/rapidocr-npu/visual_ai/systemd_user_services/visual-ai-offline-answering.service "$HOME/.config/systemd/user/visual-ai-offline-answering.service"
sudo loginctl enable-linger user
systemctl --user daemon-reload
systemctl --user enable --now visual-ai-offline-answering.service
systemctl --user is-enabled visual-ai-offline-answering.service
systemctl --user is-active visual-ai-offline-answering.service
curl --fail http://127.0.0.1:8080/health
journalctl --user -u visual-ai-offline-answering.service
```

### Launch Visual AI

```
sh /home/user/rapidocr-npu/visual_ai/run_visual_ai.sh
```

### Install camera recovery when required

```
sudo sh /home/user/rapidocr-npu/work/camera_superspeed/install-recovery.sh
systemctl is-enabled board-wn-camera-recovery.timer
systemctl is-active board-wn-camera-recovery.timer
lsusb -t
ls -l /dev/video*
```

### Verify and restore a separate private project backup

```
sha256sum VISUAL_AI_FULL_BACKUP.tar.gz
# Compare to its trusted saved checksum before extracting.
# Stop the application and resident server first.
systemctl --user stop visual-ai-offline-answering.service
# Extraction overwrites matching files; preserve current data first.
tar -xzf VISUAL_AI_FULL_BACKUP.tar.gz -C /home/user
```

Source scripts retain their own detailed compiler import/quantize/export commands. Docker /project and /work mounts must match the historical recipe. Re-export uses saved quantization; recalibration needs the dataset path in inputmeta.yml to be resolved. Starting the standalone runtime does not enable the application accuracy wrapper.

## Source and command supplement

The publication edition contains 3,143 readable source records and 2,523 retained command records. It excludes 312 vendor-generated source files, both compiled NBGs and the proprietary header. The current manifests describe the shipped files; historical command references may name excluded artifacts.

SOURCE_MANIFEST.json identifies each public path, original source hash, adapted public hash, byte count, line count and whether substitutions changed the file. MODEL_ARTIFACT_MANIFEST.json supplies unmodified model/reference hashes. SOURCE_INDEX.txt indexes every source. COMMAND_RECORD.json and COMMAND_RECORD.txt include complete saved shell scripts plus located command-like source/log lines. Original copyrighted notices remain in included vendor/upstream sources.

Names were adapted consistently for publication. Production files and the private full backup were not edited. The Word code listings wrap long lines for reading; executable text is in the archive. Python syntax checks do not establish runtime equivalence of adapted names, libraries, model paths or external dependencies.

Excluded from the public copy: virtual environments and generated build caches, Git internals, private knowledge/capture libraries, raw media, external secrets, OS contents and large historical duplicate binary weights. Retained generated compiler C, build recipes, reports, measurements, public-adapted application sources and essential production compiler/model files are included. Removed historical files are not reconstructed or presented as newly recovered.

### Coverage by source area

| Area | Retained readable files |
| --- | --- |
| Visual AI | 3 |
| inventory | 37 |
| runtime | 34 |
| visual_ai | 2563 |
| work | 818 |

### Evidence records

runtime/selftest_result.json and verification/simulator_parity.json

runtime/verified_validation.json and verified_page_result.json

work/ocr_shifted_candidate/deployment_accuracy_results.json

work/ocr_additional_validation/session_results.json and session_manifest.json

work/accuracy_results/full_npu_validation/full_comparison.json

work/accuracy_results/full_npu_validation/fitnear_comparison.json

work/accuracy_results/full_npu_validation/classifier_summary.json

inventory/resident-service-change-record.json and boot observation

work/camera_superspeed/cold-boot-capture-validation.json

### Publication and dependency rights

The original test images remain private and are not publicly distributed. Saved image-test commands require separately supplied fixtures. The numeric recognizer input and reference tensors are included.

The upstream RapidOCR/PaddleOCR OCR weights and ONNX conversions use Apache 2.0 with upstream attribution. The canonical recognizer, classifier and retained detector source match upstream catalog hashes; the detector export NBG matches the packaged detector byte-for-byte. The package includes licenses/APACHE-2.0.txt. MODEL_SOURCES_AND_LICENSES.md records source URLs, modifications and conditions. No blanket license is assigned to this mixed package.

The proprietary vip_lite.h has been excluded. Supply it from an authorized SDK; set VIPLITE_INCLUDE_DIR for install.sh. vip_lite_common.h and the cl_viv_vx_ext.h copies have explicit permissive grants with notices retained. ACUITY-generated C/header output has no retained grant and is marked LICENSE_UNVERIFIED; it is not confirmed cleared for public redistribution. Verify SDK output/template and NBG terms before publishing those files.

Model sources: https://github.com/RapidAI/RapidOCR/blob/main/README.md#models and https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml. License: https://github.com/RapidAI/RapidOCR/blob/main/LICENSE. The answering model is Qwen3-4B-Instruct-2507 (Apache 2.0); its GGUF weights are not included.

### Technical references

https://dl.radxa.com/cubie/a7a/docs/hw/datasheet/A733_Datasheet_V0.93.pdf

https://github.com/orangepi-xunlong/orangepi-build/blob/next/external/config/boards/orangepizero3w.conf

https://github.com/unnamedwild-ux/frigate_npu_vivante

These references support the hardware and related Vivante workflow. They do not independently validate this project's OCR measurements.

## Complete core source listings

These listings include complete files for the runtime, production accuracy package, primary rewrites and selected build/service recipes. Every other retained source, generated compiler file and complete command record is included in the companion archive, indexed by manifest. No source is represented by an ellipsis or substituted pseudocode.

## npu ocr py

runtime/npu_ocr.py

Complete file  |  238 source lines  |  Public SHA256 ed2b9b752c38df99456254a34b63276e899f2745ba7e024c668c1e8b05cc5949

```
"""RapidOCR on A733: VIPLite detector and recognizer, CPU angle classifier."""
from pathlib import Path
import argparse
import ctypes as C
import json
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'
_VIP_LOCK=threading.RLock()

class VIPNetwork:
    def __init__(self, binary, metadata):
        self.meta=json.loads(Path(metadata).read_text())
        self.input_meta=next(iter(self.meta['Inputs'].values()))
        self.output_meta=next(iter(self.meta['Outputs'].values()))
        self.lib=C.CDLL(str(ROOT/'librapidocr_vip.so'))
        self.lib.ocr_open.argtypes=[C.c_char_p];self.lib.ocr_open.restype=C.c_void_p
        self.lib.ocr_close.argtypes=[C.c_void_p]
        self.lib.ocr_error.restype=C.c_char_p
        for name in ('ocr_input_bytes','ocr_output_bytes'):
            getattr(self.lib,name).argtypes=[C.c_void_p];getattr(self.lib,name).restype=C.c_size_t
        self.lib.ocr_run.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
        self.lib.ocr_run.restype=C.c_int
        with _VIP_LOCK: self.handle=self.lib.ocr_open(str(Path(binary).resolve()).encode())
        if not self.handle: raise RuntimeError(self.lib.ocr_error().decode())
        self.input_bytes=self.lib.ocr_input_bytes(self.handle)
        self.output_bytes=self.lib.ocr_output_bytes(self.handle)
        self.elapsed=0.0

    @staticmethod
    def dtype(meta):
        name=meta.get('dtype',meta.get('quantize',{}).get('qtype'))
        return {'u8':np.uint8,'uint8':np.uint8,'i8':np.int8,'int8':np.int8,'float16':np.float16,'flo
at32':np.float32}[name]

    def encode(self,x):
        x=np.asarray(x,dtype=np.float32)
        if list(x.shape)!=self.input_meta['shape']:
            raise ValueError(f"NPU requires {self.input_meta['shape']}, received {list(x.shape)}")
        q=self.input_meta.get('quantize')
        dtype=self.dtype(self.input_meta)
        if q:
            lim=np.iinfo(dtype)
            # Use the serialized scale at full precision; float32 intermediates
            # can move values across half-step rounding boundaries.
            x=np.clip(np.rint(x.astype(np.float64)/q['scale']+q['zero_point']),lim.min,lim.max)
        return np.ascontiguousarray(x,dtype=dtype)

    def infer_raw(self,x):
        x=np.ascontiguousarray(x)
        if x.nbytes!=self.input_bytes: raise ValueError('NPU input byte count mismatch')
        y=np.empty(self.output_bytes//np.dtype(self.dtype(self.output_meta)).itemsize,dtype=self.dty
pe(self.output_meta))
        with _VIP_LOCK:
            if not self.handle: raise RuntimeError('NPU network has been closed')
            start=time.perf_counter()
            status=self.lib.ocr_run(self.handle,C.c_void_p(x.ctypes.data),x.nbytes,C.c_void_p(y.ctyp
es.data),y.nbytes)
            self.elapsed=time.perf_counter()-start
            if status: raise RuntimeError(self.lib.ocr_error().decode())
        return y

    def __call__(self,x):
        raw=self.infer_raw(self.encode(x))
        y=raw.astype(np.float32)
        q=self.output_meta.get('quantize')
        if q: y=(y-q['zero_point'])*q['scale']
        return y.reshape(self.output_meta['shape'])

    def close(self):
        with _VIP_LOCK:
            if getattr(self,'handle',None): self.lib.ocr_close(self.handle);self.handle=None


class VIPDetectorSession:
    def __init__(self,cfg):
        self.network=VIPNetwork(MODELS/'detector.nb',MODELS/'detector.json')
        self.valid_shape=(736,736)
        self.mean=np.asarray(cfg.get('mean',[0.5]*3),np.float32)
        self.std=np.asarray(cfg.get('std',[0.5]*3),np.float32)

    def preprocess(self,image):
        h,w=image.shape[:2]
        scale=736/max(h,w)
        nh,nw=max(1,round(h*scale)),max(1,round(w*scale))
        canvas=np.full((736,736,3),255,dtype=np.uint8)
        canvas[:nh,:nw]=cv2.resize(image,(nw,nh))
        self.valid_shape=(nh,nw)
        normalized=(canvas.astype(np.float32)/255-self.mean)/self.std
        return normalized.transpose(2,0,1)[None].astype(np.float32)

    def __call__(self,x):
        y=self.network(x)
        h,w=self.valid_shape
        return y[:,:,:h,:w]


class FullNPURecognizerSession:
    def __init__(self,cfg):
        self.network=VIPNetwork(MODELS/'recognizer.nb',MODELS/'recognizer.json')
        self.characters=(MODELS/'characters.txt').read_text(encoding='utf-8').splitlines()
        self.last_backend='npu'
        self.npu_runs=0
        self.cpu_runs=0
        self.cpu=None

    def have_key(self):return True
    def get_character_list(self):return self.characters

    def __call__(self,x):
        results=[]
        for image in np.asarray(x,np.float32):
            # Rotated crops can exceed 320 by a few pixels. Fit these to
            # the static NPU input; keep genuinely wide lines on CPU.
            if image.shape[0:2]==(3,48) and 320 < image.shape[2] <= 384:
                image=cv2.resize(image.transpose(1,2,0),(320,48)).transpose(2,0,1)
            if image.shape!=(3,48,320):
                if self.cpu is None:
                    
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
                    self.cpu=ort.InferenceSession(str(MODELS/'canonical_recognizer.onnx'),sess_optio
ns=opts,providers=['CPUExecutionProvider'])
                results.append(self.cpu.run(None,{self.cpu.get_inputs()[0].name:image[None]})[0])
                self.last_backend='cpu_wide_line';self.cpu_runs+=1
            else:
                results.append(self.network(image[None]))
                self.last_backend='npu';self.npu_runs+=1
        return np.concatenate(results,axis=0)

def ctc_tokens(y):
    seq=np.asarray(y).argmax(-1).ravel().tolist()
    return [v for i,v in enumerate(seq) if v and (i==0 or v!=seq[i-1])]

class VerifiedRecognizerSession(FullNPURecognizerSession):
    """Run NPU recognition and use canonical CPU probabilities for final output.

    Checking every crop catches confident NPU errors. This does not assume
    that the canonical CPU model is ground truth for arbitrary images.
    """
    def __init__(self,cfg):
        super().__init__(cfg)
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.checker=ort.InferenceSession(str(MODELS/'canonical_recognizer.onnx'),sess_options=opts,
providers=['CPUExecutionProvider'])
        self.checker_input=self.checker.get_inputs()[0].name
        self.cpu_verification_runs=0
        self.npu_disagreements=0
        self.npu_agreements=0
        self.npu_errors=0
        self.checks=[]

    def __call__(self,x):
        results=[]
        for image in np.asarray(x,np.float32):
            original=np.ascontiguousarray(image[None])
            # CPU errors propagate; an unchecked NPU result is never returned.
            canonical=self.checker.run(None,{self.checker_input:original})[0]
            self.cpu_verification_runs+=1
            expected=ctc_tokens(canonical)
            attempted=image.shape[0:2]==(3,48) and image.shape[2]<=384
            record={'cpu_tokens':expected,'npu_tokens':None,'status':'cpu_wide_line'}
            if attempted:
                try:
                    predicted=super().__call__(original)
                    actual=ctc_tokens(predicted)
                    same=actual==expected
                    self.npu_agreements+=int(same)
                    self.npu_disagreements+=int(not same)
                    record.update(npu_tokens=actual,status='agreement' if same else 
'cpu_correction')
                except (RuntimeError,ValueError) as exc:
                    self.npu_errors+=1
                    record.update(status='cpu_after_npu_error',error=str(exc))
            else:
                self.cpu_runs+=1
            self.checks.append(record)
            self.last_backend=record['status']
            # Canonical scores also preserve CPU confidence filtering exactly.
            results.append(canonical)
        return np.concatenate(results,axis=0)

def build_ocr(recognition_mode="verified"):
    if recognition_mode not in ("verified","npu"):raise ValueError("Unknown recognition mode")
    import rapidocr.ch_ppocr_det.main as detector_module
    import rapidocr.ch_ppocr_rec.main as recognizer_module
    from rapidocr import RapidOCR
    original_det=detector_module.get_engine;original_rec=recognizer_module.get_engine
    detector_module.get_engine=lambda _: VIPDetectorSession
    recognizer_module.get_engine=lambda _: (VerifiedRecognizerSession if 
recognition_mode=="verified" else FullNPURecognizerSession)
    try:
        engine=RapidOCR(params={
            'Global.model_root_dir':str(MODELS),
            'Cls.model_path':str(MODELS/'classifier.onnx'),
            'Rec.model_path':str(MODELS/'canonical_recognizer.onnx'),
            'Rec.rec_img_shape':[3,48,320],
            'Rec.rec_batch_num':1,
            'EngineConfig.onnxruntime.intra_op_num_threads':2,
            'EngineConfig.onnxruntime.inter_op_num_threads':1,
        })
    finally:
        detector_module.get_engine=original_det;recognizer_module.get_engine=original_rec
    detector=engine.text_det
    detector.get_preprocess=lambda _: detector.session.preprocess
    return engine

def close_ocr(engine):
    engine.text_rec.session.network.close()
    engine.text_det.session.network.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--recognition-mode',choices=('verified','npu'),default='verified')
    args=parser.parse_args()
    engine=build_ocr(args.recognition_mode)
    try:
        result=engine(args.image)
        output={'texts':list(result.txts) if result.txts is not None else [],
                'scores':[float(s) for s in result.scores] if result.scores is not None else [],
                'boxes':result.boxes.tolist() if result.boxes is not None else [],
                'backend':('VIPLite detector + NPU recognition checked/corrected by canonical CPU; 
CPU angle classifier' if args.recognition_mode=='verified' else 'VIPLite detector + VIPLite 
recognizer; CPU angle classifier'),
                'recognition_mode':args.recognition_mode,
                'recognizer_npu_runs':engine.text_rec.session.npu_runs,
                'recognizer_cpu_wide_line_runs':engine.text_rec.session.cpu_runs,
                'elapsed_seconds':float(result.elapse)}
        if args.recognition_mode=='verified':
            session=engine.text_rec.session
            output.update(cpu_verification_runs=session.cpu_verification_runs,
                          npu_agreements=session.npu_agreements,
                          npu_disagreements_corrected=session.npu_disagreements,
                          npu_errors_corrected=session.npu_errors,
                          recognition_checks=session.checks)
        text=json.dumps(output,ensure_ascii=False,indent=2)
        print(text)
        if args.output: args.output.write_text(text+'\n',encoding='utf-8')
    finally: close_ocr(engine)

if __name__=='__main__': main()
```

## vip bridge c

runtime/vip_bridge.c

Complete file  |  92 source lines  |  Public SHA256 add90b49f6b3afc85208c1fb2a9658b47ce7942440766e73bff94da3a03fc8bb

```
#include <vip_lite.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    vip_network network;
    vip_buffer input, output;
    void *input_ptr, *output_ptr;
    size_t input_bytes, output_bytes;
    int prepared;
} ocr_handle;
static unsigned references;
static char error_text[256];
const char *ocr_error(void) { return error_text; }
static int status_check(int status, const char *operation) {
    if (status) snprintf(error_text, sizeof(error_text), "%s failed: VIPLite status %d", operation, 
status);
    return status;
}
void ocr_close(ocr_handle *h) {
    if (!h) return;
    if (h->prepared) vip_finish_network(h->network);
    if (h->input_ptr) vip_unmap_buffer(h->input);
    if (h->output_ptr) vip_unmap_buffer(h->output);
    if (h->input) vip_destroy_buffer(h->input);
    if (h->output) vip_destroy_buffer(h->output);
    if (h->network) vip_destroy_network(h->network);
    free(h);
    if (references && --references == 0) vip_destroy();
}
static int make_buffer(ocr_handle *h, int output, vip_buffer *buffer, size_t *bytes) {
    vip_buffer_create_params_t p;
    memset(&p, 0, sizeof(p));
    int (*query)(vip_network, vip_uint32_t, vip_enum, void *) = output ? vip_query_output : 
vip_query_input;
#define QUERY(prop, value) if (status_check(query(h->network, 0, prop, value), "query buffer")) 
return -1
    QUERY(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p.num_of_dims);
    if (!p.num_of_dims || p.num_of_dims > 6) { snprintf(error_text,sizeof(error_text),"Invalid 
tensor dimensions"); return -1; }
    QUERY(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p.sizes);
    QUERY(VIP_BUFFER_PROP_DATA_FORMAT, &p.data_format);
    QUERY(VIP_BUFFER_PROP_QUANT_FORMAT, &p.quant_format);
    if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        QUERY(VIP_BUFFER_PROP_TF_SCALE, &p.quant_data.affine.scale);
        QUERY(VIP_BUFFER_PROP_TF_ZERO_POINT, &p.quant_data.affine.zeroPoint);
    } else if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        QUERY(VIP_BUFFER_PROP_FIXED_POINT_POS, &p.quant_data.dfp.fixed_point_pos);
    }
#undef QUERY
    size_t width;
    switch (p.data_format) {
        case VIP_BUFFER_FORMAT_UINT8: case VIP_BUFFER_FORMAT_INT8: width=1;break;
        case VIP_BUFFER_FORMAT_FP16: case VIP_BUFFER_FORMAT_INT16: case VIP_BUFFER_FORMAT_UINT16: 
width=2;break;
        case VIP_BUFFER_FORMAT_FP32: width=4;break;
        default: snprintf(error_text,sizeof(error_text),"Unsupported VIP tensor type 
%d",p.data_format);return -1;
    }
    *bytes=width;
    for (unsigned i=0;i<p.num_of_dims;i++) {
        if (!p.sizes[i] || *bytes > SIZE_MAX/p.sizes[i]) { 
snprintf(error_text,sizeof(error_text),"Invalid tensor size");return -1; }
        *bytes *= p.sizes[i];
    }
    return status_check(vip_create_buffer(&p,sizeof(p),buffer),"create buffer");
}
ocr_handle *ocr_open(const char *path) {
    if (!references && status_check(vip_init(),"init")) return NULL;
    references++;
    ocr_handle *h=calloc(1,sizeof(*h));
    if (!h) { if (--references==0) vip_destroy(); snprintf(error_text,sizeof(error_text),"Allocation
 failed");return NULL; }
    if (status_check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&h->network),"create 
network")) goto fail;
    uint32_t in_count=0,out_count=0;
    if (status_check(vip_query_network(h->network,VIP_NETWORK_PROP_INPUT_COUNT,&in_count),"query 
inputs") ||
        status_check(vip_query_network(h->network,VIP_NETWORK_PROP_OUTPUT_COUNT,&out_count),"query 
outputs")) goto fail;
    if (in_count!=1 || out_count!=1) { snprintf(error_text,sizeof(error_text),"Expected one input 
and one output");goto fail; }
    if (status_check(vip_prepare_network(h->network),"prepare network")) goto fail;
    h->prepared=1;
    if (make_buffer(h,0,&h->input,&h->input_bytes) || make_buffer(h,1,&h->output,&h->output_bytes)) 
goto fail;
    h->input_ptr=vip_map_buffer(h->input);h->output_ptr=vip_map_buffer(h->output);
    if (!h->input_ptr || !h->output_ptr) { snprintf(error_text,sizeof(error_text),"Buffer mapping 
failed");goto fail; }
    if (status_check(vip_set_input(h->network,0,h->input),"set input") || 
status_check(vip_set_output(h->network,0,h->output),"set output")) goto fail;
    return h;
fail:
    ocr_close(h);return NULL;
}
size_t ocr_input_bytes(ocr_handle *h) { return h->input_bytes; }
size_t ocr_output_bytes(ocr_handle *h) { return h->output_bytes; }
int ocr_run(ocr_handle *h,const void *input,size_t input_bytes,void *output,size_t output_bytes) {
    if (!h || input_bytes!=h->input_bytes || output_bytes!=h->output_bytes) { 
snprintf(error_text,sizeof(error_text),"Tensor byte count mismatch");return -1; }
    memcpy(h->input_ptr,input,input_bytes);
    if (status_check(vip_flush_buffer(h->input,VIP_BUFFER_OPER_TYPE_FLUSH),"flush input") ||
        status_check(vip_run_network(h->network),"run network") ||
        status_check(vip_flush_buffer(h->output,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate 
output")) return -1;
    memcpy(output,h->output_ptr,output_bytes);return 0;
}
```

## install sh

runtime/install.sh

Complete file  |  15 source lines  |  Public SHA256 d71d83e3cf656bfc2841e9fef4ffc7c3c6bf303b4cc0250caf654aea25efcdad

```
#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${OCR_VENV:=../venv}"
python3 -m venv "$OCR_VENV"
"$OCR_VENV/bin/pip" install -r requirements.txt
# Supply vip_lite.h from an authorized SDK; its proprietary copy is not shipped.
: "${VIPLITE_INCLUDE_DIR:=include}"
[ -f "$VIPLITE_INCLUDE_DIR/vip_lite.h" ] || {
    echo 'Set VIPLITE_INCLUDE_DIR to your authorized VIPLite header directory.' >&2
    exit 1
}
cc -O2 -fPIC -shared -Wall -Wextra -Werror -I"$VIPLITE_INCLUDE_DIR" -Iinclude vip_bridge.c 
-lNBGlinker -lVIPhal -o librapidocr_vip.so
"$OCR_VENV/bin/python" selftest.py --runs 100
"$OCR_VENV/bin/python" validate_verified.py
```

## run sh

runtime/run.sh

Complete file  |  4 source lines  |  Public SHA256 daef00f22dfc85e9df1d4a64dd696bd2cc2b78439ba6e2b847bbcdaf2a6f2521

```
#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ../venv/bin/python npu_ocr.py "$@"
```

## rebuild sh

runtime/compiler/rebuild.sh

Complete file  |  4 source lines  |  Public SHA256 b41da4dc0e79e1789ee81936c742b4f976bdeae18f6224047c09eee6b18d2b9e

```
#!/bin/sh
# Run inside ubuntu-npu:v2.0.10.2 with this directory mounted at /project.
set -eu
python3 /root/acuity-toolkit-whl-6.30.22/bin/pegasus.py export ovxlib --model /project/model.json 
--model-data /project/model.data --model-quantize /project/model.quantize --with-input-meta 
/project/inputmeta.yml --output-path /project/rebuilt/rec_pc --optimize 
VIP9000NANODI_PLUS_PID0X1000003B --dtype quantized --save-fused-graph --pack-nbg-unify --viv-sdk 
/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools --build-platform make --target-ide-project linux64
```

## selftest py

runtime/selftest.py

Complete file  |  43 source lines  |  Public SHA256 fa3ba3259d72b9f52ab72fba51ff5b16d87651d08acf099562389b6bbc6ba63d

```
"""Physical NPU parity and recognizer stability check for the supplied fixture."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from npu_ocr import FullNPURecognizerSession, ROOT

EXPECTED=[11194,7920,2241,7038,11183,11935,3795,2737,4645]
def ctc(y):
    s=y.argmax(-1).ravel().tolist()
    return [v for i,v in enumerate(s) if v and (i==0 or v!=s[i-1])]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=int,default=100)
    args=parser.parse_args()
    if args.runs<1: parser.error('--runs must be positive')
    x=np.fromfile(ROOT/'samples/recognizer_input.f32',np.float32).reshape(1,3,48,320)
    session=FullNPURecognizerSession({})
    expected_bytes=(ROOT/'samples/recognizer_reference.dat').read_bytes()
    timings=[];npu=[]
    try:
        for i in range(args.runs):
            start=time.perf_counter();y=session(x);timings.append(time.perf_counter()-start)
            if ctc(y)!=EXPECTED:
                raise RuntimeError({'run':i,'actual':ctc(y),'expected':EXPECTED})
            npu.append(session.network.elapsed)
        actual_raw=session.network.infer_raw(session.network.encode(x))
        actual_bytes=actual_raw.tobytes(order='C')
        byte_exact=(actual_bytes==expected_bytes)
        if not byte_exact:
            raise RuntimeError('Persistent backend output bytes differ from vendor runner')
        text=''.join(session.characters[i-1] for i in EXPECTED)
        report={'status':'PASS','runs':args.runs,'ctc':EXPECTED,'text':text,
                'vendor_runner_byte_exact':byte_exact,'backend':'full NPU recognizer',
                'full_npu_recognizer_correct':True,
                'median_total_ms':float(np.median(timings)*1000),
                'median_npu_ms':float(np.median(npu)*1000)}
        (ROOT/'selftest_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2
)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:session.network.close()

if __name__=='__main__':main()
```

##   init   py

visual_ai/ocr_accuracy/__init__.py

Complete file  |  24 source lines  |  Public SHA256 60505f90b764a9788aef4ccbf350f794d2d2ab4b3693fe8c24eca961f4afbc97

```
"""Full-frame accuracy profile over the preserved verified the OCR board runtime."""
from pathlib import Path
from .automatic_candidate import configure_candidate, infer

class AccuracyOCR:
    def __init__(self, engine):
        self.engine = engine
        self.augmentation, self.gap = configure_candidate(engine)
        self.last_accuracy_details = {}

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def __call__(self, image, source_mode=None):
        # PAPER captures already use paper_section_* names in the assistant.
        # Explicit mode is available for callers with arbitrary image names.
        kind = source_mode.upper() if source_mode else (
            'PAPER' if isinstance(image, (str, Path)) and 
Path(image).name.lower().startswith('paper_') else 'SCREEN')
        if kind not in ('PAPER', 'SCREEN'):
            raise ValueError('source_mode must be PAPER or SCREEN')
        frame = self.engine.load_img(image)
        result, self.last_accuracy_details = infer(
            self.engine, frame, kind, self.augmentation, self.gap)
        return result
```

## augment detection py

visual_ai/ocr_accuracy/augment_detection.py

Complete file  |  26 source lines  |  Public SHA256 a97721f2281db72bd55839f23ec24ab9b9c0f3565f445fecfef005b4ef0300d0

```
"""Add non-overlapping detections from an undilated mask; retain baseline boxes."""
import cv2
import numpy as np
class AddMissingLines:
    def __init__(self,inner,overlap_limit=.35,extra_expansion=None):object.__setattr__(self,'inner',
inner);object.__setattr__(self,'enabled',False);object.__setattr__(self,'overlap_limit',overlap_limi
t);object.__setattr__(self,'extra_expansion',extra_expansion)
    def __getattr__(self,name):return getattr(self.inner,name)
    def __setattr__(self,name,value):
        if name=='enabled':object.__setattr__(self,name,value)
        else:setattr(self.inner,name,value)
    def __call__(self,pred,shape):
        boxes,scores=self.inner(pred,shape)
        if not self.enabled:return boxes,scores
        old=self.inner.dilation_kernel;old_expansion=self.inner.unclip_ratio
        try:
            self.inner.dilation_kernel=None
            if self.extra_expansion is not None:self.inner.unclip_ratio=self.extra_expansion
            extra,extra_scores=self.inner(pred,shape)
        finally:self.inner.dilation_kernel=old;self.inner.unclip_ratio=old_expansion
        kept=list(boxes);out=list(scores)
        for box,score in zip(extra,extra_scores):
            area=abs(cv2.contourArea(np.asarray(box,np.float32)))
            overlaps=[cv2.intersectConvexConvex(np.asarray(box,np.float32),np.asarray(b,np.float32))
[0]/max(area,1) for b in kept]
            if not overlaps or max(overlaps)<self.overlap_limit:kept.append(box);out.append(score)
        result=np.asarray(kept,dtype=boxes.dtype)
        self.inner.last_boxes=result.tolist();self.inner.last_scores=[float(v) for v in out]
        return result,out
```

## automatic candidate py

visual_ai/ocr_accuracy/automatic_candidate.py

Complete file  |  36 source lines  |  Public SHA256 2836a2c5c3079d2439a822e38d350760de5a12ec9b310c43358a5f8bc9ccbe07

```
"""Conditional accuracy fallback; full-frame inputs, production module unchanged."""
import cv2,numpy as np
from .candidate_tiled import configure
from .augment_detection import AddMissingLines
from .recognition_retry import configure_retry
from .gap_detection import GapDetection
from .reading_order import correct_order

def configure_candidate(engine):
    configure(engine,2080);session=engine.text_det.session;previous=type(session)
    class Cached(previous):
        def preprocess(self,image):self.source_frame=image;return super().preprocess(image)
        def __call__(self,x):
            if self.reuse_cached:return self.cached_prediction
            result=super().__call__(x);self.cached_prediction=result;return result
    session.__class__=Cached;session.reuse_cached=False
    augmentation=AddMissingLines(engine.text_det.postprocess_op,.35)
    gap=GapDetection(augmentation,engine);engine.text_det.postprocess_op=gap
    configure_retry(engine)
    return augmentation,gap

def infer(engine,image,kind,augmentation,gap):
    large=max(image.shape[:2])>2500;paper=kind=='PAPER' and large
    session=engine.text_det.session;session.tiled_enabled=large;session.reuse_cached=False
    engine.cfg.Global.use_preprocess_img=not large
    post=engine.text_det.postprocess_op;post.box_thresh=.35 if large else 
.5;post.thresh=.3;post.unclip_ratio=1.0 if paper else 
1.6;post.dilation_kernel=np.ones((2,2),np.uint8);post.max_candidates=1000
    augmentation.enabled=paper;gap.enabled=False;engine.accuracy_retry_enabled=False
    first=engine(image)
    triggered=paper and any(len(str(t))>=50 and float(s)<.90 for t,s in zip([] if first.txts is None
 else first.txts,[] if first.scores is None else first.scores))
    if not triggered:return first,dict(triggered=False)
    engine.gap_original_texts=list(first.txts);engine.gap_trials=[];engine.gap_added=[]
    session.reuse_cached=True;augmentation.enabled=False;post.dilation_kernel=None;post.max_candidat
es=20000;gap.enabled=True;engine.accuracy_retry_enabled=True
    try:result=engine(image)
    finally:session.reuse_cached=False
    result,order=correct_order(result)
    return result,dict(triggered=True,reading_order=order,first_texts=list(first.txts),gap_trials=en
gine.gap_trials,gap_added=engine.gap_added,recognition_retries=engine.accuracy_retries)
```

## candidate tiled py

visual_ai/ocr_accuracy/candidate_tiled.py

Complete file  |  36 source lines  |  Public SHA256 eb56fed9a0aeceb47fb5ad3b67e070659f84732e6ecd2fc29505f44ae849cf04

```
"""Opt-in higher-resolution NPU detection; production module remains unchanged."""
import types
import cv2
import numpy as np

def configure(engine, target=1408):
    session=engine.text_det.session
    baseline_preprocess=session.preprocess
    baseline_call=type(session).__call__
    session.tiled_enabled=False
    class Tiled(type(session)):
        def preprocess(self,image):
            if not self.tiled_enabled:return baseline_preprocess(image)
            h,w=image.shape[:2];scale=target/max(h,w)
            nh,nw=round(h*scale),round(w*scale)
            canvas=np.full((target,target,3),255,np.uint8)
            canvas[:nh,:nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
            self.tiled_shape=(nh,nw)
            positions=list(range(0,target-736+1,672))
            if positions[-1]!=target-736:positions.append(target-736)
            self.positions=[(y,x) for y in positions for x in positions if y<nh and x<nw]
            return 
np.stack([((canvas[y:y+736,x:x+736].astype(np.float32)/255-self.mean)/self.std).transpose(2,0,1) for
 y,x in self.positions])
        def __call__(self,x):
            if not self.tiled_enabled:return baseline_call(self,x)
            total=np.zeros((target,target),np.float32);weights=np.zeros_like(total)
            window=np.ones((736,736),np.float32)
            ramp=np.linspace(.05,1,32,dtype=np.float32)
            window[:32]*=ramp[:,None];window[-32:]*=ramp[::-1,None]
            window[:,:32]*=ramp[None,:];window[:,-32:]*=ramp[None,::-1]
            for crop,(y,z) in zip(x,self.positions):
                result=self.network(crop[None])[0,0]
                total[y:y+736,z:z+736]+=result*window
                weights[y:y+736,z:z+736]+=window
            h,w=self.tiled_shape
            return (total[:h,:w]/np.maximum(weights[:h,:w],1e-6))[None,None]
    session.__class__=Tiled
```

## gap detection py

visual_ai/ocr_accuracy/gap_detection.py

Complete file  |  45 source lines  |  Public SHA256 c56f9b303859bb3145722503a6463ebb4e4003aa0b00e65e9d106ceed20a8289

```
"""Image-geometry missing-line proposals, scored by verified OCR; no known text."""
import re,cv2,numpy as np
from rapidocr.utils.process_img import get_rotate_crop_image
class GapDetection:
    def __init__(self,inner,engine):
        object.__setattr__(self,'inner',inner);object.__setattr__(self,'engine',engine);object.__set
attr__(self,'enabled',False)
    def __getattr__(self,name):return getattr(self.inner,name)
    def __setattr__(self,name,value):
        if name=='enabled':object.__setattr__(self,name,value)
        else:setattr(self.inner,name,value)
    def __call__(self,pred,shape):
        boxes,scores=self.inner(pred,shape)
        if not self.enabled:return boxes,scores
        e=self.engine;image=e.text_det.session.source_frame;records=[];groups={}
        tokens=lambda t:re.findall(r'\w+',t.lower())
        old_prefixes={tuple(tokens(t)[:3]) for t in e.gap_original_texts if len(tokens(t))>=3}
        for index,box in enumerate(boxes):
            edge=box[1]-box[0];width=np.linalg.norm(edge);vertical=box[3]-box[0];h=np.linalg.norm(ve
rtical);center=box.mean(0)
            if width<500 or width/h<8:continue
            headers=[b for b in boxes if np.linalg.norm(b[1]-b[0])<200 and abs(b[0,0]-box[0,0])<200 
and center[1]-4*h<b.mean(0)[1]<center[1]-.5*h]
            if not headers:continue
            column_width=max(np.linalg.norm(b[1]-b[0]) for b in boxes if abs(b[0,0]-box[0,0])<200)
            header_left=min(b[0,0] for b in headers);header_bottom=max(b[:,1].max() for b in 
headers)
            u=edge/width;v=vertical/h
            for shift in [.7,1.,1.3]:
                for height_factor in [.7,1.,1.3]:
                    nh=h*height_factor;leftcenter=(box[0]+box[3])/2-v*h*shift
                    if leftcenter[1]<=header_bottom+.15*nh:continue
                    
leftcenter=leftcenter-u*max(0,(leftcenter[0]-header_left)/max(u[0],.1));origin=leftcenter-v*nh/2
                    proposal=np.asarray([origin,origin+u*column_width,origin+u*column_width+v*nh,ori
gin+v*nh],np.float32)
                    if proposal[:,0].min()<0 or proposal[:,1].min()<0 or 
proposal[:,0].max()>=shape[1] or proposal[:,1].max()>=shape[0]:continue
                    area=abs(cv2.contourArea(proposal));overlaps=[cv2.intersectConvexConvex(proposal
,np.asarray(b,np.float32))[0]/max(area,1) for b in boxes if np.linalg.norm(b[1]-b[0])>200]
                    if max(overlaps,default=0)>.25:continue
                    crop=get_rotate_crop_image(image,proposal);r=e.base_recognize([crop]);text=str(r
.txts[0]);score=float(r.scores[0]);words=tokens(text)
                    accepted=score>=.98 and len(words)>=8 and tuple(words[:3]) not in old_prefixes
                    record=dict(anchor=index,shift=shift,height_factor=height_factor,text=text,score
=score,accepted=accepted,box=proposal.tolist());records.append(record)
                    if accepted:groups.setdefault(tuple(words[:3]),[]).append(record)
        added=[];out=list(boxes);out_scores=list(scores)
        for items in groups.values():
            # Require two geometry trials to agree on the word prefix.
            if len(items)<2:continue
            best=max(items,key=lambda r:(len(tokens(r['text'])),r['score']))
            
out.append(np.asarray(best['box'],np.float32));out_scores.append(best['score']);added.append(best)
        e.gap_trials=records;e.gap_added=added
        return np.asarray(out,dtype=boxes.dtype),out_scores
```

## reading order py

visual_ai/ocr_accuracy/reading_order.py

Complete file  |  32 source lines  |  Public SHA256 cb3de18ab677a76d70cf5879b79d374b33a625899d94b9d07350178fc5a5fe33

```
"""Resolve multi-column short-text rows using deskewed geometry, no label parsing."""
import numpy as np

def correct_order(result):
    boxes=np.asarray(result.boxes);widths=np.linalg.norm(boxes[:,1]-boxes[:,0],axis=1);heights=np.li
nalg.norm(boxes[:,3]-boxes[:,0],axis=1)
    edges=boxes[:,1]-boxes[:,0];long=(widths>500)&(np.abs(edges[:,0])>1)
    slope=float(np.median(edges[long,1]/edges[long,0])) if np.any(long) else 0.
    centers=boxes.mean(1);y=centers[:,1]-slope*centers[:,0];order=list(range(len(boxes)));start=0
    while start<len(order):
        if not (widths[start]<600 and heights[start]<100):start+=1;continue
        end=start
        while end<len(order) and widths[end]<600 and heights[end]<100:end+=1
        
block=list(range(start,end));remaining=set(block);rows=[];span=.75*float(np.median(heights[block]))
        while remaining:
            best=[]
            for anchor in remaining:
                eligible=sorted([i for i in remaining if y[anchor]<=y[i]<=y[anchor]+span],key=lambda
 i:centers[i,0])
                group=[]
                for i in eligible:
                    if 
all(min(boxes[i,:,0].max(),boxes[j,:,0].max())-max(boxes[i,:,0].min(),boxes[j,:,0].min())<=0 for j 
in group):group.append(i)
                if len(group)>len(best):best=group
            if len(best)<3:break
            rows.append(best);remaining.difference_update(best)
        keys={i:(float(y[i]),float(centers[i,0])) for i in block}
        for row in rows:
            row_y=float(np.median(y[row]))
            for i in row:keys[i]=(row_y,float(centers[i,0]))
        if rows:order[start:end]=sorted(block,key=lambda i:keys[i])
        start=end
    if order!=list(range(len(boxes))):
        result.boxes=boxes[order];result.txts=tuple(result.txts[i] for i in 
order);result.scores=tuple(result.scores[i] for i in order)
    return result,order
```

## recognition retry py

visual_ai/ocr_accuracy/recognition_retry.py

Complete file  |  23 source lines  |  Public SHA256 87befcfc2826d1564951ea2168794604443eb01ab8318693ecc52ee4a5cb82e1

```
"""Experimental confidence-based retries of internal text regions; no frame crop."""
import types

def configure_retry(engine):
    original=engine.recognize_txt;engine.base_recognize=original;engine.accuracy_retry_enabled=False
;engine.accuracy_retries=[]
    def retry(self,crops):
        result=original(crops)
        if not self.accuracy_retry_enabled:return result
        texts=list(result.txts);scores=list(result.scores);self.accuracy_retries=[]
        for i,(text,score,crop) in enumerate(zip(texts,scores,crops)):
            h,w=crop.shape[:2]
            if score>=.985 or len(text)<50 or w/h<8:continue
            trials=[]
            for fraction in [.7,.6,.5]:
                
top=round(h*((1-fraction)/2+.1));bottom=min(h,round(h*((1+fraction)/2+.1)));region=crop[top:bottom]
                r=original([region]);trials.append(dict(fraction=fraction,vertical_offset_fraction=.
1,text=str(r.txts[0]),score=float(r.scores[0])))
            best=max(trials,key=lambda x:x['score'])
            accepted=best['score']>float(score)+.005
            if accepted:texts[i]=best['text'];scores[i]=best['score']
            self.accuracy_retries.append(dict(detection_index=i,original_text=str(text),original_sco
re=float(score),trials=trials,accepted=accepted))
        result.txts=tuple(texts);result.scores=tuple(scores)
        return result
    engine.recognize_txt=types.MethodType(retry,engine)
```

## rewrite allstride2 maxpool py

work/assistant_layernorm_research/rewrite_allstride2_maxpool.py

Complete file  |  32 source lines  |  Public SHA256 dcb9ccbb841b147a357ba1e24b9c513e750cb8a7057a0834b99e2b37d72a8cd3

```
import onnx
from onnx import helper
src="/work/PP-OCRv6_rec_small.onnx"
dst="/work/PP-OCRv6_rec_small_allstride2_maxpool.onnx"
m=onnx.load(src)
orig=list(m.graph.node)
hits=[]
for n in orig:
    if n.op_type=="Conv":
        a=next((x for x in n.attribute if x.name=="strides"),None)
        if a is not None and list(a.ints)==[2,2]:
            hits.append(n.name)
print("STRIDE2_CONVS =",hits)
assert len(hits)==2,("STOP_EXPECTED_2_STRIDE2_CONVS",hits)
del m.graph.node[:]
rewritten=[]
for n in orig:
    a=next((x for x in n.attribute if x.name=="strides"),None) if n.op_type=="Conv" else None
    if a is not None and list(a.ints)==[2,2]:
        old=n.output[0]
        full=old+"__stride1_full"
        n.output[0]=full
        a.ints[:]=[1,1]
        p=helper.make_node("MaxPool",[full],[old],name=n.name+"__stride2_maxpool",kernel_shape=[1,1]
,strides=[2,2],pads=[0,0,0,0],ceil_mode=0)
        m.graph.node.extend([n,p])
        rewritten.append(n.name)
    else:
        m.graph.node.append(n)
print("SKIP_OLD_DOCKER_CHECKER = model IR newer than bundled checker")
onnx.save(m,dst)
print("REWRITTEN =",rewritten)
print("SAVED =",dst)
```

## rewrite conv40 outsplit2 py

work/assistant_layernorm_research/rewrite_conv40_outsplit2.py

Complete file  |  74 source lines  |  Public SHA256 a861e8b2148bf4c16892d5f9e8ccda12746ddc03f1b8158b0598b8e326daeba4

```
import onnx
import numpy as np
import sys
from onnx import helper, numpy_helper

SRC = "/work/PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum.onnx"
DST = "/work/PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum_conv40_outsplit2.onnx"
TARGET = "p2o.pd_op.conv2d.40.0"
WEIGHT = "conv2d_122.w_0"

m = onnx.load(SRC)

idx = None
old = None
for i, n in enumerate(m.graph.node):
    if TARGET in n.output:
        idx = i
        old = n
        break

if old is None or old.op_type != "Conv":
    raise RuntimeError("target Conv40 not found")

if len(old.input) != 2 or old.input[1] != WEIGHT:
    raise RuntimeError("unexpected Conv40 inputs: %r" % (list(old.input),))

init = {x.name: x for x in m.graph.initializer}
w = numpy_helper.to_array(init[WEIGHT]).astype(np.float32)

if tuple(w.shape) != (384, 768, 1, 1):
    raise RuntimeError("unexpected Conv40 weight shape %r" % (w.shape,))

x = old.input[0]
wa = WEIGHT + "__out_a"
wb = WEIGHT + "__out_b"
ya = TARGET + "__out_a"
yb = TARGET + "__out_b"

m.graph.initializer.extend([
    numpy_helper.from_array(w[:192, :, :, :].copy(), wa),
    numpy_helper.from_array(w[192:, :, :, :].copy(), wb),
])

attrs = {a.name: helper.get_attribute_value(a) for a in old.attribute}

new_nodes = [
    helper.make_node("Conv", [x, wa], [ya], name="Conv40_outsplit_a", **attrs),
    helper.make_node("Conv", [x, wb], [yb], name="Conv40_outsplit_b", **attrs),
    helper.make_node("Concat", [ya, yb], [TARGET], name="Conv40_outsplit_concat", axis=1),
]

del m.graph.node[idx]
for j, n in enumerate(new_nodes):
    m.graph.node.insert(idx + j, n)

onnx.checker.check_model(m)
onnx.save(m, DST)

used = {i for n in m.graph.node for i in n.input}
ni = {x.name: numpy_helper.to_array(x) for x in m.graph.initializer}
old_present = any(n.op_type == "Conv" and TARGET in n.output for n in m.graph.node)
conv_count = sum(n.name in ("Conv40_outsplit_a", "Conv40_outsplit_b") and n.op_type == "Conv" for n 
in m.graph.node)
concat_count = sum(n.name == "Conv40_outsplit_concat" and n.op_type == "Concat" for n in 
m.graph.node)
shape_ok = tuple(ni[wa].shape) == (192, 768, 1, 1) and tuple(ni[wb].shape) == (192, 768, 1, 1)
ok = (not old_present and WEIGHT not in used and conv_count == 2 and concat_count == 1 and shape_ok)

print("OLD CONV40 PRESENT:", old_present)
print("ORIGINAL WEIGHT STILL USED:", WEIGHT in used)
print("OUTSPLIT CONVS:", conv_count)
print("OUTSPLIT CONCAT:", concat_count)
print("WEIGHT A:", tuple(ni[wa].shape))
print("WEIGHT B:", tuple(ni[wb].shape))
print("CONV40 OUTSPLIT2 REWRITE:", "PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
```

## rewrite conv5 mulsum py

work/assistant_layernorm_research/rewrite_conv5_mulsum.py

Complete file  |  47 source lines  |  Public SHA256 46786d3fe75d4166766df09a697b6f440bc214a68af6d03b4b436f2696e0880a

```
import onnx
import numpy as np
from onnx import helper,numpy_helper
src="/work/PP-OCRv6_rec_small_allstride2_maxpool.onnx"
dst="/work/PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum.onnx"
m=onnx.load(src)
orig=list(m.graph.node)
init={x.name:x for x in m.graph.initializer}
targets=[("p2o.pd_op.conv2d.5.0","Conv.6",(24,96,1,1)),("p2o.pd_op.conv2d.6.0","Conv.7",(96,24,1,1))
,("p2o.pd_op.conv2d.15.0","Conv.20",(48,192,1,1)),("p2o.pd_op.conv2d.16.0","Conv.21",(192,48,1,1)),(
"p2o.pd_op.conv2d.21.0","Conv.28",(48,192,1,1)),("p2o.pd_op.conv2d.22.0","Conv.29",(192,48,1,1)),("p
2o.pd_op.conv2d.27.0","Conv.36",(48,192,1,1)),("p2o.pd_op.conv2d.28.0","Conv.37",(192,48,1,1)),("p2o
.pd_op.conv2d.35.0","Conv.47",(96,384,1,1)),("p2o.pd_op.conv2d.36.0","Conv.48",(384,96,1,1))]
repl={}
for out,name,shape in targets:
    hits=[n for n in orig if n.op_type=="Conv" and out in n.output]
    print("TARGET =",out,[(n.name,list(n.input),list(n.output)) for n in hits])
    assert len(hits)==1,(out,[(n.name,list(n.output)) for n in hits])
    n=hits[0]
    assert n.name==name,(n.name,name)
    assert n.input[1] in init,n.input[1]
    w=numpy_helper.to_array(init[n.input[1]])
    print("WEIGHT_SHAPE =",n.name,w.shape)
    assert tuple(w.shape)==shape,(n.name,w.shape,shape)
    assert len(n.input)==2,(n.name,list(n.input))
    out_ch,in_ch=shape[0],shape[1]
    w3=np.ascontiguousarray(w[:,:,0,0].reshape(1,out_ch,in_ch))
    wname=n.input[1]+"__mulsum_1x%dx%d"%(out_ch,in_ch)
    sin=n.name+"__mulsum_shape_in"
    sout=n.name+"__mulsum_shape_out"
    m.graph.initializer.extend([
        numpy_helper.from_array(w3,name=wname),
        numpy_helper.from_array(np.array([1,1,in_ch],dtype=np.int64),name=sin),
        numpy_helper.from_array(np.array([1,out_ch,1,1],dtype=np.int64),name=sout)
    ])
    old=n.output[0]
    rin=old+"__mulsum_reshape_in"
    mul=old+"__mulsum_mul"
    red=old+"__mulsum_reduce"
    repl[n.name]=[
        helper.make_node("Reshape",[n.input[0],sin],[rin],name=n.name+"__mulsum_reshape_in"),
        helper.make_node("Mul",[rin,wname],[mul],name=n.name+"__mulsum_mul"),
        helper.make_node("ReduceSum",[mul],[red],name=n.name+"__mulsum_reduce",axes=[2],keepdims=0),
        helper.make_node("Reshape",[red,sout],[old],name=n.name+"__mulsum_reshape_out")
    ]
del m.graph.node[:]
for q in orig:
    m.graph.node.extend(repl.get(q.name,[q]))
onnx.save(m,dst)
print("REPLACED =",sorted(repl))
print("SAVED =",dst)
```

## rewrite layernorm mul py

work/assistant_layernorm_research/rewrite_layernorm_mul.py

Complete file  |  70 source lines  |  Public SHA256 fe9f137f08ed2de649fb85e9e6783451f0094ce28d0ddc95b054fc4320702f05

```
import onnx, numpy as np, sys
from onnx import numpy_helper

SRC="/work/PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum_conv40_outsplit2_padadd.onnx"
DST="/work/PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum_conv40_outsplit2_padadd_lnmul.onnx"
TARGETS=["Add.161","Add.173","Add.183","Add.195","Add.205"]

m=onnx.load(SRC)
nodes=list(m.graph.node)
prod={o:n for n in nodes for o in n.output}

const={}
for t in m.graph.initializer:
    const[t.name]=numpy_helper.to_array(t)
for n in nodes:
    if n.op_type=="Constant" and len(n.output)==1:
        for a in n.attribute:
            if a.name=="value":
                const[n.output[0]]=numpy_helper.to_array(a.t)

changed=[]
for target in TARGETS:
    if target not in prod:
        raise RuntimeError(f"missing target producer: {target}")
    final=prod[target]
    if final.op_type!="Add":
        raise RuntimeError(f"{target}: final op is {final.op_type}, expected Add")
    mul=prod.get(final.input[0])
    if mul is None or mul.op_type!="Mul":
        raise RuntimeError(f"{target}: scale Mul not found")
    div=prod.get(mul.input[0])
    if div is None or div.op_type!="Div":
        raise RuntimeError(f"{target}: Div not found")
    sqrt=prod.get(div.input[1])
    if sqrt is None or sqrt.op_type!="Sqrt":
        raise RuntimeError(f"{target}: Sqrt not found")
    add_eps=prod.get(sqrt.input[0])
    if add_eps is None or add_eps.op_type!="Add":
        raise RuntimeError(f"{target}: epsilon Add not found")
    rmean2=prod.get(add_eps.input[0])
    if rmean2 is None or rmean2.op_type!="ReduceMean":
        raise RuntimeError(f"{target}: variance ReduceMean not found")
    pw=prod.get(rmean2.input[0])
    if pw is None or pw.op_type!="Pow":
        raise RuntimeError(f"{target}: Pow not found")
    sub=prod.get(pw.input[0])
    if sub is None or sub.op_type!="Sub":
        raise RuntimeError(f"{target}: centered Sub not found")
    if len(pw.input)!=2 or pw.input[1] not in const:
        raise RuntimeError(f"{target}: Pow exponent constant not resolvable")
    exp=np.asarray(const[pw.input[1]]).astype(np.float64).reshape(-1)
    if exp.size!=1 or not np.isclose(exp[0],2.0,rtol=0,atol=0):
        raise RuntimeError(f"{target}: Pow exponent is {exp.tolist()}, expected exactly 2")
    old_name=pw.name
    x=pw.input[0]
    del pw.input[:]
    pw.input.extend([x,x])
    pw.op_type="Mul"
    del pw.attribute[:]
    changed.append((target,old_name,x))

if len(changed)!=5:
    raise RuntimeError(f"changed {len(changed)} patterns, expected 5")

onnx.checker.check_model(m)
onnx.save(m,DST)
print("LAYERNORM MUL BREAKER REPLACEMENTS:",len(changed))
for t,n,x in changed:
    print(t,"|",n,"Pow(x,2) -> Mul(x,x)","|",x)
print("LAYERNORM MUL BREAKER REWRITE: PASS")
```

## build allbypass sh

work/build_allbypass.sh

Complete file  |  24 source lines  |  Public SHA256 4f4c573f7e01f2878ab55869deb84d2a0692b2ed648e24b472969adef184fa8d

```
#!/bin/sh
set -eu
PEG=/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py
mkdir -p /project/allbypass_import
cp /project/perchannel_import/model.data /project/allbypass_import/model.data
python3 - <<'PY'
from pathlib import Path
import json,copy
p=Path('/project');m=json.loads((p/'perchannel_import/model.json').read_text());layers=m['Layers'];c
ount=0
for l in layers.values():
 if l['op']=='sqrt' and l['name'] in ('Sqrt.3','Sqrt.5','Sqrt.7','Sqrt.9'):
  l['op']='pow';l['inputs'].append('@ln1_rsqrt_exp_mhalf_229:out0');count+=1
 if l['op']=='real_div' and l['name'] in ('Div.29','Div.31','Div.33','Div.35'):
  l['op']='multiply';l['parameters']={'axis':1,'bias':False}
assert count==4
(p/'allbypass_import/model.json').write_text(json.dumps(m,indent=2))
PY
python3 "$PEG" quantize --model /project/allbypass_import/model.json --model-data 
/project/allbypass_import/model.data --model-quantize /project/allbypass.quantize --with-input-meta 
/work/rec_ln1rsqrtmul_quant/PP-OCRv6_rec_ln1rsqrtmul_inputmeta.yml --quantizer 
perchannel_symmetric_affine --qtype int8 --batch-size 1 --iterations 1 --device CPU --rebuild > 
/project/allbypass_quantize.log 2>&1
python3 - <<'PY'
from pathlib import Path
import yaml
p=Path('/project');q=yaml.safe_load((p/'allbypass.quantize').read_text());q['quantize_parameters']['
@x_423:out0'].update(min_value=-1.0,max_value=1.0,scale=2.0/255,zero_point=-1);(p/'allbypass.quantiz
e').write_text(yaml.safe_dump(q,sort_keys=False))
PY
python3 "$PEG" export ovxlib --model /project/allbypass_import/model.json --model-data 
/project/allbypass_import/model.data --model-quantize /project/allbypass.quantize --with-input-meta 
/work/rec_ln1rsqrtmul_quant/PP-OCRv6_rec_ln1rsqrtmul_inputmeta.yml --output-path 
/project/rec_allbypass_export/rec_allbypass --optimize VIP9000NANODI_PLUS_PID0X1000003B --dtype 
quantized --save-fused-graph --pack-nbg-unify --viv-sdk /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools 
--build-platform make --target-ide-project linux64 > /project/allbypass_export.log 2>&1
```

## build pctail sh

work/build_pctail.sh

Complete file  |  23 source lines  |  Public SHA256 65eb1fdefcc26bff1477e4fa9082c7f5e8d27eda857bc508154e929a4c7e90e1

```
#!/bin/sh
set -eu
python3 - <<'PYCODE'
from pathlib import Path
import yaml,json
p=Path('/project');q=yaml.safe_load((p/'allbypass.quantize').read_text());layers=json.loads((p/'allb
ypass_import/model.json').read_text())['Layers']
start=next(k for k,v in layers.items() if v['name']=='p2o.pd_op.flatten.0.0');selected={start}
while True:
 more={k for k,v in layers.items() if any(s.split(':')[0].lstrip('@') in selected for s in 
v['inputs'])}
 if more<=selected:break
 selected|=more
for k in list(selected):
 for s in layers[k]['inputs']:
  a=s.split(':')[0].lstrip('@')
  if layers[a]['op']=='variable':selected.add(a)
count=0
for name in q['quantize_parameters']:
 if name.split(':')[0].lstrip('@') in selected:
  q['quantize_parameters'][name]={'qtype':'float16','quantizer':'float16'};count+=1
assert count>50
(p/'pctail.quantize').write_text(yaml.safe_dump(q,sort_keys=False));print('FP16 entries',count)
PYCODE
python3 /root/acuity-toolkit-whl-6.30.22/bin/pegasus.py export ovxlib --model 
/project/allbypass_import/model.json --model-data /project/allbypass_import/model.data 
--model-quantize /project/pctail.quantize --with-input-meta 
/work/rec_ln1rsqrtmul_quant/PP-OCRv6_rec_ln1rsqrtmul_inputmeta.yml --output-path 
/project/rec_pctail_export/rec_pctail --optimize VIP9000NANODI_PLUS_PID0X1000003B --dtype quantized 
--save-fused-graph --pack-nbg-unify --viv-sdk /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools 
--build-platform make --target-ide-project linux64 > /project/pctail_export.log 2>&1
```

## visual ai offline answering service

visual_ai/systemd_user_services/visual-ai-offline-answering.service

Complete file  |  19 source lines  |  Public SHA256 cb0cadf7df5ec514b42b9cc1bd8fe06ebfd4d5d7818aaea8719a8dce224411b1

```
[Unit]
Description=the OCR board resident Visual AI offline answering model
StartLimitIntervalSec=0

[Service]
Type=exec
WorkingDirectory=/home/user/rapidocr-npu/visual_ai
ExecStartPre=/usr/bin/python3 /home/user/rapidocr-npu/visual_ai/check_offline_port.py
ExecStart=/home/user/rapidocr-npu/visual_ai/offline_answering/llama_runtime/build-board/bin/llama-se
rver --model /home/user/rapidocr-npu/visual_ai/offline_answering/answering_models/qwen3-4b-instruct-
2507-q8_0.gguf --host 127.0.0.1 --port 8080 --ctx-size 8192 --threads 4 --parallel 1 --alias local 
--sleep-idle-seconds -1
Restart=always
RestartSec=3
TimeoutStopSec=30
KillMode=control-group
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visual-ai-offline-answering

[Install]
WantedBy=default.target
```

## check offline port py

visual_ai/check_offline_port.py

Complete file  |  7 source lines  |  Public SHA256 828c55768dc2a1f0fbeb8b1db82b188a5d034d869fd4ec58e0616ac3ce752c89

```
"""Refuse to spawn a second server when the offline endpoint is occupied."""
import socket

with socket.socket() as connection:
    connection.settimeout(1)
    if connection.connect_ex(("127.0.0.1", 8080)) == 0:
        raise SystemExit("Offline port 8080 is already occupied; refusing a duplicate server.")
```
