# Getting RapidOCR Running on the Orange Pi Zero 3W NPU

## Porting PP-OCRv6 to the Allwinner A733 / Vivante VIP9000 and Integrating It into Visual AI

## Publication status and public paths

This README combines the engineering paper with code excerpts and the rebuild/execution commands supplied from the project record. Commands are placed next to the stages they explain, rather than collected only at the end.

**The complete sources, models, compiler artifacts, SDK dependencies, and test fixtures have not yet been uploaded to this repository.** The inventory below describes files reported as retained in the development project; it is not a list of downloads currently available here. These examples are therefore conditional instructions for the forthcoming source supplement, not a tested fresh-install tutorial. No commands were executed on the target board while preparing this revision.

Use the same generic project root on the compiler host and the board:

```sh
cd "$HOME/orange-pi-zero-3w-rapidocr-npu"
```

The path above is a public layout convention, not an installation command. A clone currently provides this documentation, not the missing runtime or models. If you store the release elsewhere, substitute your own checkout root. Do not use another developer's Mac home directory, SSH alias, dated workspace, or private application folder.

| Public location relative to the project root | Contents and purpose |
|---|---|
| `runtime/` | Standalone OCR sessions, native bridge, launcher, validators, and pinned requirements |
| `venv/` | Board-side Python environment expected by the supplied validator commands |
| `runtime/models/` | NBGs, tensor metadata, canonical ONNX models, and vocabulary |
| `runtime/compiler/` | Saved production ACUITY graph, weights, quantization, input metadata, and rebuild script |
| `runtime/samples/` | Redistributable reference fixtures |
| `integration/visual_ai/` | Optional OCR integration adapter, persistent worker, and accuracy modules |
| `work/` | Retained conversion sources and clearly identified historical experiments |

Actual source basenames and API names such as `luna_ocr.py`, `pai_ocr_backend.py`, `PAI_OCR_BACKEND`, and `run_luna_assistant.sh` are retained for compatibility. They are not personal directory paths. Renaming them without updating their callers would break the examples. The public application name is **Visual AI**.

**Public-path migration must also be applied inside the source supplement.** Changing this README does not change hard-coded paths in scripts, imports, compiler metadata, launchers, or service definitions. Those dependencies must be inspected and changed to checkout-relative paths or documented configuration before the package is described as portable.

The project identifies its models as PP-OCRv6. Exact upstream identifiers, model hashes, and dictionary provenance must accompany the release; this documentation revision does not independently establish model identity.

### Abstract

This project began with a practical goal: run RapidOCR on an Orange Pi Zero 3W using its neural processing unit while preserving useful recognition accuracy and reliability.

The development board, named Luna, used an Allwinner A733 with two Cortex-A76 CPU cores, six Cortex-A55 CPU cores, and a Vivante VIP9000 NPU. The accelerator was accessed through the Vivante conversion and execution chain:

```text
ONNX → ACUITY → OpenVX/Vivante compilation → NBG → VIPLite → /dev/vipcore
```

Getting PP-OCRv6 to execute correctly required substantially more than installing RapidOCR or successfully compiling an ONNX model. The work included static input conversion, tensor-layout and quantization validation, graph restructuring, isolated operator tests, LayerNorm substitutions, simulator-to-hardware comparisons, repeated physical inference, and application integration.

The final deployed OCR system uses NPU text detection, NPU recognition attempts on supported crops, and canonical CPU recognition for every crop. CPU probabilities determine the final recognition output, preserving the reference recognizer’s text, scores, and confidence filtering. CPU processing also handles orientation classification, wide-line recognition, image preparation, and post-processing.

Later full-frame accuracy work substantially improved small-print recognition. On the original six PAPER/SCREEN test images, word recall increased from between 0% and 42.11% to between 97.81% and 100%. PAPER 2 reached 99.04% word recall, although punctuation and a temperature-symbol error remained.

A subsequent effort to eliminate CPU recognition reduced CPU work but failed the combined accuracy and speed requirements. Across 74 inputs, the selected NPU-only recognition candidate was approximately 79% slower than production and introduced recognition regressions. Further width, precision, overlap, and angle-classifier experiments did not provide a safe production improvement.

The result is a functioning, physically validated hybrid OCR implementation integrated into Visual AI. It is not a claim of perfect transcription, universal acceleration, or complete CPU elimination.

## 1. Hardware and Software Platform

The tested board was an Orange Pi Zero 3W running:

| Component | Tested configuration |
|---|---|
| SoC | Allwinner A733 |
| CPU | 2× Cortex-A76 and 6× Cortex-A55 |
| NPU | Vivante VIP9000 |
| Operating system | Orange Pi Debian 12 Bookworm image |
| Architecture | AArch64 |
| Kernel | `6.6.98-sun60iw2` |
| NPU device | `/dev/vipcore` |
| ACUITY | 6.30.22 |
| VivanteIDE | 5.11.0 |
| VIPLite userspace banner | `2.0.3.2-AW-2024-08-30` |
| Kernel NPU driver banner | `2.0.3.4-AW-2025-10-27` |
| RapidOCR | 3.9.2 |
| ONNX Runtime in the OCR environment | 1.30.0 |
| NumPy | 2.4.6 |
| OpenCV Python | 5.0.0.93 |

The compiler target was:

```text
VIP9000NANODI_PLUS_PID0X1000003B
```

The A733 CPU configuration is documented in the [Allwinner A733 datasheet](https://dl.radxa.com/cubie/a7a/docs/hw/datasheet/A733_Datasheet_V0.93.pdf). Orange Pi’s build configuration identifies the Zero 3W’s A733 device-tree file. [Orange Pi board configuration](https://github.com/orangepi-xunlong/orangepi-build/blob/next/external/config/boards/orangepizero3w.conf)

Other A733 developers have independently demonstrated Vivante NBG execution through VIPLite and `/dev/vipcore`. That establishes related platform support, but does not independently validate the OCR transformations or measurements reported here. [A733/VIP9000 implementation reference](https://github.com/unnamedwild-ux/frigate_npu_vivante)

Development used an existing Docker compiler environment on the Mac. Final execution and production validation took place on physical Luna hardware.

The compiled ARM64 runtime bridge cannot execute directly on macOS.

### 1.1 Environment setup boundary

The supplied inventory names `runtime/install.sh` and `runtime/requirements.txt`, but does not include their contents or the exact environment-creation and installation command history. Installing the Python dependencies alone does not install the Vivante driver, runtime libraries, SDK, or headers. The pinned versions in the table describe the reported OCR environment; they are not a guarantee that those packages can be installed on every image.

Before running the examples, the source supplement must document board permissions for `/dev/vipcore`, vendor library/header acquisition and compatibility, Python environment creation, and the compiler image's provenance. `ubuntu-npu:v2.0.10.2` below is the existing local compiler image tag used in the project, not a verified public image download.

**Not supplied:** the complete installation transcript, known-good initial NBG invocation, and tested clean-system provisioning instructions. These must come from the original scripts and records rather than guessed package or driver commands.

## 2. Why Installing RapidOCR Was Not Enough

RapidOCR can use ONNX Runtime for CPU inference, but its ONNX models cannot simply be passed unchanged to this accelerator.

Several distinct problems had to be separated:

- Dynamic dimensions unsuitable for the selected compiled execution path.
- Incorrect preprocessing or tensor packing.
- Quantization ranges that damaged recognition.
- Graph structures that behaved differently after compilation.
- Numerically incorrect results from particular operator configurations.
- Recognition errors that remained despite successful execution.

A successful import, quantization, compilation, or inference call was insufficient evidence of correctness.

We therefore compared multiple stages:

```text
Original ONNX reference
        ↓
Transformed ONNX
        ↓
Imported ACUITY model
        ↓
Vivante simulator
        ↓
Physical VIP9000 execution
        ↓
Decoded OCR output
```

Intermediate tensors and CTC predictions were checked separately. This helped distinguish a graph-conversion problem from a quantization, runtime, or decoding problem.

The findings apply to the tested models, compiler versions, and tensor configurations. They should not be generalized into claims that every instance of a particular Vivante operator is defective.

## 3. OCR Architecture and the Production Accuracy Policy

The OCR pipeline is:

```text
Full image
  → preprocessing
  → text detection
  → internal text-region extraction
  → orientation classification
  → recognition
  → CTC decoding
  → confidence filtering and reading order
```

The deployed allocation is:

| Processing stage | Execution |
|---|---|
| Text-detection neural network | NPU |
| Recognition on supported fixed-width crops | NPU attempt plus CPU verification |
| Canonical recognition for every crop | CPU |
| Genuinely wide-line recognition | CPU |
| Angle/orientation classification | CPU |
| Image resizing, padding, region extraction | CPU |
| Detection post-processing and CTC decoding | CPU |
| Final recognition probabilities and scores | Canonical CPU output |

An important detail is that production does **not** run CPU recognition only when an NPU result appears uncertain.

It runs canonical CPU recognition on **every crop**, compares eligible NPU predictions against it, and returns the canonical CPU probabilities. The CPU inference occurs before the NPU comparison in the implementation.

This catches confident NPU mistakes and preserves CPU confidence-filtering behavior. It also means that production retains substantial CPU recognition work.

The successful NPU recognizer is therefore an operational component of a checked pipeline, not proof that production recognition has become CPU-free.

The purpose of the system was to improve practical accuracy, speed, and reliability together. Reducing CPU use alone was not accepted as sufficient justification for deployment.

## 4. Static Detector Conversion

The detector input was fixed at:

```text
[1, 3, 736, 736]
```

The deployed detector metadata describes a UINT8 input and output.

| Tensor | Shape | Type | Scale | Zero point |
|---|---|---|---:|---:|
| Input | `[1,3,736,736]` | UINT8 | 0.007843137718737125 | 127 |
| Output | `[1,1,736,736]` | UINT8 | 0.00392148969694972 | 0 |

The baseline preprocessing preserves aspect ratio, resizes the image into a 736×736 white canvas, applies the configured normalization, and converts to NCHW.

The valid portion of the returned probability map is separated from padding before detection post-processing.

Channel ordering, normalization, shape, layout, scale, and zero point must match the conversion metadata. Guessing these values can make a correctly compiled model produce unusable results.

The later production accuracy profile reuses this same detector binary multiple times on overlapping tiles. It does not replace the detector model with a larger-input NBG.

### 4.1 Detector rebuild materials

The public package must include the actual detector import, static-shape conversion, calibration, quantization, and export commands with their input-model hashes and preprocessing configuration. Codex's supplied text does not contain those commands. The recognizer re-export command in Section 9 does not rebuild the detector.

The recorded early detector inference was approximately 27.7 ms, with a 100-run execution test. This is a historical detector-only measurement, not complete OCR latency.

## 5. Static Recognizer Conversion

The original recognizer supports variable width:

```text
[N, 3, 48, W]
```

The compiled recognizer was fixed at:

```text
Input:  [1, 3, 48, 320]
Output: [1, 40, 18710]
```

The output represents 40 CTC time steps over 18,710 classes.

The deployed recognizer input is **signed INT8**, not the UINT8 configuration used in some earlier experiments.

| Tensor | Shape | Type | Scale | Zero point |
|---|---|---|---:|---:|
| Input | `[1,3,48,320]` | INT8 | 0.007843137718737125 | −1 |
| Output | `[1,40,18710]` | FLOAT16 | Not applicable | Not applicable |

The selected recognizer build used per-channel INT8 quantization together with the validated LayerNorm division substitutions.

The historical UINT8 input configuration with scale approximately `0.0062129949` and zero point `161` belongs to an earlier research stage. It should not be presented as the deployed input format.

At one intermediate stage, floating-point conversion preserved all 40 argmax positions against the ONNX reference. This established that the import stage could preserve the reference prediction before later quantization and execution problems were investigated.

## 6. Graph Transformations

### 6.1 Stride-2 convolution restructuring

Two stride-2 convolution structures in the recognizer were rewritten as:

```text
Stride-1 convolution
        ↓
1×1 MaxPool with stride 2
```

The pooling operation performs spatial sampling after convolution. Its kernel is 1×1, so it does not introduce ordinary neighborhood maximum pooling.

The retained rewrite script checks the expected number of replacements rather than silently applying an arbitrary transformation.

### 6.2 Squeeze-and-excitation projection rewrites

Several problematic 1×1 projections operated on spatially reduced tensors of shape:

```text
[1, C, 1, 1]
```

For these specific bias-free projections, the computation was expressed as:

```text
Reshape → broadcast Multiply → ReduceSum → Reshape
```

The retained script covers ten projections, including channel mappings such as:

```text
96 → 24
24 → 96
192 → 48
48 → 192
384 → 96
96 → 384
```

This is a rewrite for the checked spatially reduced projections, not a generic replacement for every 1×1 convolution.

A conventional FullyConnected substitution did not resolve the relevant execution problem.

### 6.3 Conv40 output splitting and reconstruction

A later projection with weights shaped:

```text
[384, 768, 1, 1]
```

was split into two output-channel groups. The initial rewrite used two convolutions and concatenation.

The cumulative transformed model subsequently used a Pad/Add reconstruction variant. This distinction matters: the initial split script alone is not the complete final transformation.

### 6.4 LayerNorm square and affine rewrites

Five LayerNorm square operations were rewritten from:

```text
Pow(x, 2)
```

to:

```text
Mul(x, x)
```

The cumulative graph also includes affine subtraction restructuring.

The saved transformed ONNX graphs and final ACUITY compiler graph preserve these changes. Some transformations were developed through generated-graph or generated-C edits rather than a single standalone ONNX rewrite script.

### 6.5 Reproducing the transformation sequence

The retained script inventory identifies `rewrite_allstride2_maxpool.py`, `rewrite_conv5_mulsum.py`, `rewrite_conv40_outsplit2.py`, and `rewrite_layernorm_mul.py`. Their command-line arguments were not supplied with this text. Do not assume they all accept the same input/output flags or invent an invocation from their filenames.

The source supplement must provide the exact execution order, input/output graph hashes, changed-node counts, tensor shapes, weight mappings, and numerical checks. The four scripts alone do not reconstruct the complete cumulative model: Pad/Add reconstruction, affine restructuring, and some LayerNorm changes remain represented in saved cumulative graphs or compiler artifacts. Section 9 documents re-export of that saved production graph, which is a narrower operation than rebuilding it from an upstream ONNX model.

## 7. The LayerNorm Divide Breakthrough

One of the most important findings came from isolating the first LayerNorm arithmetic.

The relevant computation is:

```text
centered / sqrt(variance + epsilon)
```

In the tested Vivante configuration, the original node213 `VSI_NN_OP_DIVIDE` produced numerically incorrect results.

It was replaced with:

```text
inverse_std = Pow(variance + epsilon, -0.5)
normalized  = Mul(centered, inverse_std)
```

The replacement operations were tested against their **actual quantized inputs**.

| Isolated operation | Result |
|---|---|
| Replacement node212 Pow(−0.5) | 100% exact; maximum quantized error 0 |
| Replacement node213 Multiply | 100% exact; maximum quantized error 0 |

These isolation results are attributed to the companion investigation. Its original isolation dumps and checker scripts have not all been supplied for independent repetition. The reported tests validated the local arithmetic bypass. They did not establish that the complete quantized recognizer matched the floating-point CPU model.

### 7.1 Saturation around node211

The preceding node211 `Add.159` also exposed a precision problem.

In the companion investigation's tested UINT8 configuration, its output clipped 7 of 40 values at 255. This count was not established for the final signed-INT8 production graph. Signed INT8 does not, by itself, expand the represented real-valued range. The recorded output scale was approximately:

```text
0.0015841168351471424
```

with zero point 0.

Changing only this generated-C output to FLOAT16 improved the experimental CTC sequence:

```text
Before:
[5032,2241,7038,74,11935]

Node211 FLOAT16:
[5032,55,2241,7038,11935,3795,2737,4645]

Expected:
[11194,7920,2241,7038,11183,11935,3795,2737,4645]
```

The final three expected characters were recovered, but the sequence remained incorrect.

This showed that precision around LayerNorm mattered while also demonstrating that one local precision change was insufficient.

The node211 FLOAT16 experiment was a generated-C patch. It was not automatically embedded in the cumulative ONNX model.

### 7.2 Applying the bypass across LayerNorm blocks

The deployed build extended reciprocal-square-root multiplication to all five LayerNorm divisions.

The first bypass was already present in the imported graph. The build script rewrote the remaining four `sqrt`/`real_div` pairs in ACUITY JSON.

Together with the selected input quantization range, this improved the underlying NPU recognizer’s agreement on the earlier 18-line page from 9/18 to 14/18.

## 8. Large Output-Classifier Experiments

The recognizer includes a large classifier operation:

```text
[40,120] × [120,18710] → [40,18710]
```

One experiment split the 18,710 output classes into:

- Eighteen chunks of 1,024 classes.
- One final chunk of 278 classes.

The transformed CPU graph retained all 40 argmax predictions with effectively unchanged numerical output.

Physical Vivante execution nevertheless produced inconsistent results across chunks. Some matched the reference while others did not.

Reducing a problematic chunk to 256 outputs did not automatically repair the arithmetic.

This was a rejected research approach. The split classifier should not be described as a successful deployed optimization.

The finding reinforced the need to validate each compiled implementation rather than infer correctness from mathematical equivalence.

## 9. Quantization and Host Tensor Packing

Quantization was treated as part of the model rather than a final optimization switch.

The project tested:

- Asymmetric UINT8 configurations.
- Per-channel INT8 configurations.
- Explicit input ranges spanning `[-1,1]`.
- Selected FLOAT16 tensors and sequence-tail configurations.
- Calibration using existing recognition crops.
- Precision changes around LayerNorm.
- Wider static recognition inputs.

The production input encoder uses:

```python
q = clip(round(x / scale + zero_point), dtype_min, dtype_max)
```

Its intermediate arithmetic uses FLOAT64 with the serialized scale. This avoids moving values across half-step rounding boundaries through FLOAT32 approximation.

The relevant production method excerpt is shown below. It depends on its containing session class, metadata, and NumPy import; it is not a standalone program:

```python
def encode(self, x):
    x = np.asarray(x, dtype=np.float32)

    if list(x.shape) != self.input_meta['shape']:
        raise ValueError(
            f"NPU requires {self.input_meta['shape']}, "
            f"received {list(x.shape)}"
        )

    q = self.input_meta.get('quantize')
    dtype = self.dtype(self.input_meta)

    if q:
        limits = np.iinfo(dtype)
        x = np.clip(
            np.rint(
                x.astype(np.float64) / q['scale']
                + q['zero_point']
            ),
            limits.min,
            limits.max,
        )

    return np.ascontiguousarray(x, dtype=dtype)
```

Calibration metadata must also preserve channel ordering and normalization. The retained recognizer input metadata includes mean `127.5`, scale approximately `2/255`, NCHW layout, and channel-reversal settings.

Calibration experiments used a limited collection of supplied crops. They should not be described as a large, independent calibration corpus.

### 9.1 Re-export the saved production recognizer — RUN ON COMPILER HOST

From the public project root, with the saved compiler files and the existing image available:

```sh
docker run --rm \
  --platform linux/amd64 \
  --network none \
  -v "$PWD/runtime/compiler:/project" \
  ubuntu-npu:v2.0.10.2 \
  sh /project/rebuild.sh
```

The container mounts `runtime/compiler/` at `/project`. `--network none` disables container network access; it does not supply a missing image or SDK. The `linux/amd64` platform is the compiler environment, not the board's ARM64 runtime architecture.

The retained `rebuild.sh` invokes this export inside that compiler container:

```sh
python3 /root/acuity-toolkit-whl-6.30.22/bin/pegasus.py \
  export ovxlib \
  --model /project/model.json \
  --model-data /project/model.data \
  --model-quantize /project/model.quantize \
  --with-input-meta /project/inputmeta.yml \
  --output-path /project/rebuilt/rec_pc \
  --optimize VIP9000NANODI_PLUS_PID0X1000003B \
  --dtype quantized \
  --save-fused-graph \
  --pack-nbg-unify \
  --viv-sdk /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools \
  --build-platform make \
  --target-ide-project linux64
```

The `/root/acuity-toolkit-whl-6.30.22/` and `/root/Vivante_IDE/` paths are SDK locations inside the recorded compiler image, not personal project paths on a reader's computer. A different licensed SDK installation requires adapting those locations.

This operation reuses `model.quantize`; it does not repeat calibration. Recalibration requires the actual dataset manifest and valid calibration paths in `inputmeta.yml`. The expected export location is `runtime/compiler/rebuilt/`; the exact NBG selection, copy to `runtime/models/recognizer.nb`, metadata generation, and hashes must be documented from `rebuild.sh` before claiming a complete rebuild.

**Not supplied:** exact upstream-model import commands, dynamic-to-static invocation, calibration commands, all historical graph-edit invocations, and the complete detector build. These remain release gaps even though a saved production graph can be re-exported.

## 10. The VIPLite Runtime Bridge

RapidOCR was connected to VIPLite through a custom C shared library and Python session adapters.

This is not a built-in ONNX Runtime Vivante execution provider.

The bridge:

1. Initializes VIPLite.
2. Loads the NBG network.
3. Queries input and output tensor properties.
4. Creates and maps buffers.
5. Copies encoded input.
6. Flushes input memory.
7. Executes the network.
8. Invalidates output memory.
9. Copies output.
10. Releases resources when the engine closes.

It checks tensor byte counts and rejects unsupported tensor formats.

The Python layer serializes bridge calls through a shared lock. The complete OCR engine is used sequentially because detector preprocessing stores image-specific geometry.

### Core execution function

```c
int ocr_run(
    ocr_handle *h,
    const void *input,
    size_t input_bytes,
    void *output,
    size_t output_bytes
) {
    if (!h ||
        input_bytes != h->input_bytes ||
        output_bytes != h->output_bytes) {
        snprintf(
            error_text,
            sizeof(error_text),
            "Tensor byte count mismatch"
        );
        return -1;
    }

    memcpy(h->input_ptr, input, input_bytes);

    if (status_check(
            vip_flush_buffer(
                h->input,
                VIP_BUFFER_OPER_TYPE_FLUSH
            ),
            "flush input"
        ) ||
        status_check(
            vip_run_network(h->network),
            "run network"
        ) ||
        status_check(
            vip_flush_buffer(
                h->output,
                VIP_BUFFER_OPER_TYPE_INVALIDATE
            ),
            "invalidate output"
        )) {
        return -1;
    }

    memcpy(output, h->output_ptr, output_bytes);
    return 0;
}
```

The complete bridge also contains initialization, buffer-property queries, size-overflow checks, reference counting, error reporting, and cleanup. Those functions must accompany the excerpt.

### 10.1 Build the native bridge — RUN ON BOARD

On a matching ARM64 board, after obtaining compatible vendor libraries and the required headers, start from the public project root and enter the supplied runtime directory:

```sh
cd "$HOME/orange-pi-zero-3w-rapidocr-npu/runtime"
cc -O2 -fPIC -shared -Wall -Wextra -Werror \
  -Iinclude \
  vip_bridge.c \
  -lNBGlinker \
  -lVIPhal \
  -o librapidocr_vip.so
```

This is the supplied bridge build command with a generic working directory. It produces `runtime/librapidocr_vip.so`. The `-Iinclude` argument resolves to `runtime/include/`; `-lNBGlinker` and `-lVIPhal` require the compatible vendor libraries to be available to the toolchain and runtime loader. Header/library redistribution rights must be handled separately.

Do not run the ARM64 bridge on macOS or mistake this native runtime build for the x86-64 compiler export. The complete `vip_bridge.c`, not only the excerpt above, is needed for compilation.

## 11. Physical Execution and Numerical Validation

The recognizer passed 100 consecutive runs on the saved reference input.

The expected decoded CTC sequence was:

```text
[11194,7920,2241,7038,11183,11935,3795,2737,4645]
```

The decoded reference text was:

```text
绿洲仕格维花园公寓
```

Raw physical output matched the retained vendor-runner reference and simulator output byte-for-byte for this fixture.

The compared FLOAT16 output contained:

```text
1,496,800 bytes
```

Its recorded SHA-256 was:

```text
a367b42f56f88093946bff8237a6fe2f724a09da5640d7df617f08d6aaf94510
```

The saved 100-run measurements were:

| Measurement | Median |
|---|---:|
| Recognizer wrapper total | 13.364 ms |
| Measured native bridge call | 11.758 ms |

The native measurement includes bridge execution and associated transfer/cache operations. It is not an isolated accelerator-kernel measurement.

Byte-for-byte parity on one fixture and 100 stable repetitions establish reproducibility for that fixture. They do not establish general recognition accuracy.

### 11.1 Run the retained validators — RUN ON BOARD

With the complete runtime, models, fixtures, and board-side `venv/` installed, run from `runtime/`:

```sh
cd "$HOME/orange-pi-zero-3w-rapidocr-npu/runtime"
../venv/bin/python selftest.py --runs 100
../venv/bin/python validate_verified.py
../venv/bin/python validate_pipeline.py
```

`selftest.py --runs 100` performs the repeated recognizer/reference-output test. `validate_verified.py` checks canonical equivalence and inference-failure recovery. `validate_pipeline.py` performs the separately defined pipeline comparison. Their full implementations must define fixture selection, expected outputs, and result-artifact locations. A validator's exit status or successful execution is not a substitute for examining its reported comparisons.

The model and fixture hashes must match the corresponding historical records before comparing against the checksum above. New compiler exports or different fixtures may produce different checksums; do not overwrite a reference merely to make a test pass.

## 12. CPU-Checked Recognition

The verified session evaluates the original normalized crop using the canonical ONNX recognizer.

For eligible crops, it also evaluates the NPU recognizer and compares decoded CTC tokens.

The essential policy is:

```python
canonical = self.checker.run(
    None,
    {self.checker_input: original}
)[0]

expected = ctc_tokens(canonical)

# Eligible crops receive an NPU prediction and comparison.
# NPU disagreement or an inference exception is recorded.

results.append(canonical)
```

Returning `canonical` even when the token sequence agrees preserves CPU probabilities and final confidence filtering.

Consequently:

- NPU/CPU token agreement does not imply identical probabilities.
- CPU correction means correction to the canonical model, not necessarily to visible ground truth.
- CPU verification cannot recover a line that detection never found.
- NPU inference errors can recover through already-computed CPU output.
- CPU errors propagate instead of returning an unchecked NPU result.
- Startup still requires working NPU models and VIPLite initialization.

The legacy runtime mode named `npu` also contains CPU handling for genuinely wide or unsupported crops. It should not be described as strictly CPU-free recognition.

Separate experiments implemented genuinely NPU-only recognition without those fallbacks.

### 12.1 Run the standalone CPU-checked pipeline — RUN ON BOARD

This is the supplied standalone launch workflow with the private SSH alias and dated project directory removed. Connect to your own board using your normal SSH configuration, then run:

```sh
cd "$HOME/orange-pi-zero-3w-rapidocr-npu/runtime"
sh run.sh samples/test_page.jpg
```

The fixture must be supplied or replaced with your own permitted image. The complete `run.sh` must select the documented verified mode. This standalone command does **not** automatically exercise the application's full-frame accuracy wrapper.

### 12.2 Use the verified engine from Python — RUN IN THE BOARD OCR ENVIRONMENT

From `runtime/`, using the Python environment expected by the supplied sources:

```python
from luna_ocr import build_ocr, close_ocr

ocr = build_ocr("verified")

try:
    result = ocr("samples/frame.jpg")
    print(result.txts)
    print(result.scores)
    print(result.boxes)
finally:
    close_ocr(ocr)
```

Here `samples/frame.jpg` is a reader-selected image relative to `runtime/`, not a claim that the repository currently contains that file. Keep one engine alive across sequential frames to avoid rebuilding it for every image, and close it when finished. The source module name `luna_ocr` is retained because it is the actual supplied API name.

## 13. Earlier Page-Level Validation

On the earlier 18-line page:

| Measurement | Result |
|---|---:|
| NPU recognition attempts | 18 |
| CPU verification runs | 18 |
| NPU/CPU agreements | 14 |
| CPU corrections | 4 |
| Final lines matching canonical CPU | 18/18 |

Five repeated verified runs preserved the reference text, scores, and boxes.

The recorded page processing time was approximately 1.64–1.66 seconds, excluding engine construction.

An earlier small-line test measured approximately 120.3 ms using NPU detection and recognition with CPU angle classification. That measurement belongs to the earlier execution configuration.

The CPU-checked small-line tests were approximately 195–204 ms. The 120.3 ms figure should not be presented as the latency of the later CPU-checked high-resolution production profile.

## 14. Full-Frame Accuracy Improvements

Once the runtime worked, real camera frames exposed another major problem: reducing a high-resolution page to one 736×736 detector input lost small text.

The first six-frame regression used three PAPER images and three SCREEN images. Ground truth was manually transcribed from visible text, preserving source wording and typographical errors.

The retained improvement combined:

- A full-frame detection canvas with a 2,080-pixel maximum dimension.
- Overlapping 736×736 NPU detector tiles.
- A 672-pixel tile step, normally giving 64-pixel overlap.
- Weighted blending of detector probability maps.
- Recognition crops extracted from full-resolution image pixels.
- Lower box threshold for large images.
- PAPER-specific box expansion.
- Additional undilated-mask detections.
- Conditional missing-line proposals.
- Conditional recognition retries.
- Geometry-based reading-order correction.

The detector and recognizer NBG models remained unchanged during this accuracy work.

### 14.1 Full-frame preservation

The original camera frame is preserved.

Detector tiling and text-region extraction are internal OCR operations. They do not replace the full frame supplied to OpenAI.

The implementation does not claim that OCR can operate without internal region extraction. Standard recognition still consumes detected line regions.

### 14.2 Tiled detection

The source image is resized proportionally onto a white 2,080×2,080 canvas. Relevant tiles are normalized and evaluated individually using the existing 736×736 detector.

Probability maps are combined with tapered weights near tile edges. This reduces boundary artifacts while retaining small-print detail.

### 14.3 PAPER recovery

For large PAPER inputs, the initial pass uses:

```text
Segmentation threshold: 0.30
Box threshold:          0.35
Unclip ratio:           1.0
```

SCREEN retains an unclip ratio of 1.6.

A long PAPER line with recognition confidence below 0.90 can trigger a second pass. The second pass reuses the cached detector prediction rather than repeating all detector inference.

The recovery path can:

- Remove detection-mask dilation.
- Increase the post-processing candidate limit.
- Propose missing lines from nearby geometry.
- Retry selected long recognition regions at reduced heights.
- Correct the order of suitable short-text rows.

Missing-line proposals require high recognition confidence and agreement between multiple geometry trials. No reference answer text is injected into OCR.

The final retained recognition retries use height fractions of 0.7, 0.6, and 0.5, shifted downward by 10% of the original line height. A retry replaces the original only when its confidence improvement exceeds the configured margin.

These heuristics improved the tested cases but are not guaranteed to detect every missing line.

### 14.4 Use the deployed accuracy wrapper — OPTIONAL INTEGRATION EXAMPLE

The supplied adapter API is shown below. The forthcoming public layout places it in `integration/visual_ai/`. Its adapter imports and runtime/model discovery must first be made checkout-relative; moving the files alone is not sufficient.

Run in the compatible board-side application environment with `pai_ocr_backend.py` importable:

```python
from pai_ocr_backend import create_ocr_engine, close_ocr_engine

ocr = create_ocr_engine(accuracy=True)

try:
    result = ocr(
        "samples/frame.jpg",
        source_mode="PAPER",
    )
    print(result.txts)
finally:
    close_ocr_engine(ocr)
```

`samples/frame.jpg` is an example image path relative to the current working directory. Explicit `source_mode="PAPER"` avoids relying on a PAPER filename convention; use the supported SCREEN mode for SCREEN inputs. This example invokes the OCR adapter, not the answering, speech, or camera application.

The retained `luna_accuracy/` modules implement the tiling, cache, recovery, retry, and reading-order behavior described above. The complete modules and their configuration defaults are required; this short invocation is not a reimplementation of those algorithms.

## 15. Six-Frame Before/After Results

The deployed profile produced:

| Frame | Before word recall | After word recall | Before median | After median |
|---|---:|---:|---:|---:|
| PAPER 1 | 32.79% | 97.81% | 1.81 s | 5.14 s |
| PAPER 2 | 19.23% | 99.04% | 1.00 s | 17.63 s |
| PAPER 3 | 9.54% | 100.00% | 0.84 s | 8.00 s |
| SCREEN 1 | 42.11% | 100.00% | 1.48 s | 3.27 s |
| SCREEN 2 | 37.70% | 100.00% | 1.44 s | 3.22 s |
| SCREEN 3 | 0.00% | 100.00% | 0.70 s | 3.61 s |

These times include substantially more detected and recognized text after improvement. A fast result that omits most of a page is not an equally accurate comparison.

Accuracy was prioritized for this deployment. These improvements should not be advertised as a speed improvement over the original low-recall full-frame configuration.

### 15.1 PAPER 2 progression

PAPER 2 progressed through several stages:

```text
Original baseline:          19.23% word recall
Initial tiled candidate:    approximately 69.23%
Later recovery candidate:   98.56%
Deployed refinement:        99.04%
```

The last refinement recovered:

```text
maintain
```

instead of:

```text
maintaine
```

The measured three-run medians were approximately 17.65 seconds before that refinement and 17.63 seconds after it. That small difference is within timing noise.

The remaining temperature-symbol error was:

```text
Visible: 99.9°F
OCR:     99.9'F
```

Punctuation omissions also remained.

PAPER 2 therefore reached 99.04% normalized word recall and ordered word accuracy, not perfect transcription.

### 15.2 Meaning of “100%”

Word recall counts recovered reference-word occurrences after normalization. It does not guarantee correct:

- Punctuation.
- Capitalization.
- Reading order.
- Line splitting or merging.
- Extra words.
- Headers or surrounding interface text.

The three original SCREEN frames reached 100% question/options word recall. That does not mean all later SCREEN captures reached 100%.

Ordered word error rate and character-level measures were retained alongside recall.

## 16. Additional Regression Data

The additional supplied sessions contained 75 eligible inputs, including exact duplicates.

After SHA-256 deduplication, 66 unique inputs were evaluated:

- Eight PAPER images.
- Fifty raw SCREEN captures.
- Eight previously saved SCREEN section images.

The saved section images were evaluated as supplied. They were not newly cropped by the validation process.

These captures include repeated questions and correlated burst frames. They are broader than the six tuning images, but not a fully independent large text corpus.

The validation found:

| Group | Weighted word recall |
|---|---:|
| Eight additional PAPER images | 96.29% |
| Nine saved diagnostic full SCREEN captures | 92.63% |
| Fifty distinct raw SCREEN captures | 86.93% |

The nine-capture diagnostic summary is a separately reported subset, not a third disjoint group to add to the 66-frame total. Its exact membership and overlap must be resolved from the input manifests. The combined 58-frame SCREEN result is reported separately in Section 17.

Several difficult inputs remained below 98%. Some poor SCREEN captures produced almost no usable OCR.

Important remaining failures included:

- Entire question blocks absent from detection.
- Missing first prompt lines.
- Gibberish in otherwise detected lines.
- Merged words.
- Incorrect short symbols and abbreviations.
- Reading-order errors around uneven answer rows.

The confidence-based PAPER recovery did not trigger on the additional PAPER pages. This exposed a limitation: a missing line has no recognition confidence, and an incorrect line can still receive a high score.

A later deployment comparison confirmed that the final PAPER 2 refinement preserved text, scores, boxes, and recognition counters on all 66 additional inputs.

The broader evidence supported a substantial improvement over the original full-frame baseline, but not a universal 98% or 100% accuracy claim.

## 17. Attempts to Remove CPU Recognition

After protecting the improved production path, a separate experiment implemented genuinely NPU-only recognition.

It reused regression data without changing production models or Visual AI integration.

Wide lines were processed using:

```text
Window width: 320 pixels
Window step:  256 pixels
Overlap:      64 pixels
```

CTC probabilities were stitched using the window with the greatest surrounding context.

Unlike the legacy runtime’s `npu` mode, this experimental recognizer had no CPU recognition fallback and no CPU verification session.

A FLOAT16 sequence-tail model improved a frozen 48-crop probe from 38/48 exact CTC matches with the production NPU model to 42/48.

That improvement was insufficient for production promotion.

### 17.1 Original six frames

| Frame | Production recall | Improved NPU-only recall |
|---|---:|---:|
| PAPER 1 | 97.81% | 95.63% |
| PAPER 2 | 99.04% | 98.56% |
| PAPER 3 | 100.00% | 98.36% |
| SCREEN 1 | 100.00% | 100.00% |
| SCREEN 2 | 100.00% | 100.00% |
| SCREEN 3 | 100.00% | 98.21% |

### 17.2 Additional 66 frames

| Group | Production weighted recall | NPU-only weighted recall |
|---|---:|---:|
| PAPER, eight frames | 96.29% | 96.62% |
| SCREEN, 58 frames | 88.25% | 87.85% |

The SCREEN grouping here combines raw captures and saved sections, so it differs from the raw-only grouping in the preceding section.

Across 72 reference-scored images:

- Two improved in word recall.
- Twenty-four regressed.
- Forty-six had equal word recall.

Equal recall did not imply identical text.

Across all 74 inputs, including two small reference fixtures:

- Complete OCR text matched on six inputs.
- Boxes matched on 27 inputs.
- Complete score lists matched on none.

The saved comparison contains 1,821 box-associated text, score, missing-region, and added-region differences.

## 18. Resource and Latency Comparison

The 74-input experiment measured:

| Measurement | Production | Improved NPU-only recognition |
|---|---:|---:|
| Total elapsed time | 335.943 s | 602.480 s |
| Median elapsed time | 3.940 s | 6.104 s |
| Process CPU time | 631.197 CPU-s | 141.098 CPU-s |
| Detector NPU calls | 629 | 629 |
| Recognizer NPU calls | 1,206 | 5,208 |
| CPU verification calls | 1,958 | 0 |
| CPU wide-line calls | 752 | 0 |
| Peak process RSS | 573,664 KiB | 392,368 KiB |

CPU wide-line calls are a subset of canonical recognition checks in production, not additional checks to be added to the verification total.

The candidate reduced process CPU time by approximately 77.65%, but increased elapsed time by approximately 79.34%.

Process CPU time sums execution across threads and can exceed wall-clock time. Peak RSS does not include all kernel or accelerator-driver memory.

Removing CPU recognition increased the number of NPU calls and used a slower recognition configuration. Lower CPU use therefore did not translate into a faster application.

The candidate was rejected.

## 19. Further NPU Migration Experiments

### 19.1 Fitting near-width lines

Fitting normalized widths of 321–384 pixels into 320 pixels improved the focused probe:

```text
42/48 → 43/48 exact matches
76 → 63 NPU calls
```

The complete 74-input test still exposed a recall regression and required 585.507 seconds versus 335.943 seconds for production.

That was approximately 74.29% slower.

This experimental change was not promoted. It should not be confused with the already-existing near-width handling in the legacy runtime session.

### 19.2 Wider 640-pixel recognizer

A width-640 model retained the LayerNorm bypasses, per-channel INT8 CNN, and FLOAT16 sequence tail.

The transformed CPU model preserved argmax predictions with maximum output differences below `3.2×10⁻⁶`.

Physical NPU execution achieved only:

```text
37/48 exact frozen-crop predictions
```

compared with 42/48 for the selected narrower candidate.

It was rejected before unnecessary full-frame testing.

### 19.3 Angle classification

Three NPU angle-classifier variants were compared against CPU classification on first-pass crops from the 74 frames, including each crop rotated by 180 degrees.

Each variant received 3,824 decisions: 1,912 normal and 1,912 upside-down crops.

| Variant | Normal disagreements | Upside-down disagreements | Total |
|---|---:|---:|---:|
| Initial FLOAT16 | 57 | 1,057 | 1,114 |
| INT8 with stride rewrites | 48 | 183 | 231 |
| FLOAT16 with stride rewrites | 55 | 1,047 | 1,102 |

The INT8 version reduced measured inference time in this comparison, but its 231 disagreements remained unacceptable.

The FLOAT16 versions were both inaccurate and materially slower.

Orientation classification therefore remained on CPU.

### 19.4 Other rejected precision and fusion trials

Additional work tested:

- Larger window overlaps.
- Alternative overlap fusion and confidence choices.
- Squeezing wide lines.
- Per-channel-only precision.
- All-FLOAT16 recognition.
- FLOAT16 stem plus quantized downstream layers.
- Alternative LayerNorm precision configurations.

Some precision-boundary builds failed during compilation. Others executed but lost accuracy.

No tested additional migration met the combined accuracy, reliability, and speed requirements.

This is a conclusion about the evaluated implementations, not proof that the hardware can never support a better model or compiler strategy.

## 20. Stability and Failure Recovery

Validation included more than successful normal inference.

The retained records cover:

- One hundred repeated recognizer reference runs.
- Five repeated verified 18-line page runs.
- Five repeated verified small-line runs.
- Repeated original PAPER/SCREEN fixtures.
- Fresh-process reproduction of additional session results.
- Synthetic brightness, resizing, and JPEG variations.
- Injected NPU recognition failures.
- Invalid-image handling.
- Closed-handle handling and reconstruction.
- Checks that genuine NPU-only experiments contained no CPU recognition session.
- Production source and model checksum preservation.

Production recovered injected recognizer inference failures using canonical CPU output.

The NPU-only experiment surfaced injected errors without silently falling back to CPU, then recovered on a subsequent valid request.

Normal full-regression runs reported zero NPU execution errors. Accuracy errors still occurred.

Repeat and fault-injection tests covered their documented subsets. They should not be described as repeated exhaustive testing of every frame under every failure condition.

## 21. Integration into Visual AI

The OCR runtime was integrated into Visual AI through a backend adapter and persistent OCR worker.

Existing behavior was preserved:

- PAPER/SCREEN routing.
- Offline answering.
- OpenAI answering.
- Answer combination.
- Knowledge retrieval.
- Piper speech.
- Camera capture and alignment.
- OpenAI configuration.
- Bluetooth output.

The recognition worker selects the deployed accuracy profile by default. Detection-only alignment retains its original OCR profile to preserve its behavior.

The configuration supports:

```text
PAI_OCR_BACKEND=auto | orangepi | cpu
PAI_OCR_PROFILE=accuracy | baseline
```

The baseline option provides rollback without changing the NPU models.

### 21.1 Optional Visual AI launcher — PUBLIC-PATH ADAPTATION

The historical application launcher is `run_luna_assistant.sh`. If an optional, sanitized integration package is provided under `integration/visual_ai/`, its public-path launch example is:

```sh
cd "$HOME/orange-pi-zero-3w-rapidocr-npu/integration/visual_ai"
sh run_luna_assistant.sh
```

This is a path-adapted form of the supplied historical launch command, not a newly verified deployment. The launcher and any service definitions must be updated to discover the runtime under the public root before using it. The full Visual AI application, answering models, credentials, knowledge database, Bluetooth setup, and camera configuration are not prerequisites for the standalone OCR examples and are not claimed to be included in this repository.

The public application name is **Visual AI**. Existing source basenames and environment-variable names remain compatibility interfaces until the actual code is migrated.

### 21.2 Resident offline answering service

The offline answering server was moved into a persistent systemd user service:

```text
luna-offline-answering.service
```

With user lingering enabled, the server starts after boot without requiring an SSH login, loads its model once, remains resident between Visual AI runs, restarts after a crash, and logs to the journal.

The launcher checks server health instead of starting a second server or reloading the model.

Saved verification records report 22 lifecycle and reboot checks passing, exactly one server process, and a healthy loaded server before the verification SSH session.

The normal health gate was approximately 0.16 seconds.

This answering model remains CPU-based. Its residency improvement is separate from NPU OCR acceleration.

### 21.3 Deployment validation boundaries

Recorded deployment checks included:

- Physical OCR integration.
- Offline server readiness and a constrained answering test.
- Knowledge retrieval.
- OpenAI authentication and a synthetic request.
- Piper playback through the selected Bluetooth output.
- Camera capture and resolution transitions.
- Normal application startup reaching the persistent OCR worker.

The bounded startup test did not contain a visible question. It therefore should not be described as a completed interactive PAPER/SCREEN question-to-hybrid-answer test.

## 22. Camera SuperSpeed Recovery

The WN camera, USB identifier `1bcf:0b31`, initially enumerated at 5,000 Mbit/s but failed UVC initialization.

Observed failures included:

```text
UVC probe/control failure: -19
Device initialization failure: -5
```

A targeted delayed bind to the existing `uvcvideo` driver recovered streaming without forcing USB2 or changing camera hardware.

A subsequent cold boot exposed an additional unconfigured-device state. The recovery script was extended to restore configuration 1 for the exact SuperSpeed camera before binding UVC.

After recovery, physical capture passed:

```text
300/300 frames
1280×720
USB speed: 5000 Mbit/s
Elapsed: approximately 18.46 seconds
```

The recovery timer checks the specific device and avoids disturbing a camera that is already bound.

These are bounded capture checks after recovery, not proof that every cold-boot camera failure has been eliminated. This is a recovery mechanism for observed USB initialization states. The underlying initial-probe failure was not established as a fully isolated kernel defect.

## 23. Checkpoints, Cleanup, and Backup

Known-good runtime and model checkpoints were preserved before major experiments.

After the practical NPU migration work was rejected, obsolete generated binaries and redundant working copies were removed. Important source graphs, rewrite scripts, quantization tables, regression data, and results were retained.

Consequently, some historical experimental launch commands now require rebuilding removed experimental binaries. They should not be presented as currently runnable production commands.

The earlier standalone CPU-checked OCR archive was superseded by the maintained full project backup.

The maintained full application backup used a historical private filename. Its saved content was updated on October 4, 2026, despite that filename's earlier date. Private backup paths and filenames are not public installation targets or source-release artifacts.

Matching copies were verified on Luna and the Mac.

This is a project/application backup, including the preserved service configuration. It is not a complete operating-system disk image and does not automatically include external secrets, Bluetooth pairing state, or every system package.

## 24. Performance and Accuracy Summary

| Result | Recorded finding |
|---|---|
| Detector input | `[1,3,736,736]`, UINT8 |
| Recognizer input | `[1,3,48,320]`, INT8 |
| Recognizer output | `[1,40,18710]`, FLOAT16 |
| Reference recognizer repetition | 100/100 passed |
| Reference recognizer wrapper median | 13.364 ms |
| Reference native bridge median | 11.758 ms |
| Reference simulator/physical output | Byte-for-byte match |
| Earlier small-line configuration | Approximately 120.3 ms |
| CPU-checked small-line tests | Approximately 195–204 ms |
| Earlier verified 18-line page | Approximately 1.64–1.66 s |
| Earlier page NPU agreement | 14/18 |
| Earlier page after CPU checking | 18/18 canonical-reference matches |
| Deployed PAPER 1 word recall | 97.81% |
| Deployed PAPER 2 word recall | 99.04% |
| Deployed PAPER 3 word recall | 100% |
| Original three SCREEN frames | 100% core-text word recall |
| Additional eight PAPER frames | 96.29% weighted word recall |
| Selected NPU-only recognition, 74 inputs | Approximately 79% slower; accuracy regressions |
| Subsequent fitted NPU-only variant | Approximately 74% slower; not promoted |
| NPU angle classifier | Not promoted |
| Current production policy | NPU detection and recognition attempts, canonical CPU final recognition |

These figures describe specific models, inputs, software versions, and execution configurations. They are not universal Orange Pi benchmarks.

No broad, accuracy-preserving speedup against an otherwise identical CPU-only implementation was established across the complete 74-input set.

## 25. Recommended Method for Other Developers

A reproducible development sequence is:

1. Verify the board, driver, runtime libraries, and `/dev/vipcore`.
2. Execute a small known-good NBG before attempting OCR.
3. Preserve the original ONNX recognizer and vocabulary as references.
4. Freeze dimensions for the selected accelerator path.
5. Validate normalization, channel ordering, layout, and quantization independently.
6. Establish floating-point conversion equivalence.
7. Compare intermediate tensors and CTC sequences.
8. Isolate suspicious operator configurations.
9. Validate every graph rewrite numerically.
10. Compare simulator output with physical hardware.
11. Repeat inference and test failure recovery.
12. Evaluate full images with visible-text references.
13. Measure complete OCR latency and process CPU use.
14. Retain CPU processing when migration harms accuracy, reliability, or speed.
15. Protect production and save verified checkpoints before experimentation.

The Vivante SDK, runtime libraries, headers, and model artifacts are required dependencies. Installing the Python requirements alone does not provide the NPU execution stack.

## 26. Conclusion

RapidOCR can execute PP-OCRv6 detection and fixed-width recognition on the Orange Pi Zero 3W’s A733/VIP9000 NPU.

Achieving that required graph restructuring, quantization engineering, LayerNorm division bypasses, exact host tensor packing, a custom VIPLite bridge, simulator comparisons, and physical validation.

The deployed Visual AI system retains canonical CPU recognition because the evaluated NPU-only implementations did not preserve production accuracy and speed together.

The later full-frame accuracy profile substantially improved small-print recognition without changing the protected NPU models or Visual AI answering behavior. Remaining detection, recognition, punctuation, and capture-quality errors are documented.

The contribution is a working and tested implementation with explicit boundaries: an operational NPU OCR path, a reliable CPU-checked production policy, measurable full-frame accuracy improvements, and evidence explaining why further tested NPU migration was not deployed. These implementation and measurement claims summarize the supplied project record; the publication edit itself did not reproduce them.

## References and supporting records

External sources establish platform context, not independent confirmation of the OCR results:

- [Allwinner A733 datasheet](https://dl.radxa.com/cubie/a7a/docs/hw/datasheet/A733_Datasheet_V0.93.pdf).
- [Orange Pi Zero 3W build configuration](https://github.com/orangepi-xunlong/orangepi-build/blob/next/external/config/boards/orangepizero3w.conf).
- [A733 NPU driver project](https://github.com/petayyyy/a733_npu_driver/blob/main/README.md) and [Radxa-to-Orange-Pi deployment notes](https://github.com/petayyyy/a733_npu_driver/blob/main/docs/07-porting-radxa-to-orangepi.md).
- [Related Vivante NPU implementation](https://github.com/unnamedwild-ux/frigate_npu_vivante).

The project record identifies `selftest_result.json`, `verified_validation.json`, `pipeline_validation.json`, `verification/simulator_parity.json`, `ASSISTANT_LAYERNORM_FINDINGS_2026-10-03.txt`, `assistant-layernorm-comparison.md`, `OCR_ACCURACY_DEPLOYMENT_REPORT.md`, `FULL_NPU_EXPERIMENT_REPORT.md`, and `NPU_MIGRATION_FOLLOWUP.md` as supporting evidence. These records are not yet uploaded here. A public supplement should bind each report to the exact source revision, model hashes, fixtures, and measurement protocol it describes.

# Appendix A — Code and Artifact Inventory

**The following inventory describes sources reported as retained in the development project.** Their paths have been mapped to the generic public layout. These are not active download links, and the listed files are not yet included in this repository.

For publication, distribute a versioned, rights-cleared source supplement. Verify the source contents and path migration before turning these inventory entries into repository links.

## A.1 Production OCR runtime

| File | Purpose |
|---|---|
| `runtime/luna_ocr.py` | Complete Python VIPLite sessions, quantization, CTC comparison, verified policy, construction and CLI |
| `runtime/vip_bridge.c` | Complete native VIPLite bridge |
| `runtime/install.sh` | Environment installation, bridge compilation and validation |
| `runtime/run.sh` | Standalone OCR launcher |
| `runtime/requirements.txt` | Pinned OCR dependencies |
| `runtime/make_calibration.py` | Supplied-image calibration crop generation |
| `runtime/selftest.py` | Repeated recognizer and raw-output reference validation |
| `runtime/validate_verified.py` | Canonical equivalence and inference-failure recovery |
| `runtime/validate_pipeline.py` | Separate pipeline comparison |

The required runtime package would contain the following artifacts (not currently uploaded):

| Runtime-relative location | Required artifacts |
|---|---|
| `include/` | Compatible VIPLite headers, obtained under applicable vendor terms |
| `librapidocr_vip.so` | ARM64 bridge built from the complete source |
| `models/` | `detector.nb`, `detector.json`, `recognizer.nb`, `recognizer.json`, `canonical_recognizer.onnx`, `classifier.onnx`, `characters.txt` |
| `compiler/` | `model.json`, `model.data`, `model.quantize`, `inputmeta.yml`, `rebuild.sh` |
| `samples/` | Redistributable inputs used by the validators and examples |
| `verification/` | Reference outputs and comparison records |
| `SHA256SUMS` | Checksums tied to the released artifacts |

Model binaries, metadata, vocabulary, and quantization tables are part of the reproducibility package. Source code without them is insufficient.

## A.2 Graph rewrite sources

The reported retained folder `work/assistant_layernorm_research/` contains:

```text
rewrite_allstride2_maxpool.py
rewrite_conv5_mulsum.py
rewrite_conv40_outsplit2.py
rewrite_layernorm_mul.py

PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum_
conv40_outsplit2_padadd_lnmul_lnaffsub.onnx

PP-OCRv6_rec_small_allstride2_maxpool_conv5_mulsum_
conv40_outsplit2_padadd_lnmul_lnaffsub_ln1rsqrtmul.onnx

ASSISTANT_LAYERNORM_FINDINGS_2026-10-03.txt
```

The displayed ONNX names are wrapped for readability; their actual filenames contain no line breaks.

Additional build, precision, and diagnostic sources are retained in `work`:

```text
build_allbypass.sh
build_allvariance.sh
build_allfp16.sh
build_calibrated.sh
build_inputrange.sh
build_inputzero.sh
build_node211.sh
build_pcprefix.sh
build_pctail.sh
build_realcal.sh
build_sum.sh

rewrite_norm_means.py
make_mixed_precision.py
make_norm_precision.py
make_zero_centered.py
make_prefix_probe.py
check_checkpoint.py
check_prefix.py
compare_outputs.py
validate_rewrites.py
```

These files include successful builds and rejected experiments. Their presence does not imply every variant was deployed.

**Reconstruction boundary:** the four standalone ONNX rewrite scripts do not independently reproduce every cumulative transformation. Pad/Add reconstruction, affine restructuring, and some LayerNorm changes are preserved in cumulative graphs and compiler artifacts. Re-exporting the saved production compiler graph is supported; a fully automated upstream-ONNX-to-production rebuild additionally requires those historical transformations to be packaged explicitly.

## A.3 Deployed full-frame accuracy code

The reported accuracy package, mapped to `integration/visual_ai/luna_accuracy/`, comprises:

```text
__init__.py
automatic_candidate.py
candidate_tiled.py
augment_detection.py
gap_detection.py
recognition_retry.py
reading_order.py
```

Their responsibilities are:

| Module | Responsibility |
|---|---|
| `__init__.py` | Accuracy wrapper and explicit/default PAPER/SCREEN selection |
| `automatic_candidate.py` | Profile configuration, cached detection and conditional recovery |
| `candidate_tiled.py` | Full-frame tiled NPU detection and weighted map stitching |
| `augment_detection.py` | Additional non-overlapping undilated-mask detections |
| `gap_detection.py` | Geometry-based missing-line proposals |
| `recognition_retry.py` | Conditional internal-region recognition retries |
| `reading_order.py` | Geometry-based short-row ordering |

The deployed integration sources are:

- `integration/visual_ai/pai_ocr_backend.py`
- `integration/visual_ai/rapidocr_worker.py`

The backend selects the verified runtime and accuracy wrapper. The worker preserves the original detection-only alignment profile and maintains recognition counters.

## A.4 NPU-only experimental code

The retained `work/accuracy_results/full_npu_validation` contains the relevant implementation and test sources:

```text
experimental_ocr.py
run_npu_experiment.sh
npu_only.py
npu_profile.py
npu_context.py
npu_wide.py

build_stem_tail.sh
build_rec640.sh
build_classifier.sh
build_classifier_int8.sh

probe_models.py
probe_overlap.py
probe_confidence.py
probe_context.py
probe_wide.py
validate_classifier.py
run_regression.py
test_failure_recovery.py
compare_regression.py
compare_fitnear.py
build_report.py
write_migration_followup.py
```

Retirement and cleanup records distinguish preserved sources from removed generated binaries. These experiments require their documented artifacts to be rebuilt before rerunning.

## A.5 Accuracy and regression evidence

The project record identifies measurement and scoring sources with results at these public-layout locations:

- `work/ocr_regression_20261004`
- `work/ocr_paper2_refinement`
- `work/ocr_accuracy_priority`
- `work/ocr_additional_validation`
- `work/ocr_shifted_candidate`
- `work/accuracy_results/full_npu_validation`

The source supplement should include their input manifests, duplicate hashes, visible-text references, scoring implementations, raw per-image outputs, counters, timings, preservation checks, and comparison JSON.

Ground truth belongs to evaluation code and must not be imported into the production OCR implementation.

## Command coverage and remaining reconstruction gaps

All seven workflow examples supplied in Codex's command appendix are included above: saved recognizer re-export, native bridge compilation, standalone validation, standalone verified OCR, verified Python engine use, accuracy-wrapper use, and optional Visual AI launch. The lower-level Pegasus export command is also included. Working-directory setup has been added to make path assumptions explicit.

This is **not the complete development command history**. It does not yet include exact provisioning, original detector/recognizer imports, calibration, all cumulative graph rewrites, model installation, accuracy/regression runner invocations, experimental rebuilds, fault-injection invocations, or optional service/camera setup commands. Their named source files and reports must be supplied and inspected before those steps can be documented accurately. Some retired experiments also require rebuilding deleted generated artifacts.

Before describing this as a reproducible public release, include the real sources and a checked command sequence for each stage. Every stage should state execution location, inputs, outputs, expected verification, and limitations. Preserve the distinction between historical results, reproduction on the original environment, and a separately tested clean installation. Never present a filename inventory or code excerpt as proof that the full conversion can already be reproduced.

# Appendix B — Public Release Completion Checklist

A public release should include:

- Complete project-authored runtime, rewrite, accuracy, integration, and evaluation sources.
- Exact original and transformed model identifiers and hashes.
- Detector and recognizer NBG files and metadata, where redistribution is permitted.
- Vocabulary and canonical reference models.
- Saved ACUITY graph, weights, quantization tables, and input metadata.
- Calibration manifests and preprocessing definitions.
- Compiler image identity and SDK/runtime version information.
- Regression manifests, reference transcriptions, raw results, and scoring definitions.
- Clear separation of production, rejected experiments, and historical artifacts.
- Installation and rebuilding instructions that distinguish supplied artifacts from proprietary dependencies.

Vendor SDK libraries, model licenses, and test-image redistribution rights must be handled separately.

The evidence supports a working implementation and documented experimental results. It does not support claiming that the complete conversion is reproducible from a few command excerpts, that all OCR runs on the NPU, or that every camera frame is transcribed perfectly.
