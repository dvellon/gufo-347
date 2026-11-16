"""Regenerates the MTP-capture overlay from the current working-tree sources.

Host-only. Copies four Flash-Next sources into ``overlay/`` (same relative
paths) and applies anchored insertions/replacements. Every anchor must occur
exactly once, otherwise nothing is written. Files are handled as bytes so the
Windows-1252 engine.cpp and CRLF line endings are preserved.

    python make_overlay.py            # writes overlay/ (refuses to overwrite)
    python make_overlay.py --force    # rewrite overlay/ (after upstream drift)
    python make_overlay.py --check    # compare overlay/ with a fresh render

``overlay/BASE-SHA256.json`` records the base file hashes; ``build.ps1``
refuses to build if the working-tree bases have drifted from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OVERLAY = HERE / "overlay"
QFN = "src/models/qwen38_flash_next"

# --------------------------------------------------------------------------
# executor.hpp: extend LayerFeatureCapture (aggregate stays compatible with
# the existing `{layer_ids, features}` initializers).
# --------------------------------------------------------------------------
EXECUTOR_HPP_OLD = """  struct LayerFeatureCapture {
    std::span<const std::uint32_t> layer_ids;
    std::span<float> features;
  };
"""
EXECUTOR_HPP_NEW = """  struct LayerFeatureCapture {
    std::span<const std::uint32_t> layer_ids;
    std::span<float> features;
    // MTP-drafter capture overlay (drafter-training/capture).
    // Requires empty layer_ids. `final_residual`
    // receives the raw final wide residual [token][hc_count][hidden] F32 of
    // every row: exactly the trunk rows the MTP block consumes as
    // target_hidden. With head_rows_per_call > 0, `head_sink` receives the
    // full-vocabulary target logits of rows [head_first_row, n) in calls of
    // at most head_rows_per_call rows (clamped to the staging capacity),
    // through the production hc_head mixer and output() projection. Calls
    // of <= 8 rows use the decode/verify head arithmetic exactly; wider calls
    // use the wide GEMM route.
    std::span<float> final_residual{};
    std::uint32_t head_first_row{0};
    std::uint32_t head_rows_per_call{0};
    std::function<bool(std::uint32_t first_row, std::uint32_t rows,
                       const float* logits, std::string* error_msg)>
        head_sink{};
    // DFlash2 feature mode (same overlay). With a tap_sink, the layer_ids
    // taps are NOT reduced by hc_head: each tap's raw wide residual rows
    // ([n][hc_count][hidden] F32, after the completed layer and before the
    // next layer's PLE) are handed to tap_sink, one tap at a time, so the
    // caller can apply llama.cpp's plain stream mean. `features` must be
    // empty in this mode. May be combined with final_residual/head.
    std::function<bool(std::uint32_t tap, std::uint32_t rows,
                       const float* raw_rows, std::string* error_msg)>
        tap_sink{};
  };
"""

# --------------------------------------------------------------------------
# executor.cpp: validation accepts the MTP mode (no layer taps).
# --------------------------------------------------------------------------
EXECUTOR_CPP_VALIDATE_OLD = """    if (checkpoint != nullptr || ids.empty() ||
        ids.size() > kMaxFlashMapLayerTaps ||
        !std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end() ||
        ids.back() >= c.num_layers || capture->features.size() != expected) {
"""
EXECUTOR_CPP_VALIDATE_NEW = """    // Wave3 MTP/DFlash capture overlay. Modes: FlashMap (taps reduced by
    // hc_head into `features`), MTP (no taps; final residual and/or head),
    // DFlash (taps handed raw to tap_sink; may add final residual/head).
    const bool residual_out = !capture->final_residual.empty();
    const bool head_out = capture->head_rows_per_call != 0;
    const bool tap_out = static_cast<bool>(capture->tap_sink);
    const bool extended = residual_out || head_out || tap_out;
    if (checkpoint != nullptr ||
        (ids.empty() ? !(residual_out || head_out) : extended && !tap_out) ||
        (tap_out && ids.empty()) ||
        ids.size() > kMaxFlashMapLayerTaps ||
        !std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end() ||
        (!ids.empty() && ids.back() >= c.num_layers) ||
        capture->features.size() != (tap_out ? 0 : expected) ||
        (extended && c.HcDim() > kMaxFlashMapLayerTaps * c.hidden_size) ||
        (residual_out && capture->final_residual.size() !=
                             static_cast<std::size_t>(n) * c.HcDim()) ||
        (head_out && !capture->head_sink)) {
"""

# --------------------------------------------------------------------------
# executor.cpp: the FlashMap hc_head readout is skipped in DFlash mode.
# --------------------------------------------------------------------------
EXECUTOR_CPP_FLASHMAP_OLD = """  if (capture != nullptr) {
    for (std::size_t tap = 0; tap < capture->layer_ids.size(); ++tap) {
      q8t_src_ = nullptr;
"""
EXECUTOR_CPP_FLASHMAP_NEW = """  if (capture != nullptr && !capture->tap_sink) {
    for (std::size_t tap = 0; tap < capture->layer_ids.size(); ++tap) {
      q8t_src_ = nullptr;
"""

# --------------------------------------------------------------------------
# executor.cpp: MTP readout at the end of ForwardBody (after the FlashMap
# readout, before `return true;`).
# --------------------------------------------------------------------------
EXECUTOR_CPP_READOUT_ANCHOR = """  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
"""
EXECUTOR_CPP_READOUT_NEW = """  // Wave3 MTP-drafter capture overlay. s_.res still holds the final wide
  // residual of every row: the target pass, persistent caches and requested
  // logits are complete, and the FlashMap readout above rewrites only
  // s_.mixed. Without layer taps both staging buffers are free: the pinned
  // host buffer (kMaxFlashMapLayerTaps * max_batch * hidden floats) holds the
  // residual rows and then each head call's logits, and the device snapshot
  // buffer (kMaxFlashMapLayerTaps * max_batch * HcDim floats) holds the
  // head's logits on the device.
  // DFlash taps first: the raw tap rows still occupy the device snapshot
  // buffer, which the head below reuses. One tap (n * HcDim floats <=
  // the host staging capacity) per download.
  if (capture != nullptr && capture->tap_sink) {
    const std::size_t tap_floats = static_cast<std::size_t>(n) * c.HcDim();
    for (std::size_t tap = 0; tap < capture->layer_ids.size(); ++tap) {
      if (!CopyPinned(layer_capture_device_ + tap * tap_floats,
                      layer_capture_host_, tap_floats * sizeof(float),
                      hipMemcpyDeviceToHost, stream_,
                      "DFlash capture tap download", error_msg) ||
          !Check(hipStreamSynchronize(stream_), "DFlash capture tap",
                 error_msg)) {
        return false;
      }
      if (!capture->tap_sink(static_cast<std::uint32_t>(tap), n,
                             layer_capture_host_, error_msg))
        return false;
    }
  }
  if (capture != nullptr && (!capture->final_residual.empty() ||
                             capture->head_rows_per_call != 0)) {
    const std::size_t hc_dim = c.HcDim();
    if (!capture->final_residual.empty()) {
      if (!CopyPinned(s_.res, layer_capture_host_,
                      static_cast<std::size_t>(n) * hc_dim * sizeof(float),
                      hipMemcpyDeviceToHost, stream_,
                      "MTP capture residual download", error_msg) ||
          !Check(hipStreamSynchronize(stream_), "MTP capture residual",
                 error_msg)) {
        return false;
      }
      std::copy_n(layer_capture_host_, static_cast<std::size_t>(n) * hc_dim,
                  capture->final_residual.begin());
    }
    if (capture->head_rows_per_call != 0 && capture->head_first_row < n) {
      const std::size_t staged_rows =
          kMaxFlashMapLayerTaps * static_cast<std::size_t>(options_.max_batch) *
          c.hidden_size / c.vocab_size;
      const auto per_call = static_cast<std::uint32_t>(std::min<std::size_t>(
          {static_cast<std::size_t>(capture->head_rows_per_call),
           static_cast<std::size_t>(options_.max_batch), staged_rows}));
      if (per_call == 0) {
        AssignError(error_msg, "MTP capture head staging is too small");
        return false;
      }
      // The production head runs outside the prefill policy; keep that so
      // calls of <= kVecBatch rows reproduce its arithmetic exactly.
      PrefillPhase head_phase(false);
      for (std::uint32_t r0 = capture->head_first_row; r0 < n;
           r0 += per_call) {
        const std::uint32_t rows = std::min(per_call, n - r0);
        q8t_src_ = nullptr;
        half_src_ = nullptr;
        DropReadyQ8();
        xn_half_ = false;
        inject_fused_ = nullptr;
        if (!HcMix(model_->hc_head(),
                   s_.res + static_cast<std::size_t>(r0) * hc_dim, false,
                   s_.mixed, nullptr, rows, error_msg) ||
            !Dense(model_->output(), s_.mixed, layer_capture_device_, rows,
                   error_msg) ||
            !CopyPinned(layer_capture_device_, layer_capture_host_,
                        static_cast<std::size_t>(rows) * c.vocab_size *
                            sizeof(float),
                        hipMemcpyDeviceToHost, stream_,
                        "MTP capture logits download", error_msg) ||
            !Check(hipStreamSynchronize(stream_), "MTP capture head",
                   error_msg)) {
          return false;
        }
        if (!capture->head_sink(r0, rows, layer_capture_host_, error_msg))
          return false;
      }
      q8t_src_ = nullptr;
      half_src_ = nullptr;
      DropReadyQ8();
      xn_half_ = false;
      inject_fused_ = nullptr;
    }
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
"""

# --------------------------------------------------------------------------
# engine.hpp: request type + method (no data members: Model layout unchanged).
# --------------------------------------------------------------------------
ENGINE_HPP_ANCHOR = """      std::string* error_msg = nullptr,
      std::vector<float>* final_logits = nullptr);
"""
ENGINE_HPP_NEW = """      std::string* error_msg = nullptr,
      std::vector<float>* final_logits = nullptr);
  /// Wave3 MTP-drafter capture overlay: a fresh target-only prefill that
  /// hands every row's raw final wide residual (the MTP target_hidden input,
  /// HcDim F32 per token, [hc_count][hidden]) to `residual_sink` and, when
  /// head_rows_per_call > 0, full-vocabulary target logits of rows >=
  /// head_first_row to `head_sink` (row indices are absolute). Head calls
  /// for a chunk precede that chunk's residual call. Offline training data
  /// only; never entered by inference.
  struct MtpCaptureRequest {
    std::uint32_t head_first_row{0};
    std::uint32_t head_rows_per_call{0};  ///< 0 disables the head
    std::function<bool(std::uint32_t first_row, std::uint32_t rows,
                       const float* residual, std::string* error_msg)>
        residual_sink;
    std::function<bool(std::uint32_t first_row, std::uint32_t rows,
                       const float* logits, std::string* error_msg)>
        head_sink;
    /// Optional production single-row head (n_logits = 1) of the last token.
    std::vector<float>* final_logits{nullptr};
    /// DFlash2 mode: zero-based completed layers whose raw wide residual
    /// rows ([rows][hc_count][hidden] F32, before the next layer's PLE) go
    /// to tap_sink (tap index into tap_layers, absolute first row). Tap
    /// calls for a chunk precede its head and residual calls. With
    /// tap_layers set, residual_sink is optional.
    std::vector<std::uint32_t> tap_layers;
    std::function<bool(std::uint32_t tap, std::uint32_t first_row,
                       std::uint32_t rows, const float* raw_rows,
                       std::string* error_msg)>
        tap_sink;
  };
  [[nodiscard]] bool CaptureMtpTrace(std::span<const std::int32_t> tokens,
                                     const MtpCaptureRequest& request,
                                     std::string* error_msg = nullptr);
"""

# --------------------------------------------------------------------------
# engine.cpp: implementation, mirroring CaptureFlashMapFeatures' chunking.
# --------------------------------------------------------------------------
ENGINE_CPP_ANCHOR = """std::size_t Session::AllocatedBytes() const noexcept {
"""
ENGINE_CPP_NEW = """bool Model::CaptureMtpTrace(std::span<const std::int32_t> tokens,
                            const MtpCaptureRequest& request,
                            std::string* error_msg) {
  // Wave3 MTP/DFlash capture overlay. Same session, chunking and prefill
  // arithmetic as CaptureFlashMapFeatures.
  const bool taps = !request.tap_layers.empty();
  if (tokens.empty() || tokens.size() > options_.max_context ||
      (!request.residual_sink && !taps) ||
      taps != static_cast<bool>(request.tap_sink) ||
      (request.head_rows_per_call != 0 && !request.head_sink)) {
    AssignError(error_msg,
                "MTP capture needs a bounded non-empty prompt and sinks");
    return false;
  }
  if (request.final_logits != nullptr)
    request.final_logits->clear();
  auto state = executor_->CreateSession(core::SessionMode::kAutoregressive,
                                        static_cast<std::uint32_t>(tokens.size()),
                                        error_msg);
  if (!state) {
    return false;
  }
  if (vision_) {
    try {
      state->PrepareVision(executor_->stream());
    } catch (const std::exception& exception) {
      AssignError(error_msg, exception.what());
      return false;
    }
  }
  const std::size_t hc_dim = config().HcDim();
  std::vector<float> residual;
  for (std::size_t offset = 0; offset < tokens.size();) {
    const auto remaining = tokens.size() - offset;
    const auto capacity = remaining <= executor_->max_batch()
                              ? executor_->max_batch()
                              : PrefillCapacity();
    const std::size_t count = std::min<std::size_t>(capacity, remaining);
    const auto chunk = tokens.subspan(offset, count);
    const auto rest = tokens.subspan(offset + count);
    const auto next = rest.first(std::min<std::size_t>(
        rest.size() <= executor_->max_batch() ? executor_->max_batch()
                                               : PrefillCapacity(),
        rest.size()));
    const auto base = static_cast<std::uint32_t>(offset);
    rocm::Executor::LayerFeatureCapture capture{};
    if (request.residual_sink) {
      residual.resize(count * hc_dim);
      capture.final_residual = residual;
    }
    if (taps) {
      capture.layer_ids = request.tap_layers;
      capture.tap_sink = [&request, base](std::uint32_t tap,
                                          std::uint32_t rows,
                                          const float* raw_rows,
                                          std::string* sink_error) {
        return request.tap_sink(tap, base, rows, raw_rows, sink_error);
      };
    }
    capture.head_first_row =
        request.head_first_row > base ? request.head_first_row - base : 0;
    capture.head_rows_per_call = request.head_rows_per_call;
    if (request.head_rows_per_call != 0) {
      capture.head_sink = [&request, base](std::uint32_t first_row,
                                           std::uint32_t rows,
                                           const float* logits,
                                           std::string* sink_error) {
        return request.head_sink(base + first_row, rows, logits, sink_error);
      };
    }
    const bool final_chunk = offset + count == tokens.size();
    std::vector<float> logits;
    if (final_chunk && request.final_logits != nullptr)
      logits.resize(config().vocab_size);
    if (!executor_->Forward(*state, chunk, logits.empty() ? 0 : 1,
                            logits.empty() ? nullptr : logits.data(),
                            rocm::Executor::ForwardMode::kPrefill, error_msg,
                            nullptr, next, &capture)) {
      return false;
    }
    if (!logits.empty())
      *request.final_logits = std::move(logits);
    if (request.residual_sink &&
        !request.residual_sink(base, static_cast<std::uint32_t>(count),
                               residual.data(), error_msg)) {
      return false;
    }
    offset += count;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
"""

PATCHES = {
    f"{QFN}/kernels/rocm/executor.hpp": [("replace", EXECUTOR_HPP_OLD, EXECUTOR_HPP_NEW)],
    f"{QFN}/kernels/rocm/executor.cpp": [
        ("replace", EXECUTOR_CPP_VALIDATE_OLD, EXECUTOR_CPP_VALIDATE_NEW),
        ("replace", EXECUTOR_CPP_FLASHMAP_OLD, EXECUTOR_CPP_FLASHMAP_NEW),
        ("replace", EXECUTOR_CPP_READOUT_ANCHOR, EXECUTOR_CPP_READOUT_NEW),
    ],
    f"{QFN}/engine.hpp": [("replace", ENGINE_HPP_ANCHOR, ENGINE_HPP_NEW)],
    f"{QFN}/engine.cpp": [("replace", ENGINE_CPP_ANCHOR, ENGINE_CPP_NEW)],
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render() -> tuple[dict[str, bytes], dict[str, str]]:
    outputs: dict[str, bytes] = {}
    bases: dict[str, str] = {}
    for rel, patches in PATCHES.items():
        data = (REPO / rel).read_bytes()
        bases[rel] = sha256(data)
        crlf = b"\r\n" in data
        for kind, old, new in patches:
            assert kind == "replace"
            old_b = old.encode("ascii")
            new_b = new.encode("ascii")
            if crlf:
                old_b = old_b.replace(b"\n", b"\r\n")
                new_b = new_b.replace(b"\n", b"\r\n")
            count = data.count(old_b)
            if count != 1:
                raise SystemExit(f"anchor occurs {count} times in {rel}: {old.splitlines()[0]!r}")
            data = data.replace(old_b, new_b)
        outputs[rel] = data
    return outputs, bases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, bases = render()
    if args.check:
        ok = True
        for rel, data in outputs.items():
            path = OVERLAY / rel
            same = path.exists() and path.read_bytes() == data
            ok &= same
            print(f"{'OK  ' if same else 'DIFF'} {rel}")
        recorded = json.loads((OVERLAY / "BASE-SHA256.json").read_text())
        for rel, digest in bases.items():
            if recorded.get("bases", {}).get(rel) != digest:
                ok = False
                print(f"BASE DRIFT {rel}")
        tool = sha256((HERE / "mtp_capture.cpp").read_bytes())
        if recorded.get("tool_sha256", {}).get("mtp_capture.cpp") != tool:
            ok = False
            print("TOOL DRIFT mtp_capture.cpp (rerun make_overlay.py --force after review)")
        return 0 if ok else 1
    if OVERLAY.exists() and not args.force:
        raise SystemExit("overlay/ exists; pass --force to regenerate")
    for rel, data in outputs.items():
        path = OVERLAY / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "format": "gufo.wave3.mtp-capture.overlay.v1",
        "generator": "make_overlay.py",
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "bases": bases,
        "overlay": {rel: sha256(data) for rel, data in outputs.items()},
        "tool_sha256": {"mtp_capture.cpp": sha256((HERE / "mtp_capture.cpp").read_bytes())},
    }
    (OVERLAY / "BASE-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for rel, digest in manifest["overlay"].items():
        print(f"{digest}  overlay/{rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
