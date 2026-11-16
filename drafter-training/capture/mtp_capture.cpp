// Wave3 MTP-drafter teacher-forced capture (build target w3mtp_capture).
//
// Runs Qwen3.8-Flash-Next teacher-forced over token records and writes, per
// position, the inputs and targets needed to train/evaluate the shared MTP
// drafter offline:
//   * the input token ids,
//   * the raw final trunk wide residual (HcDim = hc_count * hidden floats,
//     stored BF16, or F32 with --hidden-dtype f32 for evaluation subsets) --
//     the exact rows the MTP block reads as target_hidden,
//   * target-head supervision for rows >= supervise_from: top-k ids and
//     log-probs (F16), log-sum-exp, entropy, the next token's log-prob and
//     rank.
// Optional DFlash2 features (--features dflash|both): the plain mean of the
// hc_count residual streams after completed layers 5, 15, 25, 35, 45, stored
// BF16 [token][tap][hidden] -- llama.cpp's Qwen4Exp DFlash extraction.
// One job (JSONL file) becomes one sharded .mtps file plus a .mtps.json
// sidecar, published by content hash. Backpressure (pending shard count and
// free space) bounds the local staging directory; ship_shards.py moves
// published shards to shared storage. Format: README.md ("GMTP v1").
//
// Derived from tools/qwen-flash/flashmap_capture.cpp (record framing, durable
// publication); the model side is overlay Model::CaptureMtpTrace.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <hip/hip_runtime_api.h>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

static_assert(std::endian::native == std::endian::little,
              "GMTP writer assumes a little-endian host");

constexpr std::array<std::uint8_t, 8> kFileMagic{'G', 'M', 'T', 'P',
                                                 'C', 'A', 'P', '1'};
constexpr std::array<std::uint8_t, 8> kRecordMagic{'G', 'M', 'T', 'P',
                                                   'R', 'E', 'C', '1'};
constexpr std::array<std::uint8_t, 8> kCommitMagic{'G', 'M', 'T', 'P',
                                                   'E', 'N', 'D', '1'};
constexpr std::string_view kIdentityScheme = "gguf-sha256-v1";
constexpr std::size_t kFileHeaderBytes = 512;
constexpr std::size_t kHeaderDigestOffset = 480;
constexpr std::size_t kRecordHeaderBytes = 192;
constexpr std::size_t kRecordTrailerBytes = 64;
constexpr std::size_t kAlignment = 64;
constexpr std::size_t kMaxMetadataBytes = 64 * 1024;
constexpr std::size_t kMaxJsonLineBytes = 64 * 1024 * 1024;
constexpr std::uint32_t kMaxTopK = 64;
constexpr std::uint32_t kNoNext = 0xFFFFFFFFu;
constexpr std::uint32_t kFlagHeadChecked = 1;
// DFlash2 extraction points for this target, as zero-based completed layers.
// The DFlash GGUF stores target_layers = [6,16,26,36,46], which are layer-INPUT
// indices (llama.cpp conversion/qwen.py:834-835 writes target_layer_ids + 1).
// llama.cpp reads t_layer_inp[il] = hc_stream_mean(res_hc) at the top of
// layer il, before that layer's PLE (src/models/qwen4exp.cpp:312-323,
// 464-471; commit d9ee007a, PR #30927): the output of completed layer il-1.
// gufo's loader applies the same N-1 (src/models/qwen/dflash_weights.cpp
// 257-288), and gufo's taps are completed-layer snapshots of s_.res taken
// before the next layer's PLE.
constexpr std::array<std::uint32_t, 5> kDflashTaps{5, 15, 25, 35, 45};

// ------------------------------------------------------------------ bytes

void PutU16(std::span<std::uint8_t> out, std::size_t offset,
            std::uint16_t value) {
  if (offset + 2 > out.size())
    throw std::logic_error("U16 write exceeds buffer");
  out[offset] = static_cast<std::uint8_t>(value);
  out[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

void PutU32(std::span<std::uint8_t> out, std::size_t offset,
            std::uint32_t value) {
  if (offset + 4 > out.size())
    throw std::logic_error("U32 write exceeds buffer");
  for (unsigned shift = 0; shift != 32; shift += 8)
    out[offset++] = static_cast<std::uint8_t>(value >> shift);
}

void PutU64(std::span<std::uint8_t> out, std::size_t offset,
            std::uint64_t value) {
  if (offset + 8 > out.size())
    throw std::logic_error("U64 write exceeds buffer");
  for (unsigned shift = 0; shift != 64; shift += 8)
    out[offset++] = static_cast<std::uint8_t>(value >> shift);
}

inline void StoreU16(std::uint8_t* out, std::uint16_t value) {
  std::memcpy(out, &value, sizeof(value));
}
inline void StoreU32(std::uint8_t* out, std::uint32_t value) {
  std::memcpy(out, &value, sizeof(value));
}
inline void StoreF32(std::uint8_t* out, float value) {
  std::memcpy(out, &value, sizeof(value));
}

std::array<std::uint8_t, 32> HashBytes(std::span<const std::uint8_t> bytes) {
  gufo::crypto::Sha256Hasher hash;
  hash.Update(bytes);
  return hash.Finish();
}

std::array<std::uint8_t, 32> HashString(std::string_view value) {
  return HashBytes(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

int HexDigit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

template <std::size_t N>
std::array<std::uint8_t, N> DecodeHex(std::string_view value,
                                      std::string_view label) {
  if (value.size() != N * 2)
    throw std::runtime_error(std::string(label) + " has the wrong length");
  std::array<std::uint8_t, N> result{};
  for (std::size_t index = 0; index < N; ++index) {
    const int high = HexDigit(value[index * 2]);
    const int low = HexDigit(value[index * 2 + 1]);
    if (high < 0 || low < 0)
      throw std::runtime_error(std::string(label) + " is not hexadecimal");
    result[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return result;
}

std::string Hex(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result(bytes.size() * 2, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result[index * 2] = digits[bytes[index] >> 4];
    result[index * 2 + 1] = digits[bytes[index] & 15];
  }
  return result;
}

std::size_t Align(std::size_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

// ------------------------------------------------------------ conversions

bool ToBf16(float value, std::uint16_t* out) {
  if (!std::isfinite(value))
    return false;
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  const auto result = static_cast<std::uint16_t>(rounded >> 16);
  if ((result & 0x7F80u) == 0x7F80u)
    return false;
  *out = result;
  return true;
}

/// IEEE binary16, round to nearest even; overflow -> inf, NaN -> qNaN.
std::uint16_t ToF16(float value) {
  const std::uint32_t x = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t abs = x & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u)
    return static_cast<std::uint16_t>(sign | (abs > 0x7F800000u ? 0x7E00u
                                                                 : 0x7C00u));
  if (abs >= 0x477FF000u)  // >= 65520 rounds to infinity
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  if (abs < 0x38800000u) {  // below the smallest normal half (2^-14)
    if (abs < 0x33000000u)  // <= 2^-25 rounds to zero
      return sign;
    const std::uint32_t exponent = abs >> 23;
    const std::uint32_t mantissa = (abs & 0x7FFFFFu) | 0x800000u;
    const std::uint32_t shift = 126u - exponent;  // 14..24
    std::uint32_t result = mantissa >> shift;
    const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1u);
    if (rest > half || (rest == half && (result & 1u) != 0))
      ++result;
    return static_cast<std::uint16_t>(sign | result);
  }
  std::uint32_t result = abs - 0x38000000u;  // rebias 127 -> 15
  result += 0xFFFu + ((result >> 13) & 1u);
  return static_cast<std::uint16_t>(sign | (result >> 13));
}

/// e^x for finite x <= 0 (relative error < 1e-5). Inputs below -87 clamp.
inline float FastExpNonPositive(float x) {
  x = x < -87.0F ? -87.0F : x;
  const float t = x * 1.44269504088896341F;
  const int whole = static_cast<int>(t);  // toward zero: whole >= t
  const float f = t - static_cast<float>(whole);  // (-1, 0]
  const float u = f * 0.693147180559945309F;
  float p = 1.0F / 5040.0F;
  p = p * u + 1.0F / 720.0F;
  p = p * u + 1.0F / 120.0F;
  p = p * u + 1.0F / 24.0F;
  p = p * u + 1.0F / 6.0F;
  p = p * u + 0.5F;
  p = p * u + 1.0F;
  p = p * u + 1.0F;
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(p) +
                             (static_cast<std::uint32_t>(whole) << 23);
  return std::bit_cast<float>(bits);
}

std::uint32_t SupRowBytes(std::uint32_t topk) {
  return topk == 0 ? 0 : (topk * 6 + 16 + 15) / 16 * 16;
}

std::string AbiText(std::uint32_t hidden, std::uint32_t hc_count,
                    std::uint32_t vocab, std::uint32_t topk, bool hidden_f32,
                    bool mtp_hidden = true, bool dflash = false) {
  std::string dflash_text;
  if (dflash) {
    dflash_text =
        "dflash-taps=5,15,25,35,45\n"
        "dflash-tap-point=raw-residual-after-completed-layer-before-next-ple\n"
        "dflash-reduction=plain-mean-of-hc-streams:((s0+s1)+s2)+s3)*(1/hc)\n"
        "dflash-storage=bf16-rne-le\n"
        "dflash-layout=token,tap,hidden\n";
  }
  return std::string(
             "gufo.mtp-capture.abi.v1\n"
             "architecture=qwen4exp\n"
             "hidden-source=trunk-final-wide-residual-pre-hc-head\n"
             "hidden-layout=token,hc-stream,hidden\n") +
         (!mtp_hidden  ? "hidden-storage=none\n"
          : hidden_f32 ? "hidden-storage=f32-le\n"
                       : "hidden-storage=bf16-rne-le\n") +
         "alignment=row-i-after-input-token-i\n"
         "mtp-pairing=hidden-row-i+embed(token-i+1)->token-i+2\n"
         "supervision=target-head(hc_head,output)-row-i->token-i+1\n"
         "topk-order=logit-desc,id-asc\n"
         "topk-ids=u32-le\n"
         "topk-logprob=f16-rne-le\n"
         "row-scalars=lse-f32,entropy-nats-f32,next-logprob-f32,next-rank-u32\n"
         "hidden=" + std::to_string(hidden) + "\n" +
         "hc-count=" + std::to_string(hc_count) + "\n" +
         "vocab=" + std::to_string(vocab) + "\n" +
         "topk=" + std::to_string(topk) + "\n" + dflash_text;
}

// ----------------------------------------------------------- row reduction

struct RowSummary {
  std::uint32_t argmax{};
  float lse{};
};

/// Reduces one full-vocabulary logit row into a supervision row. Returns
/// false for a non-finite logit. Never throws except std::bad_alloc.
bool ReduceRow(const float* x, std::uint32_t vocab, std::int32_t next,
               std::uint32_t topk, std::uint8_t* out, RowSummary* summary) {
  float max_value = -std::numeric_limits<float>::infinity();
  bool finite = true;
  for (std::uint32_t i = 0; i < vocab; ++i) {
    const float v = x[i];
    finite &= std::isfinite(v);
    max_value = v > max_value ? v : max_value;
  }
  if (!finite || vocab == 0)
    return false;
  float sum_lanes[8] = {};
  float weighted_lanes[8] = {};
  std::uint32_t i = 0;
  for (; i + 8 <= vocab; i += 8) {
    for (int lane = 0; lane < 8; ++lane) {
      const float d = x[i + lane] - max_value;
      const float e = FastExpNonPositive(d);
      sum_lanes[lane] += e;
      weighted_lanes[lane] += e * d;
    }
  }
  for (; i < vocab; ++i) {
    const float d = x[i] - max_value;
    const float e = FastExpNonPositive(d);
    sum_lanes[0] += e;
    weighted_lanes[0] += e * d;
  }
  double sum = 0.0;
  double weighted = 0.0;
  for (int lane = 0; lane < 8; ++lane) {
    sum += sum_lanes[lane];
    weighted += weighted_lanes[lane];
  }
  const double log_sum = std::log(sum);
  const auto lse = static_cast<float>(max_value + log_sum);
  const auto entropy = static_cast<float>(log_sum - weighted / sum);

  // Top-k in one pass. A later index never displaces an equal value, so ties
  // keep the lower id. `worse(a, b)`: a ranks below b.
  using Entry = std::pair<float, std::uint32_t>;
  thread_local std::vector<Entry> best;
  best.clear();
  const std::uint32_t k = std::min(topk, vocab);
  const auto worse = [](const Entry& a, const Entry& b) {
    return a.first < b.first || (a.first == b.first && a.second > b.second);
  };
  for (std::uint32_t j = 0; j < k; ++j)
    best.emplace_back(x[j], j);
  const auto find_worst = [&]() {
    std::size_t worst = 0;
    for (std::size_t j = 1; j < best.size(); ++j)
      if (worse(best[j], best[worst]))
        worst = j;
    return worst;
  };
  std::size_t worst = find_worst();
  float threshold = best[worst].first;
  for (std::uint32_t j = k; j < vocab; ++j) {
    if (x[j] > threshold) {
      best[worst] = {x[j], j};
      worst = find_worst();
      threshold = best[worst].first;
    }
  }
  std::sort(best.begin(), best.end(),
            [&](const Entry& a, const Entry& b) { return worse(b, a); });

  std::memset(out, 0, SupRowBytes(topk));
  for (std::uint32_t j = 0; j < topk; ++j) {
    const bool present = j < best.size();
    StoreU32(out + 4 * j, present ? best[j].second : kNoNext);
    StoreU16(out + 4 * topk + 2 * j,
             ToF16(present ? best[j].first - lse
                           : -std::numeric_limits<float>::infinity()));
  }
  std::uint8_t* scalars = out + 6 * topk;
  StoreF32(scalars, lse);
  StoreF32(scalars + 4, entropy);
  if (next >= 0 && static_cast<std::uint32_t>(next) < vocab) {
    const float target = x[next];
    std::uint32_t rank = 0;
    for (std::uint32_t j = 0; j < vocab; ++j)
      rank += x[j] > target ? 1u : 0u;
    StoreF32(scalars + 8, target - lse);
    StoreU32(scalars + 12, rank);
  } else {
    StoreF32(scalars + 8, std::numeric_limits<float>::quiet_NaN());
    StoreU32(scalars + 12, kNoNext);
  }
  if (summary != nullptr)
    *summary = {best.front().second, lse};
  return true;
}

template <class F>
void ParallelFor(std::uint32_t count, unsigned threads, const F& body) {
  const unsigned workers =
      static_cast<unsigned>(std::min<std::uint64_t>(threads, count));
  if (workers <= 1) {
    for (std::uint32_t index = 0; index < count; ++index)
      body(index);
    return;
  }
  std::atomic<std::uint32_t> next{0};
  const auto work = [&]() {
    for (;;) {
      const std::uint32_t index = next.fetch_add(1);
      if (index >= count)
        return;
      body(index);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (unsigned worker = 1; worker < workers; ++worker)
    pool.emplace_back(work);
  work();
  for (auto& thread : pool)
    thread.join();
}

// ------------------------------------------------------------- file output

std::filesystem::path ExecutablePath() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                          static_cast<DWORD>(buffer.size()));
  if (length == 0 || length == buffer.size())
    throw std::runtime_error("cannot resolve producer executable");
  buffer.resize(length);
  return std::filesystem::path(buffer);
#else
  std::error_code error;
  const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || path.empty())
    throw std::runtime_error("cannot resolve producer executable");
  return path;
#endif
}

class TraceSink {
public:
  explicit TraceSink(const std::filesystem::path& path) {
#ifdef _WIN32
    int descriptor = -1;
    const errno_t status = _wsopen_s(&descriptor, path.c_str(),
                                     _O_BINARY | _O_CREAT | _O_EXCL | _O_WRONLY,
                                     _SH_DENYRW, _S_IREAD | _S_IWRITE);
    if (status != 0 || descriptor < 0)
      throw std::runtime_error("cannot exclusively create partial shard");
    file_ = _fdopen(descriptor, "wb");
    if (file_ == nullptr) {
      _close(descriptor);
      throw std::runtime_error("cannot open partial shard stream");
    }
#else
    const int descriptor =
        ::open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0)
      throw std::runtime_error("cannot exclusively create partial shard");
    file_ = ::fdopen(descriptor, "wb");
    if (file_ == nullptr) {
      ::close(descriptor);
      throw std::runtime_error("cannot open partial shard stream");
    }
#endif
    std::setvbuf(file_, nullptr, _IOFBF, 8 << 20);
  }
  TraceSink(const TraceSink&) = delete;
  TraceSink& operator=(const TraceSink&) = delete;
  ~TraceSink() {
    if (file_ != nullptr)
      std::fclose(file_);
  }

  void Write(std::span<const std::uint8_t> bytes,
             gufo::crypto::Sha256Hasher* record = nullptr) {
    std::size_t written = 0;
    while (written < bytes.size()) {
      const std::size_t count =
          std::fwrite(bytes.data() + written, 1, bytes.size() - written, file_);
      if (count == 0)
        throw std::runtime_error("shard write failed");
      written += count;
    }
    file_hash_.Update(bytes);
    if (record != nullptr)
      record->Update(bytes);
    bytes_ += bytes.size();
  }

  void WriteZeros(std::size_t count, gufo::crypto::Sha256Hasher* record) {
    constexpr std::array<std::uint8_t, kAlignment> zeros{};
    while (count != 0) {
      const std::size_t chunk = std::min(count, zeros.size());
      Write(std::span<const std::uint8_t>(zeros).first(chunk), record);
      count -= chunk;
    }
  }

  void Durable() {
    if (std::fflush(file_) != 0)
      throw std::runtime_error("shard stdio flush failed");
#ifdef _WIN32
    if (_commit(_fileno(file_)) != 0)
      throw std::runtime_error("shard durable commit failed");
#else
    if (::fsync(::fileno(file_)) != 0)
      throw std::runtime_error("shard durable commit failed");
#endif
  }

  std::array<std::uint8_t, 32> FinishHash() { return file_hash_.Finish(); }
  std::uint64_t bytes() const { return bytes_; }

  void Close() {
    Durable();
    if (std::fclose(file_) != 0) {
      file_ = nullptr;
      throw std::runtime_error("shard close failed");
    }
    file_ = nullptr;
  }

private:
  std::FILE* file_{nullptr};
  gufo::crypto::Sha256Hasher file_hash_;
  std::uint64_t bytes_{0};
};

void PublishNoReplace(const std::filesystem::path& partial,
                      const std::filesystem::path& output) {
#ifdef _WIN32
  if (!MoveFileExW(partial.c_str(), output.c_str(), MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("non-overwriting publication failed: " +
                             output.string());
#else
  if (::link(partial.c_str(), output.c_str()) != 0 ||
      ::unlink(partial.c_str()) != 0)
    throw std::runtime_error("non-overwriting publication failed: " +
                             output.string());
#endif
}

void WriteTextNoReplace(const std::filesystem::path& output,
                        const std::string& text) {
  auto partial = output;
  partial += ".partial";
  {
    std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    if (!stream)
      throw std::runtime_error("cannot write " + partial.string());
  }
  PublishNoReplace(partial, output);
}

std::string FinalizeHipRuntime(bool* initialized) {
  if (!*initialized)
    return {};
  *initialized = false;
#ifdef _WIN32
  std::string error;
  if (const hipError_t status = hipDeviceSynchronize(); status != hipSuccess)
    error = std::string("HIP final synchronization failed: ") +
            hipGetErrorString(status);
  if (const hipError_t status = hipDeviceReset(); status != hipSuccess) {
    if (!error.empty())
      error += "; ";
    error +=
        std::string("HIP device reset failed: ") + hipGetErrorString(status);
  }
  return error;
#else
  return {};
#endif
}

std::array<std::uint8_t, 16> MakeUuid() {
  std::array<std::uint8_t, 16> result{};
  std::random_device random;
  for (auto& byte : result)
    byte = static_cast<std::uint8_t>(random());
  result[6] = static_cast<std::uint8_t>((result[6] & 0x0F) | 0x40);
  result[8] = static_cast<std::uint8_t>((result[8] & 0x3F) | 0x80);
  return result;
}

std::uint32_t SplitCode(std::string_view split) {
  if (split == "train")
    return 0;
  if (split == "val")
    return 1;
  if (split == "test")
    return 2;
  if (split == "other")
    return 3;
  throw std::runtime_error("split must be train, val, test or other");
}

struct ShapeInfo {
  std::uint32_t hidden{};
  std::uint32_t hc_count{};
  std::uint32_t vocab{};
  std::uint32_t topk{};
  std::uint32_t max_context{};
  bool hidden_f32{false};
  bool mtp_hidden{true};  ///< store the MTP final wide residual
  bool dflash{false};     ///< store the DFlash2 stream-mean taps
  std::uint32_t HcDim() const { return hidden * hc_count; }
  std::uint32_t HiddenElem() const { return hidden_f32 ? 4 : 2; }
  std::uint32_t DflashTaps() const {
    return dflash ? static_cast<std::uint32_t>(kDflashTaps.size()) : 0;
  }
  /// File header feature mask: bit0 DFlash taps present, bit1 MTP hidden
  /// absent. MTP-only shards keep 0, as before the DFlash mode existed.
  std::uint32_t FeatureMask() const {
    return (dflash ? 1u : 0u) | (mtp_hidden ? 0u : 2u);
  }
  const char* FeatureName() const {
    return dflash ? (mtp_hidden ? "both" : "dflash") : "mtp";
  }
  std::string Abi() const {
    return AbiText(hidden, hc_count, vocab, topk, hidden_f32, mtp_hidden,
                   dflash);
  }
};

std::vector<std::uint8_t> MakeFileHeader(
    const ShapeInfo& shape, std::string_view target_identity,
    const std::array<std::uint8_t, 32>& abi,
    const std::array<std::uint8_t, 32>& producer,
    const std::array<std::uint8_t, 20>& producer_git, std::uint32_t split,
    std::uint64_t shard_ordinal, const std::array<std::uint8_t, 16>& uuid) {
  std::vector<std::uint8_t> bytes(kFileHeaderBytes);
  std::ranges::copy(kFileMagic, bytes.begin());
  PutU16(bytes, 8, 1);
  PutU16(bytes, 10, 0);
  PutU32(bytes, 12, kFileHeaderBytes);
  PutU32(bytes, 16, shape.hidden);
  PutU32(bytes, 20, shape.hc_count);
  PutU32(bytes, 24, shape.HcDim());
  PutU32(bytes, 28, shape.vocab);
  PutU32(bytes, 32, shape.topk);
  PutU32(bytes, 36, SupRowBytes(shape.topk));
  PutU32(bytes, 40, shape.hidden_f32 ? 3 : 1);  // 1 BF16 RNE LE, 3 F32 LE
  PutU32(bytes, 44, 1);  // tokens: U32 LE
  PutU32(bytes, 48, 2);  // top-k log-probs: F16 RNE LE
  PutU32(bytes, 52, shape.max_context);
  PutU32(bytes, 56, kAlignment);
  PutU32(bytes, 60, shape.FeatureMask());
  std::memcpy(bytes.data() + 64, kIdentityScheme.data(),
              kIdentityScheme.size());
  std::ranges::copy(DecodeHex<32>(target_identity, "target identity"),
                    bytes.begin() + 80);
  std::ranges::copy(abi, bytes.begin() + 112);
  std::ranges::copy(producer, bytes.begin() + 144);
  std::ranges::copy(producer_git, bytes.begin() + 176);
  PutU32(bytes, 196, split);
  PutU64(bytes, 200, shard_ordinal);
  const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  PutU64(bytes, 208, static_cast<std::uint64_t>(now));
  std::ranges::copy(uuid, bytes.begin() + 216);
  PutU32(bytes, 232, shape.DflashTaps());
  for (std::uint32_t tap = 0; tap < shape.DflashTaps(); ++tap)
    PutU32(bytes, 236 + 4 * std::size_t{tap}, kDflashTaps[tap]);
  const auto digest = HashBytes(
      std::span<const std::uint8_t>(bytes).first(kHeaderDigestOffset));
  std::ranges::copy(digest, bytes.begin() + kHeaderDigestOffset);
  return bytes;
}

// ------------------------------------------------------------- input rows

struct InputRecord {
  std::string id;
  std::string source_id;
  std::vector<std::int32_t> tokens;
  std::uint32_t supervise_from{0};
  std::string metadata;  ///< canonical record metadata JSON object
};

InputRecord ParseRecord(const gufo::json::Value& value,
                        const qfn::Model* model, std::uint32_t vocab,
                        std::uint32_t max_context, std::string_view split,
                        std::string_view identity) {
  if (!value.is_object())
    throw std::runtime_error("each JSONL row must be an object");
  const auto* id = value.find("id");
  const auto* source_id = value.find("source_id");
  const auto* text = value.find("text");
  const auto* token_values = value.find("tokens");
  const auto* record_split = value.find("split");
  const auto* supervise = value.find("supervise_from");
  const auto* target = value.find("target_sha256");
  const auto* metadata = value.find("metadata");
  if (id == nullptr || !id->is_string() || id->str().empty())
    throw std::runtime_error("JSONL row requires non-empty string id");
  if (source_id == nullptr || !source_id->is_string() ||
      source_id->str().empty())
    throw std::runtime_error("JSONL row requires non-empty string source_id");
  if ((text != nullptr) == (token_values != nullptr))
    throw std::runtime_error("JSONL row requires exactly one of text/tokens");
  if (record_split != nullptr &&
      (!record_split->is_string() || record_split->str() != split))
    throw std::runtime_error("record split differs from its job split: " +
                             id->str());
  if (target != nullptr && (!target->is_string() || target->str() != identity))
    throw std::runtime_error(
        "record was generated by a different target model: " + id->str());
  InputRecord result{.id = id->str(), .source_id = source_id->str()};
  if (text != nullptr) {
    if (model == nullptr || !text->is_string() || text->str().empty())
      throw std::runtime_error("text must be a non-empty string");
    result.tokens = model->Tokenize(text->str());
  } else {
    if (!token_values->is_array() || token_values->items().empty())
      throw std::runtime_error("tokens must be a non-empty array");
    result.tokens.reserve(token_values->items().size());
    for (const auto& token : token_values->items()) {
      const double numeric = token.as_double(-1.0);
      if (!token.is_number() || !std::isfinite(numeric) || numeric < 0.0 ||
          std::floor(numeric) != numeric || numeric >= vocab)
        throw std::runtime_error("token is outside the target vocabulary");
      result.tokens.push_back(static_cast<std::int32_t>(numeric));
    }
  }
  if (result.tokens.empty() || result.tokens.size() > max_context)
    throw std::runtime_error("record exceeds the capture context: " +
                             result.id + " (" +
                             std::to_string(result.tokens.size()) + ")");
  for (std::int32_t token : result.tokens)
    if (token < 0 || static_cast<std::uint32_t>(token) >= vocab)
      throw std::runtime_error("token is outside the target vocabulary");
  if (supervise != nullptr) {
    const double numeric = supervise->as_double(-1.0);
    if (!supervise->is_number() || numeric < 0.0 ||
        std::floor(numeric) != numeric ||
        numeric > static_cast<double>(result.tokens.size()))
      throw std::runtime_error("supervise_from must be in [0, tokens]");
    result.supervise_from = static_cast<std::uint32_t>(numeric);
  }
  auto meta = gufo::json::Value::object();
  meta.append_member("id", gufo::json::Value(result.id));
  meta.append_member("source_id", gufo::json::Value(result.source_id));
  meta.append_member("split", gufo::json::Value(std::string(split)));
  meta.append_member(
      "supervise_from",
      gufo::json::Value(static_cast<unsigned long long>(result.supervise_from)));
  if (metadata != nullptr) {
    if (!metadata->is_object())
      throw std::runtime_error("metadata must be an object");
    meta.append_member("metadata", *metadata);
  }
  result.metadata = meta.dump();
  if (result.metadata.size() > kMaxMetadataBytes)
    throw std::runtime_error("record metadata exceeds 64 KiB: " + result.id);
  return result;
}

// ---------------------------------------------------------- record capture

struct CapturedRecord {
  std::vector<std::uint8_t> hidden;  ///< n * HcDim * HiddenElem() bytes
  std::vector<float> row_absmax;     ///< per-row max |residual| (F32)
  std::vector<std::uint8_t> dflash;  ///< n * taps * hidden BF16 (DFlash2)
  std::vector<std::uint8_t> sup;      ///< (n - supervise_from) * row bytes
  bool head_checked{false};
  bool argmax_match{true};
  float lse_diff{0.0F};
  // Head cross-check diagnostics (last row; captured minus production).
  float max_dlogit{0.0F};   ///< max |logit difference| over the vocabulary
  float mean_dlogit{0.0F};  ///< mean signed logit difference
  float dlogp_top1{0.0F};   ///< log-prob difference at the production argmax
  float prod_margin{0.0F};  ///< production top1 - top2 logit gap
  std::vector<float> captured_last;  ///< captured full logits of row n-1
};

struct Timers {
  double capture_s{0};
  double reduce_s{0};
  double convert_s{0};
  double write_s{0};
};

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
      .count();
}

struct RecordIndex {
  std::string id;
  std::string source_id;
  std::uint64_t offset{};
  std::uint64_t bytes{};
  std::uint32_t tokens{};
  std::uint32_t supervise_from{};
  float hidden_absmax{};
};

RecordIndex WriteRecord(TraceSink* sink, const ShapeInfo& shape,
                        const InputRecord& input,
                        const CapturedRecord& captured,
                        std::uint64_t ordinal) {
  const auto n = static_cast<std::uint32_t>(input.tokens.size());
  const std::uint32_t row_bytes = SupRowBytes(shape.topk);
  const std::uint32_t sup_rows = shape.topk == 0 ? 0 : n - input.supervise_from;
  std::vector<std::uint8_t> token_bytes(std::size_t{n} * 4);
  for (std::uint32_t index = 0; index < n; ++index)
    PutU32(token_bytes, std::size_t{index} * 4,
           static_cast<std::uint32_t>(input.tokens[index]));
  const std::size_t hidden_bytes = captured.hidden.size();
  const std::size_t sup_bytes = std::size_t{sup_rows} * row_bytes;
  const std::size_t dflash_bytes = captured.dflash.size();
  if (hidden_bytes != (shape.mtp_hidden ? std::size_t{n} * shape.HcDim() *
                                              shape.HiddenElem()
                                        : 0) ||
      dflash_bytes !=
          std::size_t{n} * shape.DflashTaps() * shape.hidden * 2 ||
      captured.sup.size() != sup_bytes)
    throw std::logic_error("captured record has the wrong shape");
  const std::size_t metadata_offset = kRecordHeaderBytes;
  const std::size_t token_offset =
      Align(metadata_offset + input.metadata.size());
  const std::size_t hidden_offset = Align(token_offset + token_bytes.size());
  // DFlash section (absent in MTP-only shards) sits between hidden and sup.
  const std::size_t dflash_offset = Align(hidden_offset + hidden_bytes);
  const std::size_t sup_offset = Align(dflash_offset + dflash_bytes);
  const std::size_t trailer_offset = Align(sup_offset + sup_bytes);
  const std::size_t total = trailer_offset + kRecordTrailerBytes;
  const std::uint32_t flags = captured.head_checked ? kFlagHeadChecked : 0;

  std::vector<std::uint8_t> header(kRecordHeaderBytes);
  std::ranges::copy(kRecordMagic, header.begin());
  PutU16(header, 8, 1);
  PutU16(header, 10, 0);
  PutU32(header, 12, kRecordHeaderBytes);
  PutU64(header, 16, total);
  PutU64(header, 24, ordinal);
  PutU32(header, 32, n);
  PutU32(header, 36, input.supervise_from);
  PutU32(header, 40, static_cast<std::uint32_t>(input.metadata.size()));
  PutU32(header, 44, flags);
  PutU64(header, 48, metadata_offset);
  PutU64(header, 56, token_offset);
  PutU64(header, 64, hidden_offset);
  PutU64(header, 72, sup_offset);
  PutU64(header, 80, trailer_offset);
  PutU64(header, 88, hidden_bytes);
  std::ranges::copy(HashString(input.id), header.begin() + 96);
  std::ranges::copy(HashString(input.source_id), header.begin() + 128);
  std::ranges::copy(HashBytes(token_bytes), header.begin() + 160);

  RecordIndex index{.id = input.id,
                    .source_id = input.source_id,
                    .offset = sink->bytes(),
                    .bytes = total,
                    .tokens = n,
                    .supervise_from = input.supervise_from,
                    .hidden_absmax = captured.row_absmax.empty()
                                         ? 0.0F
                                         : *std::max_element(
                                               captured.row_absmax.begin(),
                                               captured.row_absmax.end())};
  gufo::crypto::Sha256Hasher record_hash;
  sink->Write(header, &record_hash);
  sink->Write({reinterpret_cast<const std::uint8_t*>(input.metadata.data()),
               input.metadata.size()},
              &record_hash);
  sink->WriteZeros(token_offset - metadata_offset - input.metadata.size(),
                   &record_hash);
  sink->Write(token_bytes, &record_hash);
  sink->WriteZeros(hidden_offset - token_offset - token_bytes.size(),
                   &record_hash);
  sink->Write(captured.hidden, &record_hash);
  sink->WriteZeros(dflash_offset - hidden_offset - hidden_bytes, &record_hash);
  if (dflash_bytes != 0) {
    sink->Write(captured.dflash, &record_hash);
    sink->WriteZeros(sup_offset - dflash_offset - dflash_bytes, &record_hash);
  }
  sink->Write(captured.sup, &record_hash);
  sink->WriteZeros(trailer_offset - sup_offset - sup_bytes, &record_hash);
  std::vector<std::uint8_t> trailer(kRecordTrailerBytes);
  std::ranges::copy(record_hash.Finish(), trailer.begin());
  PutU64(trailer, 32, total);
  PutU64(trailer, 40, ordinal);
  PutU32(trailer, 48, n);
  PutU32(trailer, 52, flags);
  std::ranges::copy(kCommitMagic, trailer.begin() + 56);
  sink->Write(trailer);
  sink->Durable();
  return index;
}

/// Stores `rows` residual rows (BF16 or F32) at `first_row` and records each
/// row's max |value|.
bool ConvertResidual(const float* residual, std::uint32_t first_row,
                     std::uint32_t rows, std::uint32_t hc_dim, bool f32,
                     unsigned threads, std::vector<std::uint8_t>* hidden,
                     std::vector<float>* absmax, std::string* error) {
  const std::size_t elem = f32 ? 4 : 2;
  std::atomic<bool> ok{true};
  ParallelFor(rows, threads, [&](std::uint32_t row) {
    const float* source = residual + std::size_t{row} * hc_dim;
    std::uint8_t* destination =
        hidden->data() + (std::size_t{first_row} + row) * hc_dim * elem;
    bool row_ok = true;
    float peak = 0.0F;
    for (std::uint32_t j = 0; j < hc_dim; ++j) {
      const float value = source[j];
      const float magnitude = std::fabs(value);
      peak = magnitude > peak ? magnitude : peak;
      if (f32) {
        row_ok &= std::isfinite(value);
        StoreF32(destination + 4 * std::size_t{j}, value);
      } else {
        std::uint16_t bits = 0;
        row_ok &= ToBf16(value, &bits);
        StoreU16(destination + 2 * std::size_t{j}, bits);
      }
    }
    (*absmax)[std::size_t{first_row} + row] = peak;
    if (!row_ok)
      ok.store(false);
  });
  if (!ok.load() && error != nullptr)
    *error = f32 ? "non-finite final residual"
                 : "non-finite or BF16-overflowing final residual";
  return ok.load();
}

/// DFlash2 tap: plain mean of the hc_count streams of `rows` raw residual
/// rows, accumulated in llama.cpp's order (hc_stream_mean: s0 + s1, + s2,
/// + s3, then * 1/hc), stored BF16 at [row][tap][hidden].
bool ReduceTapRows(const float* raw, std::uint32_t tap, std::uint32_t first_row,
                   std::uint32_t rows, const ShapeInfo& shape, unsigned threads,
                   std::vector<std::uint8_t>* dflash, std::string* error) {
  const std::uint32_t taps = shape.DflashTaps();
  const std::uint32_t hidden = shape.hidden;
  const float scale = 1.0F / static_cast<float>(shape.hc_count);
  if (tap >= taps ||
      (std::size_t{first_row} + rows) * taps * hidden * 2 > dflash->size()) {
    if (error != nullptr)
      *error = "DFlash tap rows outside the record";
    return false;
  }
  std::atomic<bool> ok{true};
  ParallelFor(rows, threads, [&](std::uint32_t row) {
    const float* source = raw + std::size_t{row} * shape.HcDim();
    std::uint8_t* destination =
        dflash->data() +
        ((std::size_t{first_row} + row) * taps + tap) * hidden * 2;
    bool row_ok = true;
    for (std::uint32_t j = 0; j < hidden; ++j) {
      float sum = source[j];
      for (std::uint32_t stream = 1; stream < shape.hc_count; ++stream)
        sum = sum + source[std::size_t{stream} * hidden + j];
      std::uint16_t bits = 0;
      row_ok &= ToBf16(sum * scale, &bits);
      StoreU16(destination + 2 * std::size_t{j}, bits);
    }
    if (!row_ok)
      ok.store(false);
  });
  if (!ok.load() && error != nullptr)
    *error = "non-finite or BF16-overflowing DFlash tap";
  return ok.load();
}

bool ReduceRows(const float* logits, std::uint32_t first_row,
                std::uint32_t rows, const ShapeInfo& shape,
                const InputRecord& input, unsigned threads,
                std::vector<std::uint8_t>* sup, std::string* error) {
  const std::uint32_t row_bytes = SupRowBytes(shape.topk);
  const auto n = static_cast<std::uint32_t>(input.tokens.size());
  if (first_row < input.supervise_from || first_row + rows > n) {
    if (error != nullptr)
      *error = "head rows outside the supervised range";
    return false;
  }
  std::atomic<bool> ok{true};
  ParallelFor(rows, threads, [&](std::uint32_t local) {
    const std::uint32_t row = first_row + local;
    const std::int32_t next = row + 1 < n ? input.tokens[row + 1] : -1;
    std::uint8_t* out =
        sup->data() + std::size_t{row - input.supervise_from} * row_bytes;
    if (!ReduceRow(logits + std::size_t{local} * shape.vocab, shape.vocab,
                   next, shape.topk, out, nullptr))
      ok.store(false);
  });
  if (!ok.load() && error != nullptr)
    *error = "non-finite target logit";
  return ok.load();
}

std::string SafeName(std::string_view id) {
  std::string out(id.substr(0, 80));
  for (char& ch : out)
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' &&
        ch != '_' && ch != '.')
      ch = '_';
  return out;
}

void WriteFloats(const std::filesystem::path& path, std::span<const float> v) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(v.data()),
               static_cast<std::streamsize>(v.size_bytes()));
  if (!stream)
    throw std::runtime_error("cannot write " + path.string());
}

CapturedRecord CaptureRecord(qfn::Model& model, const ShapeInfo& shape,
                             const InputRecord& input,
                             std::uint32_t head_rows_per_call,
                             unsigned threads, Timers* timers,
                             const std::filesystem::path& dump_dir = {}) {
  const auto n = static_cast<std::uint32_t>(input.tokens.size());
  CapturedRecord result;
  if (shape.mtp_hidden) {
    result.hidden.resize(std::size_t{n} * shape.HcDim() * shape.HiddenElem());
    result.row_absmax.resize(n);
  }
  result.dflash.resize(std::size_t{n} * shape.DflashTaps() * shape.hidden * 2);
  if (shape.topk != 0)
    result.sup.resize(std::size_t{n - input.supervise_from} *
                      SupRowBytes(shape.topk));
  std::vector<float> final_logits;
  double reduce = 0;
  double convert = 0;
  qfn::Model::MtpCaptureRequest request;
  request.head_first_row = input.supervise_from;
  request.head_rows_per_call = shape.topk == 0 ? 0 : head_rows_per_call;
  if (shape.dflash) {
    request.tap_layers.assign(kDflashTaps.begin(), kDflashTaps.end());
    request.tap_sink = [&](std::uint32_t tap, std::uint32_t first_row,
                           std::uint32_t rows, const float* raw,
                           std::string* error) {
      const auto start = std::chrono::steady_clock::now();
      bool ok = false;
      try {
        ok = ReduceTapRows(raw, tap, first_row, rows, shape, threads,
                           &result.dflash, error);
      } catch (const std::exception& exception) {
        if (error != nullptr)
          *error = exception.what();
      }
      convert += Seconds(start);
      return ok;
    };
  }
  if (shape.mtp_hidden)
  request.residual_sink = [&](std::uint32_t first_row, std::uint32_t rows,
                              const float* residual, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    bool ok = false;
    try {
      ok = ConvertResidual(residual, first_row, rows, shape.HcDim(),
                           shape.hidden_f32, threads, &result.hidden,
                           &result.row_absmax, error);
    } catch (const std::exception& exception) {
      if (error != nullptr)
        *error = exception.what();
    }
    convert += Seconds(start);
    return ok;
  };
  if (shape.topk != 0) {
    request.head_sink = [&](std::uint32_t first_row, std::uint32_t rows,
                            const float* logits, std::string* error) {
      const auto start = std::chrono::steady_clock::now();
      bool ok = false;
      try {
        ok = ReduceRows(logits, first_row, rows, shape, input, threads,
                        &result.sup, error);
        // Keep the captured last row for the production cross-check.
        if (ok && first_row + rows == n)
          result.captured_last.assign(
              logits + std::size_t{rows - 1} * shape.vocab,
              logits + std::size_t{rows} * shape.vocab);
      } catch (const std::exception& exception) {
        if (error != nullptr)
          *error = exception.what();
      }
      reduce += Seconds(start);
      return ok;
    };
    request.final_logits = &final_logits;
  }
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  if (!model.CaptureMtpTrace(input.tokens, request, &error))
    throw std::runtime_error("capture failed for " + input.id + ": " + error);
  timers->capture_s += Seconds(start);
  timers->reduce_s += reduce;
  timers->convert_s += convert;
  // Cross-check the wide-GEMM head against the production single-row head.
  if (shape.topk != 0 && final_logits.size() == shape.vocab &&
      n > input.supervise_from) {
    std::vector<std::uint8_t> check(SupRowBytes(shape.topk));
    RowSummary production;
    if (!ReduceRow(final_logits.data(), shape.vocab, -1, shape.topk,
                   check.data(), &production))
      throw std::runtime_error("non-finite production head row");
    const std::uint8_t* captured =
        result.sup.data() +
        std::size_t{n - 1 - input.supervise_from} * SupRowBytes(shape.topk);
    std::uint32_t captured_argmax = 0;
    float captured_lse = 0;
    std::memcpy(&captured_argmax, captured, 4);
    std::memcpy(&captured_lse, captured + 6 * shape.topk, 4);
    result.head_checked = true;
    result.argmax_match = captured_argmax == production.argmax;
    result.lse_diff = std::fabs(captured_lse - production.lse);
    std::uint32_t second = 0;
    std::memcpy(&second, check.data() + 4, 4);
    result.prod_margin =
        second < shape.vocab
            ? final_logits[production.argmax] - final_logits[second]
            : 0.0F;
    if (result.captured_last.size() == shape.vocab) {
      double sum = 0.0;
      float peak = 0.0F;
      for (std::uint32_t v = 0; v < shape.vocab; ++v) {
        const float d = result.captured_last[v] - final_logits[v];
        sum += d;
        peak = std::max(peak, std::fabs(d));
      }
      result.max_dlogit = peak;
      result.mean_dlogit = static_cast<float>(sum / shape.vocab);
      result.dlogp_top1 =
          (result.captured_last[production.argmax] - captured_lse) -
          (final_logits[production.argmax] - production.lse);
    }
    if (!dump_dir.empty()) {
      const std::string stem = SafeName(input.id);
      WriteFloats(dump_dir / (stem + ".production.f32"), final_logits);
      if (!result.captured_last.empty())
        WriteFloats(dump_dir / (stem + ".captured.f32"), result.captured_last);
      std::vector<float> tokens(input.tokens.begin(), input.tokens.end());
      WriteFloats(dump_dir / (stem + ".tokens.f32"), tokens);
    }
  }
  result.captured_last.clear();
  result.captured_last.shrink_to_fit();
  return result;
}

// ------------------------------------------------------------------ jobs

struct Job {
  std::filesystem::path input;
  std::string input_sha256;
  std::uint64_t shard_ordinal{};
  std::string split;
};

std::string RequireString(const gufo::json::Value& value, const char* key) {
  const auto* field = value.find(key);
  if (field == nullptr || !field->is_string() || field->str().empty())
    throw std::runtime_error(std::string("jobs file requires string ") + key);
  return field->str();
}

std::uint64_t RequireU64(const gufo::json::Value& value, const char* key) {
  const auto* field = value.find(key);
  const double number = field == nullptr ? -1.0 : field->as_double(-1.0);
  if (field == nullptr || !field->is_number() || number < 0.0 ||
      std::floor(number) != number || number > 9007199254740991.0)
    throw std::runtime_error(std::string("jobs file requires integer ") + key);
  return static_cast<std::uint64_t>(number);
}

std::string ReadFile(const std::filesystem::path& path,
                     std::size_t max_bytes) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0 || size > max_bytes)
    throw std::runtime_error("file size is unsafe: " + path.string());
  std::ifstream stream(path, std::ios::binary);
  std::string bytes(static_cast<std::size_t>(size), '\0');
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (stream.gcount() != static_cast<std::streamsize>(bytes.size()))
    throw std::runtime_error("cannot read " + path.string());
  return bytes;
}

std::vector<Job> ParseJobs(const std::filesystem::path& path) {
  const auto root = gufo::json::parse(ReadFile(path, 64 << 20));
  if (RequireString(root, "format") != "gufo.mtp-capture.jobs.v1")
    throw std::runtime_error("jobs file format differs");
  const auto* jobs = root.find("jobs");
  if (jobs == nullptr || !jobs->is_array() || jobs->items().empty())
    throw std::runtime_error("jobs file requires non-empty jobs");
  std::vector<Job> result;
  std::set<std::uint64_t> ordinals;
  for (const auto& value : jobs->items()) {
    Job job;
    job.input = RequireString(value, "input");
    if (job.input.is_relative())
      job.input = path.parent_path() / job.input;
    job.input_sha256 = RequireString(value, "input_sha256");
    (void)DecodeHex<32>(job.input_sha256, "input_sha256");
    job.shard_ordinal = RequireU64(value, "shard_ordinal");
    job.split = RequireString(value, "split");
    (void)SplitCode(job.split);
    if (!ordinals.insert(job.shard_ordinal).second)
      throw std::runtime_error("duplicate shard_ordinal in jobs file");
    result.push_back(std::move(job));
  }
  return result;
}

std::string OrdinalTag(std::uint64_t ordinal) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "shard-%06llu",
                static_cast<unsigned long long>(ordinal));
  return buffer;
}

/// Shards already published locally (sidecar present) or shipped (done/).
bool ShardExists(const std::filesystem::path& out, std::uint64_t ordinal) {
  const std::string tag = OrdinalTag(ordinal);
  if (std::filesystem::exists(out / "done" / (tag + ".json")))
    return true;
  for (const auto& entry : std::filesystem::directory_iterator(out)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind(tag + "-", 0) == 0 && name.size() > 10 &&
        name.ends_with(".mtps.json"))
      return true;
  }
  return false;
}

std::size_t PendingShards(const std::filesystem::path& out) {
  std::size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(out))
    if (entry.path().extension() == ".mtps")
      ++count;
  return count;
}

struct Options {
  std::filesystem::path model;
  std::filesystem::path jobs;
  std::filesystem::path out;
  std::string producer_git;
  std::uint32_t max_context{4096};
  std::uint32_t topk{32};
  std::uint32_t head_rows_per_call{126};
  std::uint32_t max_pending{3};
  double min_free_gb{40.0};
  unsigned threads{0};
  std::uint64_t max_jobs{std::numeric_limits<std::uint64_t>::max()};
  double max_hours{0};
  std::filesystem::path stop_file;
  double max_lse_diff{2.0};
  double max_margin_mismatch{0.5};
  std::filesystem::path dump_head_check;
  bool hidden_f32{false};
  std::string features{"mtp"};  ///< mtp | dflash | both
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc)
      throw std::runtime_error("missing value for " + std::string(option));
    const std::string value = argv[++index];
    if (option == "--model")
      options.model = value;
    else if (option == "--jobs")
      options.jobs = value;
    else if (option == "--out")
      options.out = value;
    else if (option == "--producer-git")
      options.producer_git = value;
    else if (option == "--max-context")
      options.max_context = static_cast<std::uint32_t>(std::stoul(value));
    else if (option == "--topk")
      options.topk = static_cast<std::uint32_t>(std::stoul(value));
    else if (option == "--head-rows-per-call")
      options.head_rows_per_call =
          static_cast<std::uint32_t>(std::stoul(value));
    else if (option == "--max-pending")
      options.max_pending = static_cast<std::uint32_t>(std::stoul(value));
    else if (option == "--min-free-gb")
      options.min_free_gb = std::stod(value);
    else if (option == "--threads")
      options.threads = static_cast<unsigned>(std::stoul(value));
    else if (option == "--max-jobs")
      options.max_jobs = std::stoull(value);
    else if (option == "--max-hours")
      options.max_hours = std::stod(value);
    else if (option == "--stop-file")
      options.stop_file = value;
    else if (option == "--max-lse-diff")
      options.max_lse_diff = std::stod(value);
    else if (option == "--max-margin-mismatch")
      options.max_margin_mismatch = std::stod(value);
    else if (option == "--dump-head-check")
      options.dump_head_check = value;
    else if (option == "--hidden-dtype" && (value == "bf16" || value == "f32"))
      options.hidden_f32 = value == "f32";
    else if (option == "--features" &&
             (value == "mtp" || value == "dflash" || value == "both"))
      options.features = value;
    else
      throw std::runtime_error("unknown option: " + std::string(option));
  }
  if (options.model.empty() || options.jobs.empty() || options.out.empty() ||
      options.producer_git.empty() || options.max_context == 0 ||
      options.topk > kMaxTopK || options.max_pending == 0 ||
      (options.topk != 0 && options.head_rows_per_call == 0))
    throw std::runtime_error(
        "usage: w3mtp_capture --model GGUF --jobs JOBS.json --out STAGING_DIR "
        "--producer-git COMMIT [--max-context 4096] [--topk 32 (0..64)] "
        "[--head-rows-per-call 126] [--max-pending 3] [--min-free-gb 40] "
        "[--threads N] [--max-jobs N] [--max-hours H] [--stop-file PATH] "
        "[--max-lse-diff 2.0] [--max-margin-mismatch 0.5] "
        "[--dump-head-check DIR] [--hidden-dtype bf16|f32] "
        "[--features mtp|dflash|both]\n"
        "       w3mtp_capture --self-test-output EMPTY_DIR");
  (void)DecodeHex<20>(options.producer_git, "producer git commit");
  if (options.threads == 0)
    options.threads =
        std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
  return options;
}

struct ShardStats {
  std::uint64_t records{};
  std::uint64_t tokens{};
  std::uint64_t supervised_rows{};
  std::uint64_t head_checked{};
  std::uint64_t argmax_mismatch{};
  double max_lse_diff{};
  double hidden_absmax{};
  std::uint64_t rows_over_f16{};
  Timers timers;
};

std::string SidecarJson(const std::string& file_name, const std::string& sha,
                        std::uint64_t bytes, const Job& job,
                        const ShapeInfo& shape, const std::string& identity,
                        const std::string& abi, const std::string& producer,
                        const std::string& producer_git,
                        const std::vector<RecordIndex>& index,
                        const ShardStats& stats, double seconds) {
  using gufo::json::Value;
  const auto u64 = [](std::uint64_t value) {
    return Value(static_cast<unsigned long long>(value));
  };
  auto root = Value::object();
  root.append_member("format", Value("gufo.mtp-capture.shard.v1"));
  root.append_member("file", Value(file_name));
  root.append_member("sha256", Value(sha));
  root.append_member("bytes", u64(bytes));
  root.append_member("shard_ordinal", u64(job.shard_ordinal));
  root.append_member("split", Value(job.split));
  root.append_member("input_sha256", Value(job.input_sha256));
  root.append_member("target_sha256", Value(identity));
  root.append_member("abi_sha256", Value(abi));
  root.append_member("producer_sha256", Value(producer));
  root.append_member("producer_git_commit", Value(producer_git));
  root.append_member("hidden", u64(shape.hidden));
  root.append_member("hc_count", u64(shape.hc_count));
  root.append_member("vocab", u64(shape.vocab));
  root.append_member("topk", u64(shape.topk));
  root.append_member("sup_row_bytes", u64(SupRowBytes(shape.topk)));
  root.append_member("max_context", u64(shape.max_context));
  root.append_member("records_count", u64(stats.records));
  root.append_member("tokens", u64(stats.tokens));
  root.append_member("supervised_rows", u64(stats.supervised_rows));
  auto check = Value::object();
  check.append_member("rows", u64(stats.head_checked));
  check.append_member("argmax_mismatch", u64(stats.argmax_mismatch));
  check.append_member("max_lse_abs_diff", Value(stats.max_lse_diff));
  root.append_member("head_check", std::move(check));
  root.append_member("features", Value(shape.FeatureName()));
  root.append_member("hidden_dtype",
                     Value(!shape.mtp_hidden  ? "none"
                           : shape.hidden_f32 ? "f32"
                                              : "bf16"));
  auto dflash_taps = Value::array();
  for (std::uint32_t tap = 0; tap < shape.DflashTaps(); ++tap)
    dflash_taps.push_back(u64(kDflashTaps[tap]));
  root.append_member("dflash_taps", std::move(dflash_taps));
  root.append_member("hidden_absmax", Value(stats.hidden_absmax));
  root.append_member("hidden_rows_absmax_over_f16", u64(stats.rows_over_f16));
  auto timing = Value::object();
  timing.append_member("seconds", Value(seconds));
  timing.append_member("capture_s", Value(stats.timers.capture_s));
  timing.append_member("head_reduce_s", Value(stats.timers.reduce_s));
  timing.append_member("residual_convert_s", Value(stats.timers.convert_s));
  timing.append_member("write_s", Value(stats.timers.write_s));
  timing.append_member(
      "tokens_per_s",
      Value(seconds > 0 ? static_cast<double>(stats.tokens) / seconds : 0.0));
  root.append_member("timing", std::move(timing));
  auto records = Value::array();
  for (const auto& entry : index) {
    auto record = Value::object();
    record.append_member("id", Value(entry.id));
    record.append_member("source_id", Value(entry.source_id));
    record.append_member("offset", u64(entry.offset));
    record.append_member("bytes", u64(entry.bytes));
    record.append_member("tokens", u64(entry.tokens));
    record.append_member("supervise_from", u64(entry.supervise_from));
    record.append_member("hidden_absmax", Value(double{entry.hidden_absmax}));
    records.push_back(std::move(record));
  }
  root.append_member("records", std::move(records));
  return root.dump() + "\n";
}

/// Writes one shard from `job`. `capture` produces each record's data.
template <class CaptureFn, class ParseFn>
void RunShard(const Job& job, const std::filesystem::path& out,
              const ShapeInfo& shape, const std::string& identity,
              const std::array<std::uint8_t, 32>& producer,
              const std::array<std::uint8_t, 20>& producer_git,
              const ParseFn& parse, const CaptureFn& capture,
              double max_lse_diff, double max_margin_mismatch) {
  const auto started = std::chrono::steady_clock::now();
  const std::string abi_text = shape.Abi();
  const auto abi = HashString(abi_text);
  const auto uuid = MakeUuid();
  const std::string tag = OrdinalTag(job.shard_ordinal);
  const auto partial = out / (tag + "-" + Hex(uuid) + ".mtps.partial");
  ShardStats stats;
  std::vector<RecordIndex> index;
  std::array<std::uint8_t, 32> digest{};
  std::uint64_t bytes = 0;
  {
    TraceSink sink(partial);
    sink.Write(MakeFileHeader(shape, identity, abi, producer, producer_git,
                              SplitCode(job.split), job.shard_ordinal, uuid));
    std::ifstream input(job.input, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open " + job.input.string());
    std::unordered_set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (line.empty())
        continue;
      if (line.size() > kMaxJsonLineBytes)
        throw std::runtime_error("JSONL row exceeds 64 MiB");
      const InputRecord record = parse(gufo::json::parse(line));
      if (!ids.insert(record.id).second)
        throw std::runtime_error("duplicate record id in job: " + record.id);
      const CapturedRecord captured = capture(record, &stats.timers);
      if (captured.head_checked) {
        ++stats.head_checked;
        stats.argmax_mismatch += captured.argmax_match ? 0 : 1;
        stats.max_lse_diff =
            std::max(stats.max_lse_diff, double{captured.lse_diff});
        // Gross-bug gate. Wide head calls use the F16-activation GEMM while
        // the production row uses Q8_1 activations (GEMV), so logits differ
        // by activation-quantization noise; a wrong row or input differs by
        // whole distributions (large dLSE and confident argmax flips).
        if (captured.lse_diff > max_lse_diff ||
            (!captured.argmax_match &&
             captured.prod_margin > max_margin_mismatch))
          throw std::runtime_error(
              "captured head disagrees with the production head (|dLSE|=" +
              std::to_string(captured.lse_diff) + ", argmax_match=" +
              std::to_string(captured.argmax_match ? 1 : 0) +
              ", production margin=" + std::to_string(captured.prod_margin) +
              ") on " + record.id +
              "; see run_head_repro.ps1 (--head-rows-per-call 1 / 8)");
      }
      for (const float peak : captured.row_absmax) {
        stats.hidden_absmax = std::max(stats.hidden_absmax, double{peak});
        stats.rows_over_f16 += peak > 65504.0F ? 1 : 0;
      }
      const auto write_start = std::chrono::steady_clock::now();
      index.push_back(
          WriteRecord(&sink, shape, record, captured, stats.records));
      stats.timers.write_s += Seconds(write_start);
      ++stats.records;
      stats.tokens += record.tokens.size();
      stats.supervised_rows +=
          shape.topk == 0 ? 0 : record.tokens.size() - record.supervise_from;
      std::fprintf(stderr,
                   "MTP_RECORD shard=%llu ordinal=%llu id=%s tokens=%zu "
                   "supervised=%zu head_dlse=%.5f head_max_dlogit=%.5f "
                   "head_mean_dlogit=%.5f head_dlogp_top1=%.5f "
                   "prod_margin=%.4f argmax_match=%d elapsed_s=%.1f\n",
                   static_cast<unsigned long long>(job.shard_ordinal),
                   static_cast<unsigned long long>(stats.records - 1),
                   record.id.c_str(), record.tokens.size(),
                   shape.topk == 0
                       ? std::size_t{0}
                       : record.tokens.size() - record.supervise_from,
                   captured.lse_diff, captured.max_dlogit,
                   captured.mean_dlogit, captured.dlogp_top1,
                   captured.prod_margin, captured.argmax_match ? 1 : 0,
                   Seconds(started));
    }
    if (!input.eof())
      throw std::runtime_error("job JSONL read failed");
    if (stats.records == 0)
      throw std::runtime_error("job JSONL contained no records");
    if (gufo::crypto::Sha256FileHex(job.input) != job.input_sha256)
      throw std::runtime_error("job input changed during capture");
    bytes = sink.bytes();
    digest = sink.FinishHash();
    sink.Close();
  }
  const std::string sha = Hex(digest);
  const std::string name =
      tag + "-" + job.split + "-sha256-" + sha + ".mtps";
  PublishNoReplace(partial, out / name);
  const double seconds = Seconds(started);
  WriteTextNoReplace(
      out / (name + ".json"),
      SidecarJson(name, sha, bytes, job, shape, identity, Hex(abi),
                  Hex(producer), Hex(producer_git), index, stats, seconds));
  std::printf(
      "MTP_SHARD_OK shard=%llu split=%s features=%s records=%llu tokens=%llu "
      "supervised=%llu bytes=%llu seconds=%.1f tok_per_s=%.1f "
      "capture_s=%.1f head_reduce_s=%.1f convert_s=%.1f write_s=%.1f "
      "head_checked=%llu argmax_mismatch=%llu max_lse_diff=%.5f sha256=%s "
      "file=%s\n",
      static_cast<unsigned long long>(job.shard_ordinal), job.split.c_str(),
      shape.FeatureName(), static_cast<unsigned long long>(stats.records),
      static_cast<unsigned long long>(stats.tokens),
      static_cast<unsigned long long>(stats.supervised_rows),
      static_cast<unsigned long long>(bytes), seconds,
      seconds > 0 ? static_cast<double>(stats.tokens) / seconds : 0.0,
      stats.timers.capture_s, stats.timers.reduce_s, stats.timers.convert_s,
      stats.timers.write_s, static_cast<unsigned long long>(stats.head_checked),
      static_cast<unsigned long long>(stats.argmax_mismatch),
      stats.max_lse_diff, sha.c_str(), name.c_str());
  std::fflush(stdout);
}

// --------------------------------------------------------------- self-test

/// Synthetic shard through the real writer/reducer, without HIP. The data
/// formulas are mirrored by read_gmtp.py selftest-check.
int SelfTest(const std::filesystem::path& directory) {
  if (!std::filesystem::is_directory(directory) ||
      std::filesystem::directory_iterator(directory) !=
          std::filesystem::directory_iterator())
    throw std::runtime_error("self-test output must be an empty directory");
  const ShapeInfo shape_mtp{.hidden = 16,
                            .hc_count = 4,
                            .vocab = 1000,
                            .topk = 32,
                            .max_context = 64};
  ShapeInfo shape_both = shape_mtp;
  shape_both.dflash = true;
  const std::string identity(64, '1');
  const auto producer = HashString("w3mtp-selftest-producer");
  std::array<std::uint8_t, 20> producer_git{};
  producer_git.fill(0x22);
  const unsigned threads = 4;
  const auto parse = [&](const gufo::json::Value& value) {
    return ParseRecord(value, nullptr, shape_mtp.vocab, shape_mtp.max_context,
                       "val", identity);
  };
  const auto make_capture = [&](const ShapeInfo& shape) {
    return [&shape, threads](const InputRecord& input, Timers*) {
      const auto n = static_cast<std::uint32_t>(input.tokens.size());
      CapturedRecord result;
      result.hidden.resize(std::size_t{n} * shape.HcDim() *
                           shape.HiddenElem());
      result.row_absmax.resize(n);
      result.dflash.resize(std::size_t{n} * shape.DflashTaps() *
                           shape.hidden * 2);
      result.sup.resize(std::size_t{n - input.supervise_from} *
                        SupRowBytes(shape.topk));
      std::vector<float> residual(std::size_t{n} * shape.HcDim());
      for (std::uint32_t t = 0; t < n; ++t)
        for (std::uint32_t j = 0; j < shape.HcDim(); ++j)
          residual[std::size_t{t} * shape.HcDim() + j] =
              static_cast<float>(static_cast<int>((t * 31 + j) % 61) - 30) /
              8.0F;
      std::string error;
      if (!ConvertResidual(residual.data(), 0, n, shape.HcDim(), false,
                           threads, &result.hidden, &result.row_absmax,
                           &error))
        throw std::runtime_error(error);
      // Synthetic raw taps: tap k, token t, stream s, column j.
      for (std::uint32_t k = 0; k < shape.DflashTaps(); ++k) {
        std::vector<float> raw(std::size_t{n} * shape.HcDim());
        for (std::uint32_t t = 0; t < n; ++t)
          for (std::uint32_t st = 0; st < shape.hc_count; ++st)
            for (std::uint32_t j = 0; j < shape.hidden; ++j)
              raw[std::size_t{t} * shape.HcDim() + st * shape.hidden + j] =
                  static_cast<float>(
                      static_cast<int>((t * 7 + st * 13 + j * 3 + k * 11) %
                                       29) -
                      14) /
                  4.0F;
        if (!ReduceTapRows(raw.data(), k, 0, n, shape, threads,
                           &result.dflash, &error))
          throw std::runtime_error(error);
      }
      const std::uint32_t rows = n - input.supervise_from;
      std::vector<float> logits(std::size_t{rows} * shape.vocab);
      for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint64_t row = input.supervise_from + r;
        for (std::uint32_t v = 0; v < shape.vocab; ++v)
          logits[std::size_t{r} * shape.vocab + v] =
              static_cast<float>(((std::uint64_t{v} * 2654435761ull +
                                   row * 40503ull) %
                                  1000003ull) %
                                 2048ull) /
                  64.0F -
              16.0F;
      }
      if (rows != 0 &&
          !ReduceRows(logits.data(), input.supervise_from, rows, shape, input,
                      threads, &result.sup, &error))
        throw std::runtime_error(error);
      return result;
    };
  };
  const auto jobs_dir = directory / "input";
  std::filesystem::create_directory(jobs_dir);
  const auto input_path = jobs_dir / "selftest.jsonl";
  {
    std::ofstream stream(input_path, std::ios::binary);
    stream << "{\"id\":\"synthetic-0\",\"source_id\":\"synthetic/a.cpp\","
              "\"tokens\":[1,2,3,4,5,6,7],\"metadata\":{\"kind\":\"selftest\"}}"
              "\n";
    stream << "{\"id\":\"synthetic-1\",\"source_id\":\"synthetic/b.py\","
              "\"split\":\"val\",\"supervise_from\":3,"
              "\"tokens\":[9,8,7,6,5,4,3,2,1,0,999]}\n";
  }
  const auto input_sha = gufo::crypto::Sha256FileHex(input_path);
  const auto out = directory / "staging";
  std::filesystem::create_directory(out);
  // Shard 7: MTP-only (the original self-test); shard 8: features=both.
  const Job job_mtp{.input = input_path,
                    .input_sha256 = input_sha,
                    .shard_ordinal = 7,
                    .split = "val"};
  RunShard(job_mtp, out, shape_mtp, identity, producer, producer_git, parse,
           make_capture(shape_mtp), 2.0, 0.5);
  const Job job_both{.input = input_path,
                     .input_sha256 = input_sha,
                     .shard_ordinal = 8,
                     .split = "val"};
  RunShard(job_both, out, shape_both, identity, producer, producer_git, parse,
           make_capture(shape_both), 2.0, 0.5);
  std::printf("MTP_SELFTEST_WRITTEN dir=%s\n", out.string().c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<qfn::Model> model;
  bool hip_initialized = false;
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--self-test-output")
      return SelfTest(argv[2]);
    const Options options = ParseOptions(argc, argv);
    if (!std::filesystem::is_directory(options.out) ||
        std::filesystem::is_symlink(options.out))
      throw std::runtime_error("--out must be an existing directory");
    std::filesystem::create_directories(options.out / "done");
    if (!options.dump_head_check.empty())
      std::filesystem::create_directories(options.dump_head_check);
    // Stale partials are this tool's own scratch from an interrupted run.
    for (const auto& entry : std::filesystem::directory_iterator(options.out))
      if (entry.path().extension() == ".partial") {
        std::fprintf(stderr, "MTP_REMOVE_STALE_PARTIAL %s\n",
                     entry.path().string().c_str());
        std::filesystem::remove(entry.path());
      }
    const std::vector<Job> jobs = ParseJobs(options.jobs);
    const std::string producer_sha256 =
        gufo::crypto::Sha256FileHex(ExecutablePath());
    const auto producer = DecodeHex<32>(producer_sha256, "producer");
    const auto producer_git = DecodeHex<20>(options.producer_git, "git");

    if (const hipError_t status = hipInit(0); status != hipSuccess)
      throw std::runtime_error(std::string("HIP initialization failed: ") +
                               hipGetErrorString(status));
    hip_initialized = true;
    std::string error;
    qfn::ModelOptions model_options;
    model_options.max_context = options.max_context;
    model = qfn::Model::Load(options.model.string(), model_options, &error);
    if (!model)
      throw std::runtime_error("model load failed: " + error);
    const std::string identity = model->ArtifactIdentityHex();
    const ShapeInfo shape{.hidden = model->config().hidden_size,
                          .hc_count = model->config().hc_count,
                          .vocab = model->VocabSize(),
                          .topk = options.topk,
                          .max_context = options.max_context,
                          .hidden_f32 = options.hidden_f32,
                          .mtp_hidden = options.features != "dflash",
                          .dflash = options.features != "mtp"};
    std::printf(
        "MTP_CAPTURE_START target=%s producer=%s hidden=%u hc_count=%u "
        "vocab=%u topk=%u features=%s threads=%u jobs=%zu abi=%s\n",
        identity.c_str(), producer_sha256.c_str(), shape.hidden,
        shape.hc_count, shape.vocab, shape.topk, shape.FeatureName(),
        options.threads, jobs.size(),
        Hex(HashString(shape.Abi())).c_str());
    std::fflush(stdout);
    const auto run_started = std::chrono::steady_clock::now();
    std::uint64_t completed = 0;
    std::uint64_t skipped = 0;
    for (const Job& job : jobs) {
      if (completed >= options.max_jobs)
        break;
      if (!options.stop_file.empty() &&
          std::filesystem::exists(options.stop_file)) {
        std::printf("MTP_CAPTURE_STOP_FILE %s\n",
                    options.stop_file.string().c_str());
        break;
      }
      if (options.max_hours > 0 &&
          Seconds(run_started) > options.max_hours * 3600.0) {
        std::printf("MTP_CAPTURE_TIME_BUDGET hours=%.2f\n", options.max_hours);
        break;
      }
      if (ShardExists(options.out, job.shard_ordinal)) {
        ++skipped;
        continue;
      }
      // Backpressure: bounded unshipped shards and free space.
      bool announced = false;
      for (;;) {
        const auto space = std::filesystem::space(options.out);
        const double free_gb = static_cast<double>(space.available) / 1e9;
        const std::size_t pending = PendingShards(options.out);
        if (pending < options.max_pending && free_gb >= options.min_free_gb)
          break;
        if (!announced) {
          std::printf("MTP_CAPTURE_WAIT pending=%zu free_gb=%.1f\n", pending,
                      free_gb);
          std::fflush(stdout);
          announced = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
      if (gufo::crypto::Sha256FileHex(job.input) != job.input_sha256)
        throw std::runtime_error("job input SHA-256 differs: " +
                                 job.input.string());
      const auto job_parse = [&](const gufo::json::Value& value) {
        return ParseRecord(value, model.get(), shape.vocab, shape.max_context,
                           job.split, identity);
      };
      const auto capture = [&](const InputRecord& record, Timers* timers) {
        return CaptureRecord(*model, shape, record, options.head_rows_per_call,
                             options.threads, timers, options.dump_head_check);
      };
      RunShard(job, options.out, shape, identity, producer, producer_git,
               job_parse, capture, options.max_lse_diff,
               options.max_margin_mismatch);
      ++completed;
    }
    model.reset();
    if (const std::string shutdown = FinalizeHipRuntime(&hip_initialized);
        !shutdown.empty())
      throw std::runtime_error(shutdown);
    std::printf("MTP_CAPTURE_DONE completed=%llu skipped=%llu hours=%.3f\n",
                static_cast<unsigned long long>(completed),
                static_cast<unsigned long long>(skipped),
                Seconds(run_started) / 3600.0);
    return 0;
  } catch (const std::exception& exception) {
    model.reset();
    if (const std::string shutdown = FinalizeHipRuntime(&hip_initialized);
        !shutdown.empty())
      std::fprintf(stderr, "MTP_HIP_FINALIZE_FAILED: %s\n", shutdown.c_str());
    std::fprintf(stderr, "MTP_CAPTURE_FAILED: %s\n", exception.what());
    return 1;
  }
}
