# Building and running gufo on Strix Halo (Windows 11)



This page covers what we did on AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GiB unified memory, Windows 11 Pro. Our build is upstream v0.34.0 plus the patches in
[patches/](../patches/), built with upstream's own Windows recipe. Our changes will also go upstream as separate pull
requests.

## 1. Build stock gufo

Follow upstream's [BUILD-WINDOWS.md](https://github.com/gufo-org/gufo/blob/main/BUILD-WINDOWS.md). In short:

1. Install Visual Studio 2022 (or newer) with the C++ workload, Git, CMake >= 3.29, Ninja, Python 3, Rust >= 1.88
   (`x86_64-pc-windows-msvc`), and a full vcpkg clone with `VCPKG_ROOT` set.
2. Install TheRock 10.0.0 for gfx1151 and set `HIP_PATH` to it. Check the tarball's sha256 against the value in
   upstream's guide.
3. Patch the SDK's HIP math headers once: `python tools\windows\patch_sdk_math.py %HIP_PATH%`.
4. `tools\windows\vs-env.cmd cmake --preset windows-release`, then
   `tools\windows\vs-env.cmd cmake --build --preset windows-release`.
5. Check it with `build\windows-release\gufo.exe diagnose`. It should report the gfx1151 GPU.

The stock arm in our benchmark was built exactly this way, at v0.34.0 (`2845939b`).

**Our build.** Check out upstream v0.34.0 (`2845939b`), apply the patches with
`git apply --ignore-whitespace patches\*.patch` (in the order of `patches/README.md`), then run steps 3-5 above.
Our benchmarked build uses exactly this toolchain (TheRock 10.0.0), the same as stock. Set the switches listed in the [README](../README.md#switches-used-in-the-benchmark) as
environment variables before `gufo.exe serve`; with none set the build behaves as upstream.

## 2. Get the model and the public draft head

- Weights: [Unsloth Qwen3.8-Flash-Next GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), `UD-Q4_K_XL`
  (we used revision `38bb39ee`; upstream now recommends a newer one).
- Draft head: `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same repository and revision.

Check every file's sha256 against Hugging Face's LFS ids.

## 3. Run the server the way we benchmarked it

Agent / chat profile (one session; use `--sessions 8` for eight):

```cmd
gufo.exe serve --host 127.0.0.1 --port 8080 --sessions 1 llm ^
  --model <path>\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --served-model-name flash-next ^
  --context 32768 --think on --reasoning-effort xhigh --temperature 0 ^
  --speculative mtp --mtp-model <path>\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf ^
  --cache-ram-bytes 9000000000
```

Canonical prose rows use `--think off --max-tokens 128` and upstream's model-bench recipe (pp2048 / tg128).

Using your own draft head: pass it with `--mtp-model`. The training tooling exports in the same GGUF layout as the
public Q8_0 head (see [TRAIN-YOUR-OWN-DRAFTER.md](TRAIN-YOUR-OWN-DRAFTER.md)). A head that is re-quantised further
may need engine support for its tensor types.

## 4. Practical limits we hit

- **Device memory.** With 8 sessions x 32K context and a 9 GB prompt-cache RAM pin, peak dedicated GPU memory is
  about 93.5 GiB of the ~110 GiB visible to the device. 8 x 64K does not fit, so use 4 x 64K for long contexts.
- **Windows commit charge** is the binding host-side limit. A launch needs about 100.8 GiB of commit headroom, plus a
  reserve. Raise the pagefile if the server fails to start with all sessions.
- **Paging and background load** move results by 1-3 %. Antivirus scans, file indexing, search tools and remote
  desktop all count. Measure on a quiet machine, and see [METHODOLOGY.md](METHODOLOGY.md).
- **Thermals.** Long all-GPU runs on our desktop ran hot (high-80s C mean). Watch temperatures before 24/7 use.

---
Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).
