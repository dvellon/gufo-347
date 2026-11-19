# Installing gufo-347 on Windows

gufo-347 R8 is upstream gufo v0.35.0 plus the patches in this repository, built for Windows x64 and the AMD Radeon
8060S / 8050S (gfx1151, "Strix Halo", Ryzen AI Max) with TheRock 10.0.0. It runs the launch profiles that were
benchmarked in run 294. Model files are not included.

## What you need

| | |
|---|---|
| Windows | Windows 11 x64 (the build is tested on Windows 11 only) |
| GPU | AMD Radeon 8060S / 8050S (gfx1151). Other GPUs are not supported by this build |
| Driver | a current AMD Software: Adrenalin Edition driver. The release was tested with Windows driver version 32.0.31041.5005 (Device Manager > Display adapters > Radeon > Driver) |
| Memory | 128 GiB, with about 110 GiB visible to the GPU (see [Memory settings](#memory-settings)) |
| Runtime | Microsoft Visual C++ 2015-2022 x64 runtime ([vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)) |
| Disk | 0.5 GB for the program, about 114 GB for the model files |
| NPU | not needed. The launch profiles run on the GPU only |

The HIP runtime, rocBLAS / hipBLASLt and their gfx1151 kernels are inside the release zip; you do not need a HIP
SDK.

## Install with the installer script (recommended)

From the [release page](https://github.com/dvellon/gufo-347/releases), download `install-gufo-347.ps1`. Then, in a
normal (not administrator) PowerShell window:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-gufo-347.ps1
```

The script:

1. checks the machine (Windows, GPU, driver, memory, carve-out, pagefile, Visual C++ runtime, disk space) and stops
   on a blocking problem;
2. downloads `gufo-347-r8-windows-x64-gfx1151.zip` and `SHA256SUMS` from this repository's GitHub release, and
   installs only if the zip's sha256 matches (it also checks every unpacked file against the zip's
   `MANIFEST-SHA256.txt`);
3. installs to `%LOCALAPPDATA%\gufo-347` (no administrator rights needed), puts a `gufo-347` command on your user
   PATH, adds Start-menu shortcuts (one per launch profile) and an entry in Settings > Apps > Installed apps;
4. asks for the folder of your model files, or tells you where to get them.

Useful options:

| option | what it does |
|---|---|
| `-ZipPath .\gufo-347-r8-windows-x64-gfx1151.zip` | install from a zip you already downloaded (put `SHA256SUMS` next to it) |
| `-Tag <tag>` | download from a specific release instead of the latest |
| `-ModelDir D:\models\Qwen3.8-Flash-Next-GGUF` | set the model folder without being asked |
| `-Dest C:\gufo-347` | install somewhere else (keep the path short) |
| `-CheckOnly` | only run the machine checks |
| `-VerifyModelHashes` | also check the sha256 of the model files (reads 114 GB) |
| `-NoPath`, `-NoShortcuts`, `-NoAppsEntry` | skip the PATH entry, the shortcuts, or the Installed apps entry |
| `-NonInteractive` | never ask questions |

Running the script again upgrades in place: the new version is unpacked and verified before the old one is
replaced, and your model folder setting is kept. Run `Get-Help .\install-gufo-347.ps1 -Detailed` for all options.

## Install by hand

Download the zip and `SHA256SUMS`, check the hash, unpack to a short folder:

```powershell
(Get-FileHash .\gufo-347-r8-windows-x64-gfx1151.zip -Algorithm SHA256).Hash.ToLower()
Get-Content .\SHA256SUMS        # the two must match
Expand-Archive .\gufo-347-r8-windows-x64-gfx1151.zip -DestinationPath C:\gufo-347
```

Then use `run.cmd` in the unpacked folder instead of the `gufo-347` command (see `QUICKSTART.txt` in the zip).

## Get the model files

About 114 GB from Hugging Face, [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
at revision `38bb39ee97821de2c9009abb7e93950eec396e66`: the UD-Q4_K_XL quantisation (4 files) and the public shared
Q8_0 MTP draft head. Keep the repository layout:

```powershell
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 `
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" --local-dir D:\models\Qwen3.8-Flash-Next-GGUF
gufo-347 set-model D:\models\Qwen3.8-Flash-Next-GGUF
```

The sha256 of the five files are listed in the zip's `QUICKSTART.txt`; `-VerifyModelHashes` checks them.

## Run and test

Open a new terminal (so it sees the new PATH), then:

```
gufo-347 agent-1-session
```

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API, model name `flash-next`). In a second terminal:

```
gufo-347 test
```

It sends one short chat request and prints the reply. Other commands: `gufo-347 list` (profiles),
`gufo-347 agent-8-sessions -Port 8081`, `gufo-347 where`, `gufo-347 version`, `gufo-347 help`. The Start-menu
shortcuts open a window running one profile; closing the window stops the server.

| profile | for | exactness |
|---|---|---|
| `agent-1-session`, `agent-8-sessions` | one / up to 8 coding agents | strict: the model's own greedy output |
| `chat` | interactive chat (1 session) | strict |
| `canonical-bench` | reproducing upstream's tg128 benchmark row | strict |
| `relaxed-agent-1-session`, `relaxed-agent-8-sessions`, `relaxed-chat` | the same, faster | **non-standard**: near-tie acceptance, output can differ from greedy |

The agent and chat profiles have no default reply cap: send `max_tokens` with each request.

## Uninstall

`gufo-347 uninstall`, or Settings > Apps > Installed apps > gufo-347, or run
`%LOCALAPPDATA%\gufo-347\uninstall-gufo-347.ps1`. This removes the program, the PATH entry, the shortcuts and the
Installed apps entry. Your model files are never touched.

## Memory settings

The 8-session profiles (both agent profiles and their relaxed versions) need about 93.5 GiB of GPU memory at peak,
so about 110 GiB must be visible to the GPU:

- GPU memory carve-out: set the dedicated GPU memory to 96 GB, in the UEFI/BIOS ("UMA frame buffer size" or similar)
  or in AMD Software ("Variable Graphics Memory"), depending on the machine. Windows then shows about 96 GiB
  dedicated plus half of the remaining RAM as shared GPU memory.
- Commit limit: a launch needs about 101 GiB of Windows commit headroom plus a reserve. With a 96 GB carve-out only
  about 32 GiB of RAM is left to Windows, so raise the pagefile (System > About > Advanced system settings >
  Performance > Settings > Advanced > Virtual memory), for example to 96 GB or more, and restart.
- The `chat` profiles use 1 session and need less memory.

The installer's machine check (`-CheckOnly`) shows the installed memory, the GPU-visible memory and the commit limit.

## Troubleshooting

**The installer says no gfx1151 GPU was found.** Check Device Manager > Display adapters: it must show AMD Radeon
8060S or 8050S. If it shows "Microsoft Basic Display Adapter", install the AMD Adrenalin driver. This build has no
GPU code for other GPUs.

**gufo starts but finds no GPU, or fails in the HIP runtime.** Update AMD Software: Adrenalin Edition; the release
was tested with driver 32.0.31041.5005. Make sure `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` are not set to
hide the GPU. gufo-347 uses the HIP runtime in its own `app\bin` folder, so an installed HIP SDK (`HIP_PATH`) is not
needed and not used. Do not add `app\bin` to PATH and do not copy its DLLs elsewhere.

**"msvcp140.dll / vcruntime140_1.dll was not found".** Install the Visual C++ 2015-2022 x64 runtime (link above).

**Messages about `xrt_coreutil.dll` or the NPU.** The NPU is optional. AMD's NPU driver provides `xrt_coreutil.dll`;
gufo loads it only for NPU work, which the launch profiles do not use.

**The server fails to start, or runs out of memory.** See [Memory settings](#memory-settings).

**"The install folder is too long".** The release contains paths of up to 132 characters; use a short folder, for
example `-Dest C:\gufo-347`.

**PowerShell refuses to run the script.** Use the command shown above (`-ExecutionPolicy Bypass` applies to that one
run only), or run `Unblock-File .\install-gufo-347.ps1` first.

**Port 8080 is in use.** Add `-Port 8081` (and `gufo-347 test -Port 8081`).

**Heat.** Long all-GPU runs ran hot on our desktop (high-80s C mean). Watch temperatures before running 24/7.

The installer writes `install.log` in the install folder.
