#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix CUBLAS on NVIDIA Jetson (Thor, Orin, etc.) for PyTorch.

PyTorch's pip-bundled libcublas.so may be incompatible with Jetson's GPU architecture
(e.g., SM 11.0 on Thor). This script replaces torch's bundled CUBLAS with the system
CUBLAS from JetPack, which is compiled for the correct SM target.

Usage:
    strands-cosmos-fix-cublas          # auto-detect and fix
    strands-cosmos-fix-cublas --check  # check only, don't fix
    strands-cosmos-fix-cublas --revert # restore original bundled CUBLAS

What it does:
    1. Detects if running on Jetson (aarch64 + /usr/local/cuda)
    2. Tests torch.mm() on CUDA — if CUBLAS_STATUS_INVALID_VALUE, fix is needed
    3. Backs up torch's bundled libcublas*.so as .ORIG_BUNDLED
    4. Copies system CUBLAS into torch's lib directory
    5. Verifies the fix works
"""

import glob
import os
import platform
import shutil
import sys


def find_torch_cublas_dir():
    """Find torch's bundled CUBLAS directory."""
    try:
        import torch
        torch_lib = os.path.dirname(torch.__file__)
        # torch bundles CUBLAS via nvidia-cu* packages
        # Pattern: .../site-packages/nvidia/cu*/lib/
        site_packages = os.path.dirname(os.path.dirname(torch_lib))
        candidates = glob.glob(os.path.join(site_packages, "nvidia", "cu*", "lib"))
        for candidate in sorted(candidates, reverse=True):
            if os.path.exists(os.path.join(candidate, "libcublas.so.13")):
                return candidate
            if os.path.exists(os.path.join(candidate, "libcublas.so.12")):
                return candidate
        # Fallback: check torch/lib directly
        torch_lib_dir = os.path.join(torch_lib, "lib")
        for f in os.listdir(torch_lib_dir):
            if f.startswith("libcublas.so"):
                return torch_lib_dir
    except Exception:
        pass
    return None


def find_system_cublas():
    """Find system CUBLAS libraries (from JetPack)."""
    search_paths = [
        "/usr/local/cuda/targets/sbsa-linux/lib",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib/aarch64-linux-gnu",
    ]
    # Also check versioned CUDA dirs
    for cuda_dir in sorted(glob.glob("/usr/local/cuda-*/"), reverse=True):
        search_paths.insert(0, os.path.join(cuda_dir, "targets", "sbsa-linux", "lib"))
        search_paths.insert(1, os.path.join(cuda_dir, "targets", "aarch64-linux", "lib"))
        search_paths.insert(2, os.path.join(cuda_dir, "lib64"))

    for path in search_paths:
        cublas_files = glob.glob(os.path.join(path, "libcublas.so.*"))
        cublaslt_files = glob.glob(os.path.join(path, "libcublasLt.so.*"))
        if cublas_files and cublaslt_files:
            # Find the actual .so file (not symlinks to other symlinks)
            cublas = sorted([f for f in cublas_files if not os.path.islink(f) or os.path.exists(f)], key=len)[-1]
            cublaslt = sorted([f for f in cublaslt_files if not os.path.islink(f) or os.path.exists(f)], key=len)[-1]
            # Resolve symlinks to get actual files
            cublas = os.path.realpath(cublas)
            cublaslt = os.path.realpath(cublaslt)
            if os.path.exists(cublas) and os.path.exists(cublaslt):
                return cublas, cublaslt, path
    return None, None, None


def test_cublas():
    """Test if CUBLAS works by running torch.mm on CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "CUDA not available"
        a = torch.randn(4, 4, device="cuda")
        b = torch.randn(4, 4, device="cuda")
        c = torch.mm(a, b)
        return True, f"OK — torch.mm works ({c.shape})"
    except RuntimeError as e:
        if "CUBLAS_STATUS_INVALID_VALUE" in str(e):
            return False, f"CUBLAS broken: {e}"
        return False, f"CUDA error: {e}"
    except Exception as e:
        return None, f"Cannot test: {e}"


def check():
    """Check CUBLAS status and print diagnostics."""
    print("🔍 strands-cosmos CUBLAS diagnostics")
    print("=" * 50)

    # Platform
    print(f"Platform: {platform.system()} {platform.machine()}")
    is_jetson = platform.machine() in ("aarch64", "arm64") and os.path.exists("/usr/local/cuda")
    print(f"Jetson detected: {'✅ Yes' if is_jetson else '❌ No'}")

    # Torch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"SM: {torch.cuda.get_device_capability(0)}")
    except ImportError:
        print("PyTorch: not installed")
        return

    # Torch's bundled CUBLAS
    torch_dir = find_torch_cublas_dir()
    if torch_dir:
        print(f"Torch CUBLAS dir: {torch_dir}")
        has_backup = any(f.endswith(".ORIG_BUNDLED") for f in os.listdir(torch_dir))
        if has_backup:
            print("  ⚠️  Backup found (.ORIG_BUNDLED) — fix was previously applied")
    else:
        print("Torch CUBLAS dir: not found")

    # System CUBLAS
    cublas, cublaslt, sys_path = find_system_cublas()
    if cublas:
        print(f"System CUBLAS: {cublas} ({os.path.getsize(cublas) / 1e6:.1f} MB)")
        print(f"System cublasLt: {cublaslt} ({os.path.getsize(cublaslt) / 1e6:.1f} MB)")
    else:
        print("System CUBLAS: not found")

    # Test
    print()
    ok, msg = test_cublas()
    if ok is True:
        print(f"✅ CUBLAS test: {msg}")
    elif ok is False:
        print(f"❌ CUBLAS test: {msg}")
        if is_jetson and cublas:
            print("\n💡 Fix available! Run: strands-cosmos-fix-cublas")
    else:
        print(f"⚠️  CUBLAS test: {msg}")

    return ok


def fix():
    """Replace torch's bundled CUBLAS with system CUBLAS."""
    print("🔧 Fixing CUBLAS for Jetson...")
    print("=" * 50)

    # Find torch's CUBLAS
    torch_dir = find_torch_cublas_dir()
    if not torch_dir:
        print("❌ Could not find torch's CUBLAS directory")
        sys.exit(1)

    # Find system CUBLAS
    cublas_src, cublaslt_src, sys_path = find_system_cublas()
    if not cublas_src:
        print("❌ Could not find system CUBLAS")
        print("   Ensure NVIDIA JetPack / CUDA toolkit is installed")
        sys.exit(1)

    print(f"Torch CUBLAS dir: {torch_dir}")
    print(f"System CUBLAS:    {sys_path}")

    # Detect the .so version in torch's dir
    cublas_so = None
    cublaslt_so = None
    for f in os.listdir(torch_dir):
        if f.startswith("libcublas.so.") and not f.endswith(".ORIG_BUNDLED"):
            cublas_so = f
        if f.startswith("libcublasLt.so.") and not f.endswith(".ORIG_BUNDLED"):
            cublaslt_so = f

    if not cublas_so or not cublaslt_so:
        print("❌ Could not find libcublas*.so in torch directory")
        sys.exit(1)

    cublas_dst = os.path.join(torch_dir, cublas_so)
    cublaslt_dst = os.path.join(torch_dir, cublaslt_so)

    # Backup originals
    cublas_backup = cublas_dst + ".ORIG_BUNDLED"
    cublaslt_backup = cublaslt_dst + ".ORIG_BUNDLED"

    if not os.path.exists(cublas_backup):
        print(f"  Backing up {cublas_so} → {cublas_so}.ORIG_BUNDLED")
        shutil.copy2(cublas_dst, cublas_backup)
    else:
        print(f"  Backup already exists: {cublas_so}.ORIG_BUNDLED")

    if not os.path.exists(cublaslt_backup):
        print(f"  Backing up {cublaslt_so} → {cublaslt_so}.ORIG_BUNDLED")
        shutil.copy2(cublaslt_dst, cublaslt_backup)
    else:
        print(f"  Backup already exists: {cublaslt_so}.ORIG_BUNDLED")

    # Copy system CUBLAS
    print(f"  Copying system libcublas → {cublas_dst}")
    shutil.copy2(cublas_src, cublas_dst)

    print(f"  Copying system libcublasLt → {cublaslt_dst}")
    shutil.copy2(cublaslt_src, cublaslt_dst)

    print()
    print("✅ CUBLAS replaced with system version")
    print()

    # Verify
    print("Verifying fix...")
    ok, msg = test_cublas()
    if ok is True:
        print(f"✅ {msg}")
        print("\n🎉 CUBLAS fix successful! Cosmos models will now work on this Jetson.")
    elif ok is False:
        print(f"❌ {msg}")
        print("\n⚠️  Fix did not resolve the issue. You may need to revert:")
        print("   strands-cosmos-fix-cublas --revert")
    else:
        print(f"⚠️  {msg}")


def revert():
    """Revert to original bundled CUBLAS."""
    print("↩️  Reverting CUBLAS to original bundled version...")

    torch_dir = find_torch_cublas_dir()
    if not torch_dir:
        print("❌ Could not find torch's CUBLAS directory")
        sys.exit(1)

    reverted = 0
    for f in os.listdir(torch_dir):
        if f.endswith(".ORIG_BUNDLED"):
            original_name = f.replace(".ORIG_BUNDLED", "")
            src = os.path.join(torch_dir, f)
            dst = os.path.join(torch_dir, original_name)
            print(f"  Restoring {original_name}")
            shutil.copy2(src, dst)
            os.remove(src)
            reverted += 1

    if reverted:
        print(f"\n✅ Reverted {reverted} files to original bundled CUBLAS")
    else:
        print("No backups found — nothing to revert")


def main():
    """CLI entry point."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if "--revert" in sys.argv:
        revert()
    elif "--check" in sys.argv:
        ok = check()
        sys.exit(0 if ok else 1)
    else:
        # Auto-detect: check first, fix if needed
        ok, msg = test_cublas()
        if ok is True:
            print("✅ CUBLAS is already working — no fix needed")
            print(f"   {msg}")
        elif ok is False:
            print(f"❌ {msg}")
            print()
            fix()
        else:
            print(f"⚠️  {msg}")
            print("Cannot determine if fix is needed. Run with --check for diagnostics.")


if __name__ == "__main__":
    main()
