#!/usr/bin/env python3
"""resave_b2.py — re-load a stripped bundle with the b2 wheel and re-save it (producer stamp), then load it with the runtime (cpu_only).

    coreai-models/.venv/bin/python resave_b2.py <stripped.aimodel> <final.aimodel>
"""
import asyncio, json, shutil, sys, time
from pathlib import Path
from coreai.authoring import AIModelAsset
import coreai.runtime as rt

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
asset = AIModelAsset.load(src)
shutil.rmtree(dst, ignore_errors=True)
asset.program.save_asset(dst, rt.AIModelAssetMetadata())
print(f"resaved in {time.time()-t0:.1f}s; metadata:", json.loads((dst / "metadata.json").read_text()))

async def main():
    m = await rt.AIModel.load(dst, rt.SpecializationOptions.cpu_only())
    print("runtime load OK; functions:", m.function_names)
asyncio.run(main())
print(f"done in {time.time()-t0:.1f}s")
