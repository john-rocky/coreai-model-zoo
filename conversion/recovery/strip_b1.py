#!/usr/bin/env python3
"""strip_b1.py — strip debug locations from a coreai-torch 0.4.0-era .aimodel, in the b1 venv.

    _recovery_venvs/strip/bin/python strip_b1.py <in.aimodel> <out.aimodel>

Vendored from coreai-torch 0.4.1 `debugging/debug_info.strip_debug_info` (0.4.0 lacks it).
Runs in the coreai-torch 0.4.0 + coreai-core 1.0.0b1 venv, whose bytecode reader still parses
the old fused locations. The output must then be re-saved once with the b2 wheel (resave_b2.py)
so metadata.json carries `producer: coreai-core 1.0.0b2`.
"""
import inspect, sys, time
from pathlib import Path

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Location
from coreai.authoring import AIProgram
import coreai_torch._debug_locations as dl

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
program = AIProgram._load_bytecode(src / "main.mlirb")
print(f"loaded {src.name} in {time.time()-t0:.1f}s")

module_op = program._mlir_module.operation
context = module_op.context

# 0.4.0 helper signatures (differ from 0.4.1+): _create_unknown_location(context, metadata_attrs=None),
# _create_operation_id_metadata(op_id, context). No unknown_src argument.
module_location = dl._create_unknown_location(context)
dl.set_op_location(module_op, module_location)

count = 0
for nested_op in dl._get_nested_operations(module_op):
    op_id = dl.OperationID(type="coreai", value=count)
    count += 1
    loc = dl._create_unknown_location(context, [dl._create_operation_id_metadata(op_id, context)])
    dl.set_op_location(nested_op, loc)
    for region in nested_op.regions:
        for block in region:
            for arg in block.arguments:
                dl.set_block_arg_location(arg, loc)
print(f"relocated {count} ops in {time.time()-t0:.1f}s")

import shutil
shutil.rmtree(dst, ignore_errors=True)
program.save_asset(dst)
print(f"saved {dst} in {time.time()-t0:.1f}s")
