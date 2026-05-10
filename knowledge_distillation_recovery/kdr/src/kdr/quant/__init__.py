"""Quantization simulation backends for kdr.

Public surface:
* `kdr.quant.specs`: typed Pydantic specs for K, V, and weight quantization.
* `kdr.quant.interface`: `QuantBackend` Protocol + `QuantBlockSubset` typed model.
* `kdr.quant.factory`: per-backend partition-and-dispatch entry point.
* `kdr.quant.modelopt_backend`: default backend wrapping NVIDIA modelopt.
* `kdr.quant.native_backend`: hand-rolled STE simulators filling modelopt gaps.
"""
