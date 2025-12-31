# manually targeting `../target-linux-x64/nsys` for the new nsys version. for some reason `nsys` keeps targeting my old 2024 version.
# use sudo nvidia-smi -lgc 2730 to lock freq scaling 

# profiling for pure performance timmings
ITERS=10000 PERF=1 /opt/nvidia/nsight-systems-cli/2025.6.1/target-linux-x64/nsys profile \
    -t cuda,osrt,nvtx \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    -o reports/inference \
    python3 examples/profile_inference.py

# profile for analyzing runs in depth
ITERS=100 PERF=0 /opt/nvidia/nsight-systems-cli/2025.6.1/target-linux-x64/nsys profile \
    -t cuda,osrt,nvtx \
    --cuda-trace-all-apis=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat:4 \
    --force-overwrite=true \
    -o reports/inference \
    python3 examples/profile_inference.py 

# `--capture-range-end` and `--force-overwrite` conflict. nsys will generate unique files names regardless.
# remove all the previous `inference.{n}.nsys-rep` in /reports