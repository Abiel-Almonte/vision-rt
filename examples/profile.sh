nsys profile \
    -t cuda,osrt,nvtx \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    -o inference \
    python3 examples/profile_inference.py 
