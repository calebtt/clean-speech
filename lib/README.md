# Vendored LocalVQE GGML engine

`liblocalvqe.so` + its `libggml*` dependencies are the prebuilt GGML inference
engine for **LocalVQE** (reference-aware AEC + NS + dereverb), used by the
`hybrid_localvqe` echo canceller. The PyTorch reference of LocalVQE is *not*
chunk-streamable (whole-utterance normalization), but this engine's
`localvqe_process_frame_f32()` is bit-identical batch-vs-frame, so it runs in the
realtime loop.

The model weights are `../models/localvqe-v1.3-4.8M-f32.gguf`.

Source: https://github.com/localai-org/LocalVQE (Apache-2.0). Rebuild:

```bash
git clone --recurse-submodules https://github.com/localai-org/LocalVQE
cd LocalVQE/ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_BUILD_SHARED=ON \
      -DLOCALVQE_CUDA=OFF -DLOCALVQE_VULKAN=OFF
cmake --build build --target localvqe_shared -j
# copy build/bin/lib{localvqe,ggml*}.so* here
```

`ggml` discovers the per-CPU backend variants (`libggml-cpu-*.so`) in this
directory at load time (the Python binding sets `GGML_BACKEND_PATH`), so the
folder is self-contained.
