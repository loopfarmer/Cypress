# Cypress, maintained

This is a fork of [NVlabs/Cypress](https://github.com/NVlabs/Cypress), which hasn't moved since its ISPD 2025 release. It's kept for [edaroute](https://github.com/loopfarmer/edaroute) and changes Cypress in three ways:

- **Current software:** it builds and runs on PyTorch 2 (tested on 2.5 and 2.15), Python 3.11 or later, NumPy 2 and C++20.
- **Macs:** it builds on macOS, including Apple silicon, CPU-only.
- **CPU ops:** the CPU pin-position and net-crossing ops have caught up with the CUDA ones. Their CPU versions had fallen behind and no longer ran.

The commit messages say what changed and why. Upstream's README follows below.

## Build on macOS, CPU-only

```bash
brew install cmake boost bison flex libomp cairo
git clone --recursive https://github.com/loopfarmer/Cypress.git && cd Cypress
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$PWD/../.venv/bin/python && make -j && make install
```

CMake finds Homebrew's bison, flex and libomp by itself.

## Run on macOS

```bash
cd install
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ../.venv/bin/python dreamplace/Placer.py config.json
```

Set `"gpu": 0` in the config.

- **`DYLD_FALLBACK_LIBRARY_PATH`** lets cairocffi find Homebrew's cairo. macOS strips it from any program under `/usr/bin`, so don't run the placer through `/usr/bin/time`.

The ops are compiled with Homebrew's OpenMP headers but use PyTorch's OpenMP runtime, so only one runtime is ever loaded. Two runtimes deadlock with PyTorch 2.15, and hang at exit with 2.5.

### Limits on CPU

- **No rotation.** The CPU op doesn't rotate parts, so keep `enable_rotation` at 0.
- **Runs aren't repeatable.** With more than one thread, floating-point sums land in a different order each run, so two runs with the same seed can differ.

---

# Cypress: VLSI-Inspired Scalable GPU-Accelerated PCB Placement

Cypress is a scalable, GPU-accelerated PCB placement method inspired by VLSI. It incorporates
tailored cost functions, constraint handling, and optimized techniques adapted for PCB layouts.

<p align="center">
  <img src="images/big-3.gif" width="60%" alt="Cypress PCB Placement Animation">
</p>


## How to Build

You can build in two ways:

- Build without Docker by following the instructions of the DREAMPlace build at README_DREAMPlace.md.
- Use the provided Dockerfile to build an image with the required library dependencies.

## Run PCB Placement

```sh
cd /Cypress
python dreamplace/Placer.py path/to/your/config.json
```

## Run Hyperparameter search

```sh
./tuner/run_tuner.sh 1 1 	test/tune/pcb-configspace.json 	/path/to/aux_file 	test/tuner/tuner-ppa.json "" 	100 4 0 0 10 	./tuner 	./results/result_path
```

## Acknowledgement

Cypress builds upon and extends the following open-source projects:

- [DREAMPlace](https://github.com/limbo018/DREAMPlace): A deep learning toolkit-enabled VLSI placement framework that provides the foundation for our GPU-accelerated placement engine.

- [AutoDMP](https://github.com/NVlabs/AutoDMP): An automated design methodology platform that inspired aspects of our hyperparameter tuning approach.

We thank the authors and contributors of these projects for their valuable work that made Cypress possible.

## Citation

If you find this work useful, please cite the following paper:

```bibtex
@inproceedings{zhang2025cypress,
  title={Cypress: VLSI-Inspired PCB Placement with GPU Acceleration},
  author={Zhang, Niansong and Agnesina, Anthony and Shbat, Noor and Leader, Yuval and Zhang, Zhiru and Ren, Haoxing},
  booktitle={Proceedings of the 2025 International Symposium on Physical Design},
  pages={xx--xx},
  year={2025}
}
```