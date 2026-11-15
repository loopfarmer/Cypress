# Cypress, maintained

This is a fork of [NVlabs/Cypress](https://github.com/NVlabs/Cypress), which hasn't moved since its ISPD 2025 release. It's kept for [edaroute](https://github.com/loopfarmer/edaroute) and changes Cypress in these ways:

- **Current software:** it builds and runs on PyTorch 2, Python 3.11 or later, NumPy 2 and C++20. It's tested on PyTorch 2.15. On macOS, PyTorch 2.5 doesn't load the ops: its OpenMP runtime lacks `__kmpc_dispatch_deinit`, which Homebrew's OpenMP headers call.
- **Macs:** it builds on macOS, including Apple silicon, CPU-only.
- **CPU ops:** the CPU pin-position and net-crossing ops have caught up with the CUDA ones. Their CPU versions had fallen behind and no longer ran. The CPU pin-position op now rotates parts too, so `enable_rotation` works on CPU.
- **Rotation gradient:** upstream's CUDA kernel had the wrong sign in one term of the angle gradient, which is wrong exactly at 90° and 270°. Both ops now have it right; the CPU one is checked against finite differences.
- **Faster net crossing:** pairs of segments that can't score are skipped before their gradient and atomics, as are pairs of nets too far apart to touch. On CPU that's 29 times faster on 8 threads, with the same gradient and the same placements. The CUDA kernel has the same skips, and adds each net's score atomically, where upstream's plain add raced.
- **Faster CPU density map:** each thread adds into its own map, so the op scales past 4 threads.
- **Stalled placements:** `legalize_max_overflow` lets the legalizer finish a global placement that stalled above `stop_overflow`, instead of failing it (see below).

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

- **Runs repeat only on one thread.** With more than one thread, floating-point sums land in a different order each run, so two runs with the same seed can differ.

### Known problems, on any device

- **Placements that stall fail, unless you allow legalization.** On these boards global placement often stalls a little above `stop_overflow`, with small parts trapped inside large ones and identical parts stacked at one spot. Upstream then logs "placement failed", skips legalization, writes the design's own placement as its `.gp.pl` and exits 0. With the default settings on one thread, 9 of the 20 released benchmarks do this. The fork adds `legalize_max_overflow`: global placements that end at or below it are legalized anyway, and the macro legalizer resolves the overlaps. With it at 0.5, all 20 benchmarks place legally. The default, 0.07, keeps upstream's behaviour.
- **Rotation barely searches.** `enable_rotation` is off by default upstream, and its search doesn't learn:
  - the orientation optimizer is `Adam` with `lr=0`, so the orientation logits never move, and the angle gradient goes unused;
  - from iteration 500, every 100 iterations it draws 100 random orientation sets. When one improves both wirelength and crossings, it keeps a fresh random draw, not the set it evaluated;
  - the first evaluations, before iteration 1, use random orientations too, so turning rotation on changes the whole run.

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