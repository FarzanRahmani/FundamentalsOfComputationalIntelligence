# Fundamentals of Computational Intelligence

Theory submissions, Python notebooks, and optimization examples from my seventh-semester B.Sc. course at Iran University of Science and Technology, taught by Dr. Naser Mozayani. Author: Farzan Rahmani.

## Contents

| Path | Material |
| --- | --- |
| `HW1/` | First assignment statement and numerical-computing notebook |
| `HW2/` | Multiple practical notebooks, including TensorFlow/Keras work, plus the assignment and report |
| `HW3/` | Written submission and NumPy/Matplotlib practical notebook |
| `HW4/` | Assignment documents and written work |
| `HW5/` | Fuzzy-system exercises, including scikit-fuzzy and Gymnasium experiments, theory material, and TA notebooks |
| `HW6/` | Genetic-algorithm polynomial root-finding experiments and related scripts |
| `PSO_implementation/PSO.py` | Worked particle-swarm optimization example with two particles and explicit update steps |

## Getting Started

Use Python and Jupyter for the notebooks. Dependencies vary by assignment: NumPy, pandas, and Matplotlib appear in numerical work; neural-network exercises use TensorFlow/Keras; fuzzy-control examples use scikit-fuzzy and Gymnasium. Some notebooks download external data using `gdown`.

Install the packages required by the chosen notebook and check its input paths before running cells. No single pinned environment is provided, and notebooks may contain paths or setup cells from the original machine or hosted runtime.

For a small standalone example requiring only Python, run from the repository root:

```sh
python PSO_implementation/PSO.py
```

The PSO script prints particle positions, velocities, and the global best for the objective `g(x) = -(x - 5)^2 + 10`. It demonstrates fixed updates rather than exposing a general-purpose optimization API.

## Reading the Work

Use each assignment PDF to interpret the associated notebook and submitted report. Run notebook cells in order and review hyperparameters before long experiments. Written explanations include Persian text, and saved outputs reflect the original environment and random seeds.

## License

See [LICENSE](LICENSE) for license terms. Teaching-assistant materials retain their original attribution.
