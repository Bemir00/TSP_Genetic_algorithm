# Travelling Salesman Problem

This project compares three simple ways of finding a short closed tour through a list of cities:

- Random search with 100 random tours.
- Nearest-neighbour greedy search from every possible starting city.
- A genetic algorithm.

The tour visits every city once and returns to the first city. Tour length is calculated with Euclidean distance.

## Genetic algorithm

The code in `src/main.py` uses:

- A permutation of city IDs as a tour.
- Tournament selection, with a default tournament size of 3.
- Ordered crossover (OX).
- Swap mutation or inversion mutation.
- Euclidean tour length as fitness. Lower is better.
- Elitism, keeping the best tour for the next generation.

The main run uses 200 generations, a population of 100, and inversion mutation.

## Results

| Instance | Cities | Random (best of 100) | Greedy | Genetic algorithm | Known optimum | Gap |
|---|---:|---:|---:|---:|---:|---:|
| berlin52 | 52 | 26,202 | 8,182 | **8,114** | 7,542 | 7.6% |
| kroA100 | 100 | — | 24,698 | **22,535** | 21,282 | 5.9% |
| kroA150 | 150 | — | 31,482 | **31,362** | 26,524 | 18.2% |

The gap is larger for `kroA150`. The larger search space would need more generations or a larger population to get closer to the known optimum.

## Parameter comparison

`run_param_comparison()` compares:

- Population size.
- Crossover probability.
- Mutation probability.
- Mutation type: swap or inversion.

The study uses shorter runs than the main experiment.

## Generated charts

The program writes these files to `results/`:

- Convergence curves.
- Algorithm comparison bar charts.
- Best tour plots.
- Parameter comparison charts.

The folder is recreated when the program starts.

## Input files

The included files use the TSPLIB `NODE_COORD_SECTION` format:

- `berlin 11 tsp.txt`
- `berlin52 tsp.txt`
- `kroA100 tsp.txt`
- `kroA150 tsp.txt`

## Running

The program reads the instance files relative to the current working directory. Run it from `src`:

```bash
cd src
python -m venv .venv
pip install -r requirements.txt
python main.py
```

## Tech stack

Python, NumPy, pandas, Matplotlib.
