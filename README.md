Robust Ellipsoid RRT, a robust approach to sampling based planning methods.
Full documentation nearing completion.

If you have drake installed locally, add this directory to your python path 
```shell
user@userpc:~$ export PYTHONPATH="${PYTHONPATH}:pathtorerrt/rerrt"
```
and run examples can be run with
```python
python3 examples/[example_file].py
```
Current examples in the examples/ dir include RERRT and RRT for both a 2D car system and a furuta pendulum system. These examples grow RRT/RERRT trees for the systems and visualize them.

An example of a simple assessment of the robustness of the system is provided as well for the car and furuta pendulum systems, and can be run with
```
python3 robusttests/car_test.py
python3 robusttests/furuta_test.py
```
The script grows RRT and RERRT trees in the same manner as in the examples files, then simulates the trees with uncertainty and outputs brief statistics on whether trajectories remain in the valid state space and whether they come within a set epsilon of the desired final state.

Example output and results are provided below from car_test.py, however a code update with a more precise integration scheme is necessary for significant results for the furuta system, due to the highly sensitive nature of the dynamics which currently suffer from numerics errors in simulation.
