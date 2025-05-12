# LMOGA: LLM-Assisted Multi-Objective Genetic Algorithm Search-based ADS Testing

This project contains the implementation of LMOGA to test Apollo in LGSVL simulator. 
LMOGA applies multi-objective genetic algorithm to generate virtual scenarios which can find safety violations of ADSs.

The generation approach requires the following dependencies to run:

	1. SVL simulator: https://www.svlsimulator.com/
	
	2. Apollo autonomous driving platform: https://github.com/ApolloAuto/apollo


# Prerequisites

* A 8-core processor and 16GB memory minimum
* Ubuntu 18.04 or later
* Python 3.6 or higher
* NVIDIA graphics card: NVIDIA proprietary driver (>=455.32) must be installed
* CUDA upgraded to version 11.1 to support Nvidia Ampere (30x0 series) GPUs
* Docker-CE version 19.03 and above
* NVIDIA Container Toolkit


# SVL - A Python API for SVL Simulator

Documentation is available on: https://www.svlsimulator.com/docs/

# Apollo - A high performance, flexible architecture which accelerates the development, testing, and deployment of Autonomous Vehicles

Website of Apollo: https://apollo.auto/

Installation of Apollo: https://github.com/ApolloAuto/apollo/blob/master/docs/01_Installation%20Instructions/apollo_software_installation_guide.md

# Run

- To get started, launch and run Apollo (details about how to run is in Installation of Apollo)
- Run LGSVL with all the assets like vehicles, maps,... get from SORA-SVL: https://github.com/YuqiHuai/SORA-SVL
- Config `APOLLO_HOST` and `BRIDGE_PORT` in `simulator.py` to connect Apollo and LGSVL through `bridge.sh`
- To run the search process, config the parameters in `start_experiment.py` and run that file
- To replay the recorded safety-violation scenarios, execute the `main()` of `replay.py` and set the file path of the scenario to be replayed.


