# Final Project for 6.5931 (Spring 2025): Modeling Rack-Scale Data and ZeRO Parallelism

## Overview

This repository contains the final project for **6.5931: Hardware Architecture for Deep Learning** at MIT, completed in Spring 2025.  
The project builds upon the base template provided in the course lab repository and has been extended to use **Timeloop** and **Accelergy** to model and analyze **Data Parallelism** and **ZeRO Parallelism** at rack scale.

## Project Structure

All relevant files are located in the `workspace/final_project/` directory:

- `main.ipynb` – Main Jupyter notebook to execute simulations and generate plots
- `utils/` – Helper functions and utilities used across the project
- `plots/` – Directory where output plots are saved
- `layer_shapes/` – Contains the base workload descriptions
- `designs/` – Contains architectural definitions for modeling

## Getting Started with Docker

Before starting, set the appropriate `USER_ID` and `USER_GID` in `docker-compose.yaml`,  and `DOCKER_ARCH` in your environment. Then launch the container using:

```bash
docker compose up
