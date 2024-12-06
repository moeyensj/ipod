# IPOD: Iterative Precovery and Differential Correction of Asteroid Orbits

## Overview

IPOD is a Python package designed for the iterative precovery and differential correction of asteroid orbits. It leverages advanced algorithms to refine orbit determinations by incorporating new observations and correcting existing ones. The package is particularly useful for astronomers and researchers working in the field of asteroid tracking and orbit determination.

## Features

- **Iterative Precovery**: Efficiently searches for and incorporates historical observations of known asteroids.
- **Differential Correction**: Refines orbit parameters using new and existing observations.
- **Parallel Processing**: Utilizes Ray for distributed computing, allowing for scalable processing of large datasets.
- **Integration with Precovery Database**: Seamlessly connects with precovery databases to fetch and store observational data.

## Installation

To install IPOD, ensure you have Python 3.11 or later, and use the following command:
```bash
pdm install
```

For testing and development, you can install additional dependencies:
```bash
pdm install -G tests
```

## Usage

### Basic Usage

To use IPOD for orbit determination, you can call the main function `iterative_precovery_and_differential_correction` with the required parameters:
```python
from ipod.main import iterative_precovery_and_differential_correction
```
Example usage:
```python
orbits, orbit_members, precovery_candidates, search_summary = iterative_precovery_and_differential_correction(
    orbits=your_orbits_data,
    observations=your_observations_data,
    database_directory="path/to/database",
    max_processes=4
)
```
