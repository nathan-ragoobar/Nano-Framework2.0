# nano-distributed/nano-distributed/README.md

# Nano Distributed Computing Framework

## Overview

The Nano Distributed Computing Framework is a lightweight library designed for implementing distributed computing solutions using data parallelism and model parallelism. It provides a set of tools and abstractions for efficient communication and synchronization between distributed processes.

## Features

- **Communication**: Supports both MPI (Message Passing Interface) and TCP socket communication.
- **Data Parallelism**: Implements strategies for distributing data across multiple workers and aggregating results.
- **Model Parallelism**: Provides functionalities for distributing model parameters and computations across multiple devices.
- **Parameter Server**: Manages shared model parameters in a distributed setting.
- **Synchronization**: Includes mechanisms for barrier synchronization and mutual exclusion.
- **Utilities**: Offers serialization, deserialization, and timing functionalities.

## Installation

To use the Nano Distributed Computing Framework, clone the repository and build the project using CMake:

```bash
git clone <repository-url>
cd nano-distributed
mkdir build
cd build
cmake ..
make
```

## Usage

### Data Parallel Training Example

To see how to implement data parallel training using the library, refer to the `examples/data_parallel_training.cpp` file.

### Parameter Server Example

For an example of managing model parameters in a distributed setting, check the `examples/parameter_server.cpp` file.

## API Reference

The library is organized into several components:

- **Communication**:
  - `mpi_wrapper.hpp`: Wrapper around MPI functions.
  - `tcp_socket.hpp`: TCP socket class for network communication.
  - `communicator.hpp`: Abstracts communication methods.

- **Parallelism**:
  - `data_parallel.hpp`: Implements data parallelism strategies.
  - `model_parallel.hpp`: Implements model parallelism strategies.
  - `parameter_server.hpp`: Manages shared model parameters.

- **Synchronization**:
  - `barrier.hpp`: Implements barrier synchronization.
  - `mutex.hpp`: Defines a Mutex class for mutual exclusion.
  - `atomic.hpp`: Provides atomic operations for thread-safe programming.

- **Utilities**:
  - `serialization.hpp`: Implements serialization and deserialization functions.
  - `timer.hpp`: Provides a Timer class for measuring elapsed time.

## Testing

Unit tests are provided for each component of the library. To run the tests, navigate to the `build` directory and execute:

```bash
make test
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.