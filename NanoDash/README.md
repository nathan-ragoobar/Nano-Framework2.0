# Dashboard Application Setup Guide

## Dependencies

### Ubuntu/Linux
1. **Basic Build Tools**
```bash
sudo apt update
sudo apt install build-essential cmake git
```

2. **Qt6 Dependencies**
```bash
sudo apt install qt6-base-dev libqt6charts6-dev qt6-tools-dev
sudo apt install libqt6widgets6 libqt6charts6 libqt6core6
```

3. **Deployment Tools**
```bash
sudo apt install patchelf
```

### Windows (MSYS2)
1. Install MSYS2 from https://www.msys2.org/
2. Open MSYS2 terminal and run:
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-qt6 mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-qt6-charts
```

## Building the Project

### Ubuntu/Linux
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Run
./Dashboard
```

### Windows (MSYS2)
```bash
# From MSYS2 MinGW64 terminal
mkdir build && cd build

# Configure with CMake
cmake .. -G "MinGW Makefiles"

# Build
mingw32-make

# Run
./Dashboard.exe
```

## Project Structure
```
Dashboard/
├── CMakeLists.txt
├── main.cpp
├── mainwindow.cpp
├── mainwindow.h
├── mainwindow.ui
├── TrainingVisualizer.cpp
└── TrainingVisualizer.hpp
```

## Additional Notes

### Qt Path Configuration
- Linux: Qt path is usually auto-detected
- Windows: You might need to set Qt path in CMakeLists.txt:
```cmake
set(CMAKE_PREFIX_PATH "C:/Qt/6.8.0/mingw_64")
```

### Common Issues
1. **Qt not found**: Verify Qt installation and CMAKE_PREFIX_PATH
2. **Deployment fails**: 
   - Linux: Install patchelf
   - Windows: Ensure correct MinGW version matches Qt build

### For Development
VSCode extensions recommended:
- C/C++
- CMake Tools
- Qt Tools



###Compiling on Windows using MSYS2

```bash
pacman -Qs qt6
pacman -S mingw-w64-x86_64-qt6-charts
cmake ..
cmake --build .
```