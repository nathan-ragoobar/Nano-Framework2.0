## Compiling on Windows
I have tested two ways to compile on windows. Using GitBash and Visual Studio developer prompt.\
Note: For some reason, it takes an entire hour to compile on Windows vs a few minutes on Ubuntu.\
Even after it compiles it still trains slower than on Ubuntu on WSL\

### Compiling in Windows using GitBash/MSYS2
NB: I used these instructions on GitBash but it **should** work for MSYS2 as well\

1.1 Install  [MSYS2](https://www.msys2.org/#installation)\
1.2 Verify that g++ and cmake are installed\
```bash
g++ --version
cmake --version
```
If it isn't installed, then install these tools\
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-make
```
1.3 Create build directory\
```bash
mkdir build
cd build
```
1.4 Generate make files with cmake and compile\
    - To build for debugging, run build command without the Release flag
```bash

cmake -G "MinGW Makefiles" ..
cmake --build .
cmake --build . --config Release
```

### Compiling in Windows using MSVC

1.1 Install Visual Studio 2022 with the C++ toolkit <https://visualstudio.microsoft.com>\
1.2 Open Developer Command Prompt for Visual Studio 2022\
1.3 Switch to project root directory\
1.4 To clean the build directory, use one of these methods:
```batch
# Method 1: Delete and recreate build directory
rmdir /s /q build
mkdir build

# Method 2: Remove contents but keep directory
cd build
del /f /s /q *
cd ..
```
1.5 Run the following commands to build:
```batch
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build .
cmake --build . --config Release
.\Debug\MyProject.exe
```