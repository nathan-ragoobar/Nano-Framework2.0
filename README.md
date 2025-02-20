### Compiling in Windows

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
.\Debug\MyProject.exe
```