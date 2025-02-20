###Compiling in Windows

1.1 Install Visual Studio 2022 with the C++ toolkit <https://visualstudio.microsoft.com>\
1.2 Open Developer Command Prompt for Visual Studio 2022\
1.3 Switch to project root directory\
1.4 Run the following commands to build

```bash
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build .
.\Debug\MyProject.exe
```