###Compiling on Windows using MSYS2

```bash
pacman -Qs qt6
pacman -S mingw-w64-x86_64-qt6-charts
cmake ..
cmake --build .
```