cmake_minimum_required(VERSION 3.16)

project(Dashboard VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to Qt installation
set(CMAKE_PREFIX_PATH "C:/Qt/6.8.0/mingw_64")
#Static linking
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")


find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Charts)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Charts)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        TrainingVisualizer.cpp
        TrainingVisualizer.hpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Dashboard
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET Dashboard APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(Dashboard SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(Dashboard
            ${PROJECT_SOURCES}
        )
    endif()
endif()

find_package(Qt6 REQUIRED COMPONENTS 
    Core
    Gui
    Widgets
    Charts
)

# Link against Qt Widgets and Charts
target_link_libraries(Dashboard PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Charts
)


# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.Dashboard)
endif()
set_target_properties(Dashboard PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS Dashboard
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Dashboard)
endif()

#[[
# Copy the Dashboard.exe and any needed Qt DLLs to the build folder
add_custom_command(TARGET Dashboard
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Dashboard>
        $<TARGET_FILE_DIR:Dashboard>/Dashboard.exe

    # Example for copying QtCore, QtGui, QtWidgets, QtCharts dlls (adjust if needed)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Core>
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Gui>
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Widgets>
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Charts>
        $<TARGET_FILE_DIR:Dashboard>

 #This is statically linked at the top so remove this
    # Also copy the GCC runtime DLL (adjust path if needed)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libgcc_s_seh-1.dll"
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libwinpthread-1.dll"
        $<TARGET_FILE_DIR:Dashboard>


    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:Dashboard>/platforms
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Qt/6.8.0/mingw_64/plugins/platforms/qwindows.dll"
        $<TARGET_FILE_DIR:Dashboard>/platforms/

    # Add style plugins
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:Dashboard>/styles
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Qt/6.8.0/mingw_64/plugins/styles/qwindowsvistastyle.dll"
        $<TARGET_FILE_DIR:Dashboard>/styles/

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libstdc++-6.dll"
        $<TARGET_FILE_DIR:Dashboard>
            
)
   ]] 

   
# Remove all previous copy commands and replace with just windeployqt
add_custom_command(TARGET Dashboard POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libgcc_s_seh-1.dll"
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libwinpthread-1.dll"
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/msys64/mingw64/bin/libstdc++-6.dll"
        $<TARGET_FILE_DIR:Dashboard>
    COMMAND C:/Qt/6.8.0/mingw_64/bin/windeployqt.exe 
        --verbose 1
        --no-compiler-runtime
        --no-translations
        --no-opengl-sw
        $<TARGET_FILE:Dashboard>
    COMMENT "Deploying Qt dependencies..."
)
