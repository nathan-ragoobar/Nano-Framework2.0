cmake_minimum_required(VERSION 3.16)

project(Dashboard VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to Qt installation only on Windows
if(WIN32)
    set(CMAKE_PREFIX_PATH "C:/Qt/6.8.0/mingw_64")
endif()

#[[
# Point to Qt installation only on Linux
if(UNIX AND NOT APPLE)
    set(CMAKE_PREFIX_PATH "/usr/lib/qt6")  # or your custom Qt installation path
endif()
]]

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

   
# Platform-specific deployment
if(WIN32)
    # Check if we're using MSYS2
    if(MINGW)
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
            COMMENT "Deploying Qt dependencies for Windows/MSYS2..."
        )
    endif()
elseif(UNIX AND NOT APPLE)
    # Ubuntu/Linux deployment
    find_program(DEPLOY_QT_TOOL qt6-deploy)
    if(NOT DEPLOY_QT_TOOL)
        message(WARNING "qt6-deploy not found. Install with: sudo apt install qt6-tools-dev")
    else()
        add_custom_command(TARGET Dashboard POST_BUILD
            # Copy Qt plugins and libraries
            COMMAND ${DEPLOY_QT_TOOL} 
                $<TARGET_FILE:Dashboard>
                --appimage
            # Set RPATH
            COMMAND patchelf 
                --set-rpath '$ORIGIN/lib' 
                $<TARGET_FILE:Dashboard>
            COMMENT "Deploying Qt dependencies for Linux..."
        )
    endif()

    # Install dependencies on Ubuntu if needed
    message(STATUS "Make sure required packages are installed:")
    message(STATUS "sudo apt install qt6-tools-dev qt6-base-dev libqt6charts6-dev patchelf")
endif()
