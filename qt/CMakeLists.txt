cmake_minimum_required(VERSION 3.20)
project(claude-his-search VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# macOS Qt6 via Homebrew
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/qt@6")
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

set(APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/AppIcon.icns")
set_source_files_properties(${APP_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

add_executable(claude-his-search
    main.cpp
    types.h
    chatwidget.h
    sessiondelegate.h
    parser.h parser.cpp
    engine.h engine.cpp
    mainwindow.h mainwindow.cpp
    ${APP_ICON}
)

target_link_libraries(claude-his-search PRIVATE
    Qt6::Widgets
    Qt6::Concurrent
)

# macOS bundle 设置
if(APPLE)
    set_target_properties(claude-his-search PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.claude.his-search"
        MACOSX_BUNDLE_BUNDLE_NAME "Claude History Search"
        MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    )
endif()
