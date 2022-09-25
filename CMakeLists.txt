cmake_minimum_required(VERSION 3.16)
project(main)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
#option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

# Add your project files
#file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
