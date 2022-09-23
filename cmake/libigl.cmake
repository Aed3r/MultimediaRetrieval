if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(libigl)

if (CMAKE_COMPILER_IS_MINGW)
  add_definitions(-DWC_NO_BEST_FIT_CHARS=0x400)   # Note - removed -DPOCO_WIN32_UTF8
  add_definitions(-D_WIN32 -DMINGW32 -DWINVER=0x500 -DODBCVER=0x0300 -DPOCO_THREAD_STACK_SIZE)
  add_compile_options(-Wa,-mbig-obj)   # Note: new - fixes "file too big"
endif (CMAKE_COMPILER_IS_MINGW)