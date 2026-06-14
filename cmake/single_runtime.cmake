# single_runtime.cmake — CMake toolchain snippet for single-OpenMP / single-BLAS builds
#
# Include this in your HPC-tool build (LAMMPS, OpenMM, RASPA) when linking against libtorch
# from a pip wheel. It forces the build to use torch's bundled OpenMP and BLAS, avoiding the
# "cannot generate a safe runtime search path" error and runtime segfaults from dual runtimes.
#
# Usage in your CMakeLists.txt:
#   include(${CMAKE_CURRENT_LIST_DIR}/single_runtime.cmake)
#   setup_single_runtime()   # call before find_package(Torch)
#
# Or from the command line:
#   cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/single_runtime.cmake ...

# Find torch and extract its lib directory
macro(setup_single_runtime)
    # 1. Locate torch's bundled libraries
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_cmake_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT _torch_cmake_prefix)
        # Fallback: try python3
        execute_process(
            COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _torch_cmake_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(_torch_cmake_prefix)
        list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
        get_filename_component(_torch_lib_dir "${_torch_cmake_prefix}/../lib" ABSOLUTE)
        message(STATUS "[single_runtime] Torch lib dir: ${_torch_lib_dir}")

        # 2. Force OpenMP to use torch's bundled libgomp (not the compiler's)
        # Torch pip wheels bundle their own libgomp.so.1 — we must use it
        find_library(_torch_gomp gomp PATHS "${_torch_lib_dir}" NO_DEFAULT_PATH)
        if(_torch_gomp)
            set(OpenMP_gomp_LIBRARY "${_torch_gomp}" CACHE FILEPATH "Torch's bundled libgomp" FORCE)
            set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
            message(STATUS "[single_runtime] OpenMP forced to torch's libgomp: ${_torch_gomp}")
        else()
            message(WARNING "[single_runtime] Torch's libgomp not found in ${_torch_lib_dir}")
        endif()

        # 3. Handle MKL_INCLUDE_DIR (pip torch cu12x wheels often have broken MKL headers)
        # If torch ships libmkl_* but no mkl_include/, we need a shim
        find_library(_torch_mkl mkl_rt PATHS "${_torch_lib_dir}" NO_DEFAULT_PATH)
        if(_torch_mkl)
            # Check if MKL headers exist
            find_path(_mkl_inc mkl.h PATHS "${_torch_lib_dir}/../include" NO_DEFAULT_PATH)
            if(NOT _mkl_inc)
                # Pip torch wheels don't ship MKL headers — point to a stub or skip MKL
                message(STATUS "[single_runtime] Torch links MKL but headers missing (pip wheel)")
                # Option: use conda MKL headers if available, or fall back to OpenBLAS
                find_path(_conda_mkl_inc mkl.h PATHS "$ENV{CONDA_PREFIX}/include" NO_DEFAULT_PATH)
                if(_conda_mkl_inc)
                    set(MKL_INCLUDE_DIR "${_conda_mkl_inc}" CACHE PATH "MKL headers from conda" FORCE)
                    message(STATUS "[single_runtime] Using conda MKL headers: ${_conda_mkl_inc}")
                else()
                    message(WARNING "[single_runtime] No MKL headers found — may need OpenBLAS fallback")
                endif()
            endif()
        endif()

        # 4. Ensure torch's libraries are found first at link time
        link_directories(BEFORE "${_torch_lib_dir}")

        # 5. Set rpath to include torch lib (runtime resolution)
        set(CMAKE_INSTALL_RPATH "${_torch_lib_dir};${CMAKE_INSTALL_RPATH}")
        set(CMAKE_BUILD_RPATH "${_torch_lib_dir};${CMAKE_BUILD_RPATH}")
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    else()
        message(WARNING "[single_runtime] Could not locate torch — is it installed?")
    endif()
endmacro()


# Diagnostic: check for duplicate runtimes in a built binary
function(check_single_runtime TARGET)
    if(NOT TARGET ${TARGET})
        message(WARNING "[single_runtime] Target ${TARGET} not found")
        return()
    endif()

    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Checking ${TARGET} for duplicate runtimes..."
        COMMAND sh -c "ldd $<TARGET_FILE:${TARGET}> | grep -E 'libgomp|libomp|libiomp|libopenblas|libmkl|libcblas' | sort | uniq -c | awk '{if($$1>1) print \"CONFLICT:\", $$0}'"
        COMMENT "Scanning for duplicate OpenMP/BLAS runtimes"
        VERBATIM
    )
endfunction()
