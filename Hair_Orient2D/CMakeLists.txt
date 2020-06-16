cmake_minimum_required(VERSION 3.8)
project(Orient2D)

set(CMAKE_CXX_STANDARD 11)

#file(GLOB *.h *.cpp)

set(SOURCE_FILES main.cpp Color.h Im.hpp lineqn.h MathUtils.hpp Orient2D.hpp OrientMap.hpp
        util.h Vec.hpp XForm.h Renderer.h SimpleFragmentShader.fragmentshader SimpleVertexShader.vertexshader
        )


find_package( OpenCV REQUIRED )

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE(JPEG REQUIRED)
FIND_PACKAGE(PNG  REQUIRED)
#FIND_PACKAGE(FFTW REQUIRED)


INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR}
                    ${PNG_INCLUDE_DIR}
        ${GLFW3_INCLUDE_PATH}
        ${GLEW_INCLUDE_PATH}
        ${GLM_INCLUDE_PATH}
        /home/yi/OpenMesh-6.3/src
        #${FFTW3F_INCLUDE_DIR}
        )

add_executable(Orient2D ${SOURCE_FILES} )

target_link_libraries(Orient2D
        ${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
        ${GLFW3_LIBRARY}
        ${GLEW_LIBRARY}
        /home/yi/OpenMesh-6.3/build/Build/lib/libOpenMeshCore.so
        /home/yi/OpenMesh-6.3/build/Build/lib/libOpenMeshTools.so

        ${OpenCV_LIBS}
        ${JPEG_LIBRARIES}
        ${PNG_LIBRARIES}
        #${FFTW3F_LIBRARIES}
        "/usr/lib/x86_64-linux-gnu/libfftw3.a"
        )
