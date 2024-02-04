cmake_minimum_required(VERSION 3.0)
project(splicer LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

aux_source_directory(. SOURCES)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    PkgConfig::ffmpeg
    ${OpenCV_LIBS}
)
