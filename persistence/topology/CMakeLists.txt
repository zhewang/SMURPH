include_directories(${persistence_SOURCE_DIR}/topology)
file(GLOB topology_sources ${persistence_SOURCE_DIR}/topology/*.cpp ${persistence_SOURCE_DIR}/topology/munkres/*.cpp ${persistence_SOURCE_DIR}/topology/*.h ${persistence_SOURCE_DIR}/topology/munkres/*.h)

set(CMAKE_CXX_FLAGS "-std=c++11")
add_library(topology SHARED ${topology_sources})
target_link_libraries(topology geometry )
add_dependencies(topology geometry)
