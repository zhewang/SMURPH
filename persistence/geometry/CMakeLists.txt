include_directories(${persistence_SOURCE_DIR}/geometry)
file(GLOB geometry_sources ${persistence_SOURCE_DIR}/geometry/*.cpp ${persistence_SOURCE_DIR}/geometry/*.h)

add_library(geometry SHARED ${geometry_sources})
