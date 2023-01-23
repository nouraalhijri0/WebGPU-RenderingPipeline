cmake_minimum_required(VERSION 2.8)

project(framework_cs248fall2021)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-nonportable-include-path -fno-exceptions -fno-rtti")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -D_DEBUG=1 -Wno-unused -O0")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0 -DNDEBUG=1 -flto -O3")

file(GLOB sources src/*.cpp)
file(GLOB_RECURSE headers src/*.h)

if (EMSCRIPTEN)
	message("Using emscripten!")
	
	file(GLOB_RECURSE platform_sources src/ems/*.cpp)
	set(CMAKE_EXECUTABLE_SUFFIX ".html")
	
	# TODO: all the debug flags are link only
	# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -s DEMANGLE_SUPPORT=1 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s DISABLE_EXCEPTION_CATCHING=1 --closure 1")

	set(CMAKE_EXE_LINKER_FLAGS "-s ASSERTIONS=0 -s EVAL_CTORS=1 -s WASM=1 -s USE_WEBGPU=1 -sUSE_GLFW=3 -s NO_EXIT_RUNTIME=1  -lglfw -lGL -sAUTO_JS_LIBRARIES --shell-file ${CMAKE_CURRENT_LIST_DIR}/src/ems/shell.html")

	# Linker flags to optimize for smallest output code size
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web -s TEXTDECODER=2 -s ABORTING_MALLOC=0 -s ALLOW_MEMORY_GROWTH=0 -s SUPPORT_ERRNO=0 -s MALLOC=emmalloc -s NO_FILESYSTEM=1 --output_eol=linux")
elseif (WIN32)
	message("Not using emscripten!, WIN")
	file(GLOB_RECURSE platform_sources src/win/*)
	link_directories(${CMAKE_CURRENT_LIST_DIR}/lib/dawn/bin/win/x64)
	link_directories(${CMAKE_CURRENT_LIST_DIR}/lib/GLFW/win)
		
elseif (APPLE)
	message("Not using emscripten!, APPLE")
	set (CMAKE_CXX_STANDARD 14)
	file(GLOB_RECURSE platform_sources src/mac/*)
	link_directories(${CMAKE_CURRENT_LIST_DIR}/lib/dawn/bin/mac)
	add_library(dawn_native SHARED IMPORTED)
	add_library(dawn_proc SHARED IMPORTED)
	add_library(dawn_platform SHARED IMPORTED)
	set_target_properties(dawn_native PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/dawn/bin/mac/Debug/libdawn_native.dylib)
	set_target_properties(dawn_proc PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/dawn/bin/mac/Debug/libdawn_proc.dylib)
	set_target_properties(dawn_platform PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/dawn/bin/mac/Debug/libdawn_platform.dylib)
	find_library(COCOA_LIBRARY Cocoa)
	find_library(QUARTZCORE_LIBRARY QuartzCore)
	add_library(glfw3 SHARED IMPORTED)
	set_target_properties(glfw3 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/GLFW/mac/lib-x86_64/libglfw.3.dylib)
endif()

add_executable(framework_cs248fall2021 ${sources} ${platform_sources} ${headers})

if (EMSCRIPTEN)
	target_include_directories(framework_cs248fall2021  PRIVATE 
										${CMAKE_CURRENT_LIST_DIR}/inc)
elseif (WIN32)
	target_include_directories(framework_cs248fall2021  PRIVATE 
										${CMAKE_CURRENT_LIST_DIR}/inc
										${CMAKE_CURRENT_LIST_DIR}/lib/dawn/inc)
	target_link_libraries(framework_cs248fall2021  glfw3)
								
elseif (APPLE)
	target_include_directories(framework_cs248fall2021  PRIVATE 
										${CMAKE_CURRENT_LIST_DIR}/inc
										${CMAKE_CURRENT_LIST_DIR}/lib/dawn/inc)
	target_link_libraries(framework_cs248fall2021 dawn_native dawn_proc dawn_platform ${COCOA_LIBRARY} ${QUARTZCORE_LIBRARY} glfw3)
endif()
