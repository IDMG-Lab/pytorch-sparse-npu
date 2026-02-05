cmake_minimum_required(VERSION 3.16.0)
project(opp)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

# --------------------------------------------------------------------------------------
# Discover and register operator directories
# --------------------------------------------------------------------------------------
# Each operator lives in <repo_root>/<op_name>/{op_host,op_kernel,...}.
# We add each operator subdirectory so its CMakeLists.txt can call register_operator().

# Optional: build only selected operators.
# Usage: cmake ... -DASCEND_OP_NAME=gathercsr
#        cmake ... -DASCEND_OP_NAME=gathercsr,ind2ptr
set(_selected_ops "")
if(NOT "${ASCEND_OP_NAME}" STREQUAL "")
  string(REPLACE "," ";" _selected_ops "${ASCEND_OP_NAME}")
endif()

file(GLOB _top_items LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_d ${_top_items})
  if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_d})
    continue()
  endif()
  # Skip non-operator infrastructure directories
  if(_d STREQUAL "cmake" OR _d STREQUAL "scripts" OR _d STREQUAL "build_out" OR _d STREQUAL "host" OR _d STREQUAL "kernel" OR _d STREQUAL "framework" OR _d STREQUAL "testcases")
    continue()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_d}/CMakeLists.txt)
    if(_selected_ops AND (NOT _d IN_LIST _selected_ops))
      continue()
    endif()
    add_subdirectory(${_d})
  endif()
endforeach()

if(ENABLE_CROSS_COMPILE)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(CROSS_COMPILE_PLATFORM aarch64)
    else()
        set(CROSS_COMPILE_PLATFORM x86_64)
    endif()
    set(PLATFORM ${CMAKE_SYSTEM_PROCESSOR})
    set(CMAKE_COMPILE_COMPILER_LIBRARY ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/devlib/linux/${CROSS_COMPILE_PLATFORM}/)
    set(CMAKE_COMPILE_RUNTIME_LIBRARY ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/devlib/${CROSS_COMPILE_PLATFORM}/)
    if(CMAKE_CROSS_LIBRARY_PATH)
        set(CMAKE_COMPILE_COMPILER_LIBRARY ${CMAKE_CROSS_LIBRARY_PATH})
        set(CMAKE_COMPILE_RUNTIME_LIBRARY ${CMAKE_CROSS_LIBRARY_PATH})
    endif()
    set(CMAKE_SYSTEM_PROCESSOR ${CROSS_COMPILE_PLATFORM})
    set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_CROSS_PLATFORM_COMPILER})
else()
    set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/host)
    add_subdirectory(host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernel)
    add_subdirectory(kernel)
endif()
if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()

# modify vendor_name in install.sh and upgrade.sh
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/scripts
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/scripts/* ${CMAKE_BINARY_DIR}/scripts/
    COMMAND sed -i "s/vendor_name=customize/vendor_name=${vendor_name}/g" ${CMAKE_BINARY_DIR}/scripts/*
)
add_custom_target(modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh)

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(gen_version_info ALL
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${CMAKE_CURRENT_BINARY_DIR}
)

if(NOT ASCEND_PACK_SHARED_LIBRARY)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/scripts/ DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

    install(FILES ${CMAKE_SOURCE_DIR}/custom.proto DESTINATION packages OPTIONAL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.info
            DESTINATION packages/vendors/${vendor_name}/)

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
    set(CPACK_PACKAGE_FILE_NAME "custom_opp_${SYSTEM_INFO}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()
