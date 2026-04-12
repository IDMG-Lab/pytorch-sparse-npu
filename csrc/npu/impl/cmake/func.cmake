include(ExternalProject)

function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

# -------------------------------------------------------------------------------------------------
# Operator registry helpers
# -------------------------------------------------------------------------------------------------
# Each operator directory (e.g. <repo_root>/gathercsr) contains its own op_host/ and op_kernel/.
# We register operator directories explicitly so:
#   1) host sources collection is deterministic (no broad globbing over the whole repo);
#   2) per-operator CMakeLists.txt becomes meaningful (can add per-op customizations later);
#   3) single-op builds can be implemented by only adding selected op subdirectories.

function(register_operator OP_NAME)
  # Record the operator name and its absolute directory.
  get_filename_component(_op_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

  get_property(_ops GLOBAL PROPERTY OPS_REGISTERED_NAMES)
  if(NOT _ops)
    set(_ops "")
  endif()
  if("${OP_NAME}" IN_LIST _ops)
    return()
  endif()

  set_property(GLOBAL APPEND PROPERTY OPS_REGISTERED_NAMES "${OP_NAME}")
  set_property(GLOBAL APPEND PROPERTY OPS_REGISTERED_DIRS  "${_op_dir}")
endfunction()

function(get_registered_operators OUT_NAMES OUT_DIRS)
  get_property(_names GLOBAL PROPERTY OPS_REGISTERED_NAMES)
  get_property(_dirs  GLOBAL PROPERTY OPS_REGISTERED_DIRS)
  if(NOT _names)
    set(_names "")
  endif()
  if(NOT _dirs)
    set(_dirs "")
  endif()
  set(${OUT_NAMES} "${_names}" PARENT_SCOPE)
  set(${OUT_DIRS}  "${_dirs}"  PARENT_SCOPE)
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_SOURCE_DIR}
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  set(ENV{ENABLE_SOURCE_PACAKGE} ${OPBUILD_ENABLE_SOURCE})
  if(${ASCEND_PACK_SHARED_LIBRARY})
    if (NOT vendor_name)
      message(FATAL_ERROR "ERROR: vendor_name is invalid!")
      return()
    endif()
    set(ENV{ASCEND_VENDOR_NAME} ${vendor_name})
    set(ENV{OPS_PRODUCT_NAME} ${ASCEND_COMPUTE_UNIT})
    set(ENV{SYSTEM_PROCESSOR} ${CMAKE_SYSTEM_PROCESSOR})
  endif()
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  unset(ENV{ENABLE_SOURCE_PACAKGE})
  if(${ASCEND_PACK_SHARED_LIBRARY})
    unset(ENV{ASCEND_VENDOR_NAME})
    unset(ENV{OPS_PRODUCT_NAME})
    unset(ENV{SYSTEM_PROCESSOR})
  endif()
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  if(NOT ${ASCEND_PACK_SHARED_LIBRARY})
    install(FILES ${OPINFO_OUTPUT}
            DESTINATION ${OPINFO_INSTALL_DIR}
    )
  endif()
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_gen_options.py
                          ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT}
                          ${OP_COMPILE_OPTIONS}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR)
  if (${EXEC_RESULT})
      message("add ops compile options info: ${EXEC_INFO}")
      message("add ops compile options error: ${EXEC_ERROR}")
      message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR}
            ${NPUSUP_OUT_DIR}
  )
  add_custom_target(npu_supported_ops ALL
    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  )
  if(NOT ${ASCEND_PACK_SHARED_LIBRARY})
    install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
      DESTINATION ${NPUSUP_INSTALL_DIR}
    )
  endif()
endfunction()

function(add_simple_kernel_compile)
  set(options "")
  set(single_value_args "OPS_INFO;OUT_DIR;TILING_LIB;OP_TYPE;SRC;COMPUTE_UNIT;JSON_FILE;DYNAMIC_PATH")
  set(multi_value_args "OPTIONS;CONFIGS")
  cmake_parse_arguments(BINCMP "${options}" "${single_value_args}" "${multi_value_args}" ${ARGN})
  if (NOT DEFINED BINCMP_OUT_DIR)
    set(BINCMP_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/binary)
  endif()
  if (NOT DEFINED BINCMP_TILING_LIB)
    set(BINCMP_TILING_LIB $<TARGET_FILE:cust_optiling>)
  endif()
  if (${ASCEND_PACK_SHARED_LIBRARY})
    if (NOT TARGET op_kernel_pack)
      add_custom_target(op_kernel_pack
                        COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_pack_kernel.py
                        --input-path=${BINCMP_OUT_DIR}
                        --output-path=${BINCMP_OUT_DIR}/library
                        --enable-library=${ASCEND_PACK_SHARED_LIBRARY}
                        --platform=${CMAKE_SYSTEM_PROCESSOR})
      add_library(ascend_kernels INTERFACE)
      target_link_libraries(ascend_kernels INTERFACE kernels)
      target_link_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
      target_include_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
      add_dependencies(ascend_kernels op_kernel_pack)
      add_dependencies(op_kernel_pack ${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT})
    endif()
  endif()
  # add Environment Variable Configurations of ccache
  set(_ASCENDC_ENV_VAR)
  if(${CMAKE_CXX_COMPILER_LAUNCHER} MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()

  if (NOT DEFINED BINCMP_OPS_INFO)
    set(BINCMP_OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${BINCMP_COMPUTE_UNIT}-ops-info.ini)
  endif()
  if (NOT ${ENABLE_CROSS_COMPILE})
    add_custom_target(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${BINCMP_OP_TYPE}
                      --src-file=${BINCMP_SRC}
                      --compute-unit=${BINCMP_COMPUTE_UNIT}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${BINCMP_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${BINCMP_DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${BINCMP_JSON_FILE}
                      --build-tool=$(MAKE))
    add_dependencies(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT} cust_optiling)
  else()
    if (${ENABLE_BINARY_PACKAGE} AND NOT DEFINED HOST_NATIVE_TILING_LIB)
      message(FATAL_ERROR "Native host libs was not set for cross compile!")
    endif()
    add_custom_target(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${BINCMP_OP_TYPE}
                      --src-file=${BINCMP_SRC}
                      --compute-unit=${BINCMP_COMPUTE_UNIT}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${HOST_NATIVE_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${BINCMP_DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${BINCMP_JSON_FILE}
                      --build-tool=$(MAKE))
  endif()
  add_dependencies(ascendc_bin_${BINCMP_COMPUTE_UNIT}_gen_ops_config ${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT})
  add_dependencies(${BINCMP_OP_TYPE}_${BINCMP_COMPUTE_UNIT} ops_info_gen_${BINCMP_COMPUTE_UNIT})
endfunction()

function(ascendc_device_library)
    message(STATUS "Ascendc device library generating")
    cmake_parse_arguments(DEVICE "" "TARGET;OPTION" "SRC" ${ARGN})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E echo "cmake_minimum_required(VERSION 3.16.0)\nproject(cust_tiling_sink)\ninclude(${CMAKE_SOURCE_DIR}/cmake/device_task.cmake)\n"
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink/CMakeLists.txt
        RESULT_VARIABLE result
    )
    string(REPLACE ";" " " DEVICE_SRC "${DEVICE_SRC}")
    ExternalProject_Add(tiling_sink_task
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tiling_sink
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
        -DTARGET=${DEVICE_TARGET}
        -DOPTION=${DEVICE_OPTION}
        -DSRC=${DEVICE_SRC}
        -DVENDOR_NAME=${vendor_name}
        <SOURCE_DIR>
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        INSTALL_COMMAND ""
        BUILD_ALWAYS TRUE
    )
    ExternalProject_Get_Property(tiling_sink_task BINARY_DIR)
    set(TILINGSINK_LIB_PATH "")
    if ("${DEVICE_OPTION}" STREQUAL "SHARED")
        set(TILINGSINK_LIB_PATH "${BINARY_DIR}/libcust_opmaster.so")
    else()
        set(TILINGSINK_LIB_PATH "${BINARY_DIR}/libcust_opmaster.a")
    endif()
    install(FILES ${TILINGSINK_LIB_PATH}
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_master_device/lib
    )
endfunction()
function(add_opregistry_target)
  string(REPLACE ";" "-" COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")
  add_custom_target(op_registry_pack
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_pack_opregistry.py
                    --input-path=${CMAKE_SOURCE_DIR}/build_out/
                    --base-path=${CMAKE_SOURCE_DIR}/build_out/tmp/vendors/
                    --output-path=${CMAKE_SOURCE_DIR}/build_out/library/
                    --vendor-name=${vendor_name}
                    --compute-unit=${COMPUTE_UNIT}
                    --framework-type=${ASCEND_FRAMEWORK_TYPE}
                    --platform=${CMAKE_SYSTEM_PROCESSOR})
  add_library(ascend_opregistry INTERFACE)
  target_link_libraries(ascend_opregistry INTERFACE opregistry)
  target_link_directories(ascend_opregistry INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library)
  target_include_directories(ascend_opregistry INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library)
  add_dependencies(ascend_opregistry op_registry_pack)
  if(EXISTS  "${CMAKE_SOURCE_DIR}/framework/caffe_plugin")
    add_dependencies(op_registry_pack cust_caffe_parsers)
  elseif(EXISTS  "${CMAKE_SOURCE_DIR}/framework/tf_plugin")
    add_dependencies(op_registry_pack cust_tf_parsers)
  elseif(EXISTS  "${CMAKE_SOURCE_DIR}/framework/onnx_plugin")
    add_dependencies(op_registry_pack cust_onnx_parsers)
  endif()
endfunction()

function(add_kernels_install)
  # install kernel file
  if (${ENABLE_SOURCE_PACKAGE})
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic/
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic/
    )
  endif()

  # install *.o files and *.json files 
  if (${ENABLE_BINARY_PACKAGE})
    set(INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/)
      foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/
                DESTINATION ${INSTALL_DIR}/kernel/${compute_unit}/
        )
      endforeach()
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/config/
              DESTINATION ${INSTALL_DIR}/kernel/config/
      )
  endif()
endfunction()

function(add_kernels_compile)
  # --------------------------------------------------------------------------------------
  # Host build: collect sources from registered operators and run opbuild
  # --------------------------------------------------------------------------------------
  get_registered_operators(_op_names _op_dirs)
  
  set(ops_srcs "")
  foreach(_dir ${_op_dirs})
    if(EXISTS ${_dir}/op_host)
      file(GLOB _srclist ${_dir}/op_host/*.cpp)
      list(APPEND ops_srcs ${_srclist})
    endif()
  endforeach()

  if(NOT ops_srcs)
    message(FATAL_ERROR "No operator host sources found. Ensure operator directories contain op_host/*.cpp and are added via add_subdirectory in top-level CMakeLists.txt.")
  endif()

  opbuild(OPS_SRC ${ops_srcs}
          OUT_DIR ${ASCEND_AUTOGEN_PATH}
  )

  file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)

  add_library(cust_op_proto SHARED
      $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
      ${ops_srcs}
      ${ASCEND_AUTOGEN_PATH}/op_proto.cc
  )
  target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
  target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
  if(ENABLE_CROSS_COMPILE)
      target_link_directories(cust_op_proto PRIVATE ${CMAKE_COMPILE_COMPILER_LIBRARY} ${CMAKE_COMPILE_RUNTIME_LIBRARY})
  endif()
  target_link_libraries(cust_op_proto PRIVATE intf_pub exe_graph register tiling_api -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive)
  set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)

  file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
  add_library(cust_optiling SHARED ${ops_srcs})
  if (${fallback_src})
      target_sources(cust_optiling PRIVATE ${fallback_src})
  endif()
  target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
  target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
  if(ENABLE_CROSS_COMPILE)
      target_link_directories(cust_optiling PRIVATE ${CMAKE_COMPILE_COMPILER_LIBRARY} ${CMAKE_COMPILE_RUNTIME_LIBRARY})
  endif()
  target_link_libraries(cust_optiling PRIVATE nnopbase intf_pub exe_graph register tiling_api -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive)
  set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)

  file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
  file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
  if(NOT ASCEND_PACK_SHARED_LIBRARY)
      add_library(cust_opapi SHARED ${aclnn_src})
  else()
      file(GLOB op_registry ${ASCEND_AUTOGEN_PATH}/custom_op_registry.cpp)
      add_library(cust_opapi SHARED ${aclnn_src} ${op_registry})
      target_compile_definitions(cust_opapi PRIVATE ACLNN_WITH_BINARY)
  endif()
  if(ENABLE_CROSS_COMPILE)
      target_link_directories(cust_opapi PRIVATE ${CMAKE_COMPILE_COMPILER_LIBRARY} ${CMAKE_COMPILE_RUNTIME_LIBRARY})
  endif()
  if(NOT ASCEND_PACK_SHARED_LIBRARY)
      target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase)
  else()
      add_library(cust_op_proto_obj OBJECT $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}> ${ops_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
      target_compile_definitions(cust_op_proto_obj PRIVATE OP_PROTO_LIB)
      target_compile_options(cust_op_proto_obj PRIVATE -fvisibility=hidden)
      if(ENABLE_CROSS_COMPILE)
          target_link_directories(cust_op_proto_obj PRIVATE ${CMAKE_COMPILE_COMPILER_LIBRARY} ${CMAKE_COMPILE_RUNTIME_LIBRARY})
      endif()
      target_link_libraries(cust_op_proto_obj PRIVATE intf_pub exe_graph register tiling_api -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive)
      add_library(cust_optiling_obj OBJECT ${ops_srcs})
      target_compile_definitions(cust_optiling_obj PRIVATE OP_TILING_LIB)
      target_compile_options(cust_optiling_obj PRIVATE -fvisibility=hidden)
      if(ENABLE_CROSS_COMPILE)
          target_link_directories(cust_optiling_obj PRIVATE ${CMAKE_COMPILE_COMPILER_LIBRARY} ${CMAKE_COMPILE_RUNTIME_LIBRARY})
      endif()
      target_link_libraries(cust_optiling_obj PRIVATE intf_pub exe_graph register tiling_api -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive)
      target_compile_options(cust_opapi PRIVATE -DLOG_CPP)
      target_include_directories(cust_opapi INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library/)
      target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase cust_optiling_obj cust_op_proto_obj ascend_opregistry ascend_kernels)
      add_dependencies(cust_opapi ascend_opregistry)
  endif()

  add_custom_target(optiling_compat ALL COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling> ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so)

  if(NOT ASCEND_PACK_SHARED_LIBRARY)
      install(TARGETS cust_op_proto LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
      install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
      file(GLOB GROUP_PROTO_HEADERS ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
      if (GROUP_PROTO_HEADERS)
          install(FILES ${GROUP_PROTO_HEADERS} DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
      endif()
      install(TARGETS cust_optiling LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
      install(TARGETS cust_opapi LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
      install(FILES ${aclnn_inc} DESTINATION packages/vendors/${vendor_name}/op_api/include)
  else()
      file(GLOB group_inc ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
      install(TARGETS cust_opapi LIBRARY DESTINATION op_api/lib)
      install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h DESTINATION op_api/include)
      install(FILES ${group_inc} DESTINATION op_api/include)
      install(FILES ${aclnn_inc} DESTINATION op_api/include)
  endif()

  # --------------------------------------------------------------------------------------
  # Kernel build: compile kernel sources for each compute unit
  # --------------------------------------------------------------------------------------
  if (${ENABLE_SOURCE_PACKAGE})
    set(DYNAMIC_PATH ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic)
    # execute_process(COMMAND sh -c "mkdir -p ${DYNAMIC_PATH} &&
    #                               find ${CMAKE_SOURCE_DIR} -maxdepth 3 -type f \\( -path '*/op_kernel/*' \\) -print | while read f; do cp -f \"$f\" ${DYNAMIC_PATH}/; done && rm -f ${DYNAMIC_PATH}/CMakeLists.txt"
    #                 RESULT_VARIABLE EXEC_RESULT
    #                 ERROR_VARIABLE EXEC_ERROR
    # )
    execute_process(COMMAND sh -c "mkdir -p ${DYNAMIC_PATH} &&
                                  find ${CMAKE_SOURCE_DIR} -type f \\( -path '*/op_kernel/*' \\) -print | while read f; do cp -f \"$f\" ${DYNAMIC_PATH}/; done && rm -f ${DYNAMIC_PATH}/CMakeLists.txt"
                    RESULT_VARIABLE EXEC_RESULT
                    ERROR_VARIABLE EXEC_ERROR
    )# 这里不再限制对应的最大啊深度
    if (${EXEC_RESULT})
      message(FATAL_ERROR, "copy_source_files failed, gen error:${EXEC_ERROR}" )
    endif()
  endif()

  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
      OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # define a target:binary to prevent kernel file from being rebuilt during the preinstall process
    if (NOT TARGET binary)
      add_custom_target(binary)
    endif()

    if (${ENABLE_BINARY_PACKAGE} OR ${ENABLE_SOURCE_PACKAGE})
      if (${ENABLE_BINARY_PACKAGE})
        # gen binary_info_config.json and <file_name>.json
        add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config
                          COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py
                                  -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                          COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py
                                  -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                                  -s ${compute_unit}
                          COMMAND ${CMAKE_COMMAND} -E make_directory
                                  ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
                          COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/*.json
                                  ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
        )
      else()
        if (NOT TARGET ascendc_bin_${compute_unit}_gen_ops_config)
          add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config)
        endif()
      endif()
      add_dependencies(binary ascendc_bin_${compute_unit}_gen_ops_config)

      # get op_type-op_name from aic-${compute_unit}-ops-info.ini
      execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_get_op_name.py
                              --ini-file=${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
                      OUTPUT_VARIABLE OP_TYPE_NAME
                      RESULT_VARIABLE EXEC_RESULT
                      ERROR_VARIABLE EXEC_ERROR
      )
      if (${EXEC_RESULT})
        message(FATAL_ERROR, "get op name failed, gen error: ${EXEC_ERROR}")
      endif()

      # compile op one by one with ascendc_compile_kernel.py
      string(REPLACE "\n" ";" TYPE_NAME_LIST "${OP_TYPE_NAME}")
      foreach(TYPE_NAME IN LISTS TYPE_NAME_LIST)
        if (NOT "${TYPE_NAME}" STREQUAL "")
          string(REPLACE "-" ";" bin_sep ${TYPE_NAME})
          list(GET bin_sep 0 op_type)
          list(GET bin_sep 1 op_file)
          # locate kernel source: prefer legacy op_kernel/, otherwise search */op_kernel/
          set(kernel_src "${CMAKE_SOURCE_DIR}/op_kernel/${op_file}.cpp")
          if(NOT EXISTS ${kernel_src})
            file(GLOB_RECURSE _k_candidates "${CMAKE_SOURCE_DIR}/*/op_kernel/${op_file}.cpp")
            list(LENGTH _k_candidates _k_len)
            if(_k_len EQUAL 1)
              list(GET _k_candidates 0 kernel_src)
            elseif(_k_len GREATER 1)
              message(FATAL_ERROR "kernel source ${op_file}.cpp is not unique: ${_k_candidates}")
            else()
              # Also search in subdirectories like sparse/*/op_kernel, scatter/*/op_kernel
              file(GLOB_RECURSE _k_candidates "${CMAKE_SOURCE_DIR}/sparse/*/op_kernel/${op_file}.cpp")
              list(LENGTH _k_candidates _k_len)
              if(_k_len EQUAL 1)
                list(GET _k_candidates 0 kernel_src)
              else()
                file(GLOB_RECURSE _k_candidates "${CMAKE_SOURCE_DIR}/scatter/*/op_kernel/${op_file}.cpp")
                list(LENGTH _k_candidates _k_len)
                if(_k_len EQUAL 1)
                  list(GET _k_candidates 0 kernel_src)
                else()
                  message(FATAL_ERROR "kernel source not found for ${op_file}.cpp")
                endif()
              endif()
            endif()
          endif()
          add_simple_kernel_compile(OP_TYPE ${op_type}
                                    SRC ${kernel_src}
                                    COMPUTE_UNIT ${compute_unit}
                                    JSON_FILE ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
                                    DYNAMIC_PATH ${DYNAMIC_PATH})
        endif()
      endforeach()
    endif()
  endforeach()

  # generate npu_supported_ops.json
  add_npu_support_target(TARGET npu_supported_ops
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
    INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
  )

  if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
  endif()

  if(NOT ASCEND_PACK_SHARED_LIBRARY)
    add_kernels_install()
  else()
    add_opregistry_target()
  endif()
endfunction()