# Macro used to define that all following targets should be treated as
# 3rd-party targets.
macro(KF_SET_THIRD_PARTY_FOLDER)
    set(KF_THIRD_PARTY_FOLDER TRUE)
endmacro(KF_SET_THIRD_PARTY_FOLDER)

# Macro used to define that all following targets should not be treated as
# 3rd-party targets.
macro(KF_UNSET_THIRD_PARTY_FOLDER)
    set(KF_THIRD_PARTY_FOLDER FALSE)
endmacro(KF_UNSET_THIRD_PARTY_FOLDER)

# Replacement for the normal add_library() command. The syntax remains the same
# in that the first argument is the target name, and the following arguments
# are the source files to use when building the target.
macro(KF_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${KF_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/kf/)
endmacro(KF_ADD_LIBRARY)

# Replacement for the normal add_executable() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(KF_ADD_EXECUTABLE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${KF_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/kf/)
endmacro(KF_ADD_EXECUTABLE)