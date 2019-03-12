add_library(kf SHARED kalman_filter.cpp)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${CPPLIB_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)