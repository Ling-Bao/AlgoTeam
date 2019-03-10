INCLUDE_DIRECTORIES(
    .
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

KF_ADD_LIBRARY(kf kalman_filter.cpp)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${CPPLIB_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)