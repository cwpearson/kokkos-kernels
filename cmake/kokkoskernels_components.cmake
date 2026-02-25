# Define component dependencies and enable
# them selectively based on what the user
# requests.

kokkoskernels_add_option("ENABLE_ALL_COMPONENTS" ON BOOL
  "Whether to build all the library's components. Default: ON")

set(KokkosKernels_ENABLE_COMPONENT_BATCHED_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_BATCHED_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_BATCHED"
  ${KokkosKernels_ENABLE_COMPONENT_BATCHED_DEFAULT} BOOL
  "Whether to build the batched component. Default: OFF")

set(KokkosKernels_ENABLE_COMPONENT_BLAS_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_BLAS_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_BLAS"
  ${KokkosKernels_ENABLE_COMPONENT_BLAS_DEFAULT} BOOL
  "Whether to build the blas component. Default: OFF")

set(KokkosKernels_ENABLE_COMPONENT_LAPACK_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_LAPACK_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_LAPACK"
  ${KokkosKernels_ENABLE_COMPONENT_LAPACK_DEFAULT} BOOL
  "Whether to build the lapack component. Default: OFF")

set(KokkosKernels_ENABLE_COMPONENT_GRAPH_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_GRAPH_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_GRAPH"
  ${KokkosKernels_ENABLE_COMPONENT_GRAPH_DEFAULT} BOOL
  "Whether to build the graph component. Default: OFF")

set(KokkosKernels_ENABLE_COMPONENT_SPARSE_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_SPARSE_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_SPARSE"
  ${KokkosKernels_ENABLE_COMPONENT_SPARSE_DEFAULT} BOOL
  "Whether to build the sparse component. Default: OFF")

set(KokkosKernels_ENABLE_COMPONENT_ODE_DEFAULT OFF)
if(KokkosKernels_ENABLE_ALL_COMPONENTS)
  set(KokkosKernels_ENABLE_COMPONENT_ODE_DEFAULT ON)
endif()
kokkoskernels_add_option("ENABLE_COMPONENT_ODE"
  ${KokkosKernels_ENABLE_COMPONENT_ODE_DEFAULT} BOOL
  "Whether to build the ode component. Default: OFF")

# Graph depends on everything else because it depends
# on Sparse at the moment, breaking that dependency will
# allow for Graph to build pretty much as a standalone
# component.
if(KokkosKernels_ENABLE_COMPONENT_GRAPH)
  set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_BLAS    ON CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_SPARSE  ON CACHE BOOL "" FORCE)
endif()

# Sparse depends on everything else so no real benefit
# here unfortunately...
if(KokkosKernels_ENABLE_COMPONENT_SPARSE)
  set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_BLAS    ON CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_LAPACK  ON CACHE BOOL "" FORCE)
  set(KokkosKernels_ENABLE_COMPONENT_GRAPH   ON CACHE BOOL "" FORCE)
endif()

# KOKKOSKERNELS_ALL_COMPONENTS_ENABLED says whether all components are on,
# regardless of how this came to be
# this is in the cache so we can use it as a global variable,
# but marking it as advanced should hide it from GUIs
if(KokkosKernels_ENABLE_COMPONENT_BATCHED
   AND KokkosKernels_ENABLE_COMPONENT_BLAS
   AND KokkosKernels_ENABLE_COMPONENT_LAPACK
   AND KokkosKernels_ENABLE_COMPONENT_GRAPH
   AND KokkosKernels_ENABLE_COMPONENT_SPARSE
   AND KokkosKernels_ENABLE_COMPONENT_ODE)
  set(KOKKOSKERNELS_ALL_COMPONENTS_ENABLED ON CACHE BOOL "" FORCE)
else()
  set(KOKKOSKERNELS_ALL_COMPONENTS_ENABLED OFF CACHE BOOL "" FORCE)
endif()
mark_as_advanced(FORCE KOKKOSKERNELS_ALL_COMPONENTS_ENABLED)
# Now that component enables are finalized, also set upper-case
# versions of component enables for the config.h

set(KOKKOSKERNELS_ENABLE_COMPONENT_BATCHED ${KokkosKernels_ENABLE_COMPONENT_BATCHED})
set(KOKKOSKERNELS_ENABLE_COMPONENT_BLAS    ${KokkosKernels_ENABLE_COMPONENT_BLAS})
set(KOKKOSKERNELS_ENABLE_COMPONENT_LAPACK  ${KokkosKernels_ENABLE_COMPONENT_LAPACK})
set(KOKKOSKERNELS_ENABLE_COMPONENT_GRAPH   ${KokkosKernels_ENABLE_COMPONENT_GRAPH})
set(KOKKOSKERNELS_ENABLE_COMPONENT_SPARSE  ${KokkosKernels_ENABLE_COMPONENT_SPARSE})
set(KOKKOSKERNELS_ENABLE_COMPONENT_ODE     ${KokkosKernels_ENABLE_COMPONENT_ODE})
