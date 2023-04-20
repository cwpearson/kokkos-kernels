#pragma once

#include <Kokkos_Core.hpp>

namespace KokkosKernels {
namespace Experimental {

struct Result {

};

template <typename ExecSpace, typename View>
KOKKOS_INLINE_FUNCTION
void pack_view(const ExecSpace &space, const View &view) {

}

template <typename ExecSpace, typename View = void>
KOKKOS_INLINE_FUNCTION
void pack_impl(const ExecSpace &space, Result &result) {

}

template <typename ExecSpace, typename View, typename... Views>
KOKKOS_INLINE_FUNCTION
void pack_impl(const ExecSpace &space, Result &result, const View &view, const Views & ...views) {
    // pack the first view
    pack_view(space, view);

    // pack the rest of the views
    pack_impl(space, result, views...);
}


template <typename ExecSpace, typename... Views>
KOKKOS_INLINE_FUNCTION
Result pack(const ExecSpace &space, const Views & ...views) {
    Result result;
    pack_impl(space, result, views...);
    return result;
}

}
}
