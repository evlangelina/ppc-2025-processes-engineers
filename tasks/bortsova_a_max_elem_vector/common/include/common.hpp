#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace bortsova_a_max_elem_vector {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace bortsova_a_max_elem_vector
