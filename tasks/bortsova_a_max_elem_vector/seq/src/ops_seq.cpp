#include "bortsova_a_max_elem_vector/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>

#include "bortsova_a_max_elem_vector/common/include/common.hpp"

namespace bortsova_a_max_elem_vector {

bortsova_a_max_elem_vectorMPISEQ::bortsova_a_max_elem_vectorMPISEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::min();
}

bool bortsova_a_max_elem_vectorMPISEQ::ValidationImpl() {
  return !GetInput().data.empty();
}

bool bortsova_a_max_elem_vectorMPISEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool bortsova_a_max_elem_vectorMPISEQ::RunImpl() {
  const auto &vec = GetInput().data;
  
  if (vec.empty()) {
    return false;
  }

  int max_elem = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (vec[i] > max_elem) {
      max_elem = vec[i];
    }
  }

  GetOutput() = max_elem;
  return true;
}

bool bortsova_a_max_elem_vectorMPISEQ::PostProcessingImpl() {
  // If RunImpl succeeded, the output is valid (even if it's INT_MIN)
  return true;
}

}  // namespace bortsova_a_max_elem_vector
