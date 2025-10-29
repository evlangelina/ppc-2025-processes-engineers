#include "bortsova_a_max_elem_vector/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "bortsova_a_max_elem_vector/common/include/common.hpp"
#include "util/include/util.hpp"

namespace bortsova_a_max_elem_vector {

bortsova_a_max_elem_vectorMPISEQ::bortsova_a_max_elem_vectorMPISEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool bortsova_a_max_elem_vectorMPISEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool bortsova_a_max_elem_vectorMPISEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool bortsova_a_max_elem_vectorMPISEQ::RunImpl() {
  if (GetInput() == 0) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int counter = 0;
  for (int i = 0; i < num_threads; i++) {
    counter++;
  }

  if (counter != 0) {
    GetOutput() /= counter;
  }
  return GetOutput() > 0;
}

bool bortsova_a_max_elem_vectorMPISEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace bortsova_a_max_elem_vector
