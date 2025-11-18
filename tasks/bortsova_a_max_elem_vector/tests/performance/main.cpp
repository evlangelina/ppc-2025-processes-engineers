#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "bortsova_a_max_elem_vector/common/include/common.hpp"
#include "bortsova_a_max_elem_vector/mpi/include/ops_mpi.hpp"
#include "bortsova_a_max_elem_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace bortsova_a_max_elem_vector {

class BortsovaAMaxElemVectorPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kCount_ = 250000000;
  InType input_data_{};
  int expected_max_ = 0;

  void SetUp() override {
    std::vector<int> vec(kCount_);
    std::mt19937 gen(12345);  // NOLINT(cert-msc51-cpp)
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    for (size_t i = 0; i < kCount_; ++i) {
      vec[i] = dist(gen);
    }

    expected_max_ = 2000000;
    vec[kCount_ / 2] = expected_max_;

    input_data_.data = std::move(vec);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return output_data == expected_max_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BortsovaAMaxElemVectorPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BortsovaAMaxElemVectorMpi, BortsovaAMaxElemVectorSeq>(
        PPC_SETTINGS_bortsova_a_max_elem_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BortsovaAMaxElemVectorPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BortsovaAMaxElemVectorPerfTests, kGtestValues, kPerfTestName);

}  // namespace bortsova_a_max_elem_vector
