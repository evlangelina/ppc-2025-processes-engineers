#include <gtest/gtest.h>

#include "bortsova_a_max_elem_vector/common/include/common.hpp"
#include "bortsova_a_max_elem_vector/mpi/include/ops_mpi.hpp"
#include "bortsova_a_max_elem_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace bortsova_a_max_elem_vector {

class BortsovaAMaxElemVectorPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BortsovaAMaxElemVectorPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, bortsova_a_max_elem_vectorMPI, bortsova_a_max_elem_vectorMPISEQ>(PPC_SETTINGS_bortsova_a_max_elem_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BortsovaAMaxElemVectorPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BortsovaAMaxElemVectorPerfTests, kGtestValues, kPerfTestName);

}  // namespace bortsova_a_max_elem_vector
