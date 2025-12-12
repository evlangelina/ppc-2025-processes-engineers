#include <gtest/gtest.h>

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"
#include "bortsova_a_transmission_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(BortsovaATransmissionGatherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BortsovaATransmissionGatherMPI, BortsovaATransmissionGatherSEQ>(PPC_SETTINGS_bortsova_a_transmission_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BortsovaATransmissionGatherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BortsovaATransmissionGatherPerfTests, kGtestValues, kPerfTestName);

}  // namespace bortsova_a_transmission_gather
