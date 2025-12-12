#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"
#include "bortsova_a_transmission_gather/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    input_.send_data.resize(static_cast<std::size_t>(count));
    std::iota(input_.send_data.begin(), input_.send_data.end(), 1.0);
    input_.root = 0;

    std::string test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    is_mpi_test_ = test_name.find("_mpi_") != std::string::npos;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.recv_data.empty() && !input_.send_data.empty()) {
      return false;
    }

    int world_size = 1;
    if (is_mpi_test_ && ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    std::size_t expected_size = input_.send_data.size() * static_cast<std::size_t>(world_size);
    if (output_data.recv_data.size() != expected_size) {
      return false;
    }

    for (int r = 0; r < world_size; ++r) {
      std::size_t offset = static_cast<std::size_t>(r) * input_.send_data.size();
      for (std::size_t i = 0; i < input_.send_data.size(); ++i) {
        if (output_data.recv_data[offset + i] != input_.send_data[i]) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  bool is_mpi_test_ = false;
};

namespace {

TEST_P(BortsovaATransmissionGatherFuncTests, GatherTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BortsovaATransmissionGatherMPI, InType>(
                                               kTestParam, PPC_SETTINGS_bortsova_a_transmission_gather),
                                           ppc::util::AddFuncTask<BortsovaATransmissionGatherSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_bortsova_a_transmission_gather));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BortsovaATransmissionGatherFuncTests::PrintFuncTestName<BortsovaATransmissionGatherFuncTests>;

INSTANTIATE_TEST_SUITE_P(GatherTests, BortsovaATransmissionGatherFuncTests, kGtestValues, kPerfTestName);

}  // namespace

class BortsovaAGatherEdgeCasesTestsMPI : public ::testing::Test {
 protected:
  void RunMPIGatherTest(const InType &input, int expected_count_per_rank) {
    auto task = std::make_shared<BortsovaATransmissionGatherMPI>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const auto &result = task->GetOutput().recv_data;
    std::size_t expected_total =
        static_cast<std::size_t>(expected_count_per_rank) * static_cast<std::size_t>(world_size);
    ASSERT_EQ(result.size(), expected_total);

    for (int r = 0; r < world_size; ++r) {
      std::size_t offset = static_cast<std::size_t>(r) * static_cast<std::size_t>(expected_count_per_rank);
      for (std::size_t i = 0; i < input.send_data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[offset + i], input.send_data[i]);
      }
    }
  }
};

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, SingleElementGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {42.0};
  input.root = 0;
  RunMPIGatherTest(input, 1);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, LargeDataGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data.resize(1000);
  std::iota(input.send_data.begin(), input.send_data.end(), 1.0);
  input.root = 0;
  RunMPIGatherTest(input, 1000);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, NegativeValuesGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
  input.root = 0;
  RunMPIGatherTest(input, 6);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, FloatingPointPrecisionGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {0.1, 0.2, 0.3, 1e-10, 1e10};
  input.root = 0;
  RunMPIGatherTest(input, 5);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, TenElementsGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data.resize(10);
  std::iota(input.send_data.begin(), input.send_data.end(), 1.0);
  input.root = 0;
  RunMPIGatherTest(input, 10);
}

class BortsovaAGatherEdgeCasesTestsSEQ : public ::testing::Test {
 protected:
  void RunSEQGatherTest(const InType &input, const std::vector<double> &expected) {
    auto task = std::make_shared<BortsovaATransmissionGatherSEQ>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    const auto &result = task->GetOutput().recv_data;
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      EXPECT_DOUBLE_EQ(result[i], expected[i]);
    }
  }
};

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, SingleElementGather) {
  InType input;
  input.send_data = {42.0};
  input.root = 0;
  RunSEQGatherTest(input, {42.0});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, LargeDataGather) {
  InType input;
  input.send_data.resize(1000);
  std::iota(input.send_data.begin(), input.send_data.end(), 1.0);
  input.root = 0;

  std::vector<double> expected(1000);
  std::iota(expected.begin(), expected.end(), 1.0);
  RunSEQGatherTest(input, expected);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, NegativeValuesGather) {
  InType input;
  input.send_data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, FloatingPointPrecisionGather) {
  InType input;
  input.send_data = {0.1, 0.2, 0.3, 1e-10, 1e10};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, EmptyDataGather) {
  InType input;
  input.send_data = {};
  input.root = 0;
  RunSEQGatherTest(input, {});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, TwoElementsGather) {
  InType input;
  input.send_data = {1.5, 2.5};
  input.root = 0;
  RunSEQGatherTest(input, {1.5, 2.5});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, IdenticalValuesGather) {
  InType input;
  input.send_data = {7.0, 7.0, 7.0, 7.0, 7.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, AlternatingSignsGather) {
  InType input;
  input.send_data = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

}  // namespace bortsova_a_transmission_gather
