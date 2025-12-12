#pragma once

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BortsovaATransmissionGatherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int world_rank_ = 0;
  int world_size_ = 1;
  int send_count_ = 0;
};

}  // namespace bortsova_a_transmission_gather
