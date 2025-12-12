#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"

namespace bortsova_a_transmission_gather {

BortsovaATransmissionGatherMPI::BortsovaATransmissionGatherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool BortsovaATransmissionGatherMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int root = GetInput().root;
  if (root < 0 || root >= world_size_) {
    return false;
  }

  return true;
}

bool BortsovaATransmissionGatherMPI::PreProcessingImpl() {
  send_count_ = static_cast<int>(GetInput().send_data.size());
  return true;
}

bool BortsovaATransmissionGatherMPI::RunImpl() {
  int root = GetInput().root;
  const std::vector<double> &send_data = GetInput().send_data;

  int local_count = send_count_;
  MPI_Bcast(&local_count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (local_count == 0) {
    GetOutput().recv_data.clear();
    return true;
  }

  std::size_t total_size = static_cast<std::size_t>(world_size_) * static_cast<std::size_t>(local_count);
  std::vector<double> gather_buffer(total_size, 0.0);
  std::vector<bool> received(static_cast<std::size_t>(world_size_), false);

  std::copy(send_data.begin(), send_data.end(),
            gather_buffer.begin() + (static_cast<std::ptrdiff_t>(world_rank_) * local_count));
  received[static_cast<std::size_t>(world_rank_)] = true;

  int step = 1;
  while (step < world_size_) {
    if ((world_rank_ % (2 * step)) == 0) {
      int source = world_rank_ + step;
      if (source < world_size_) {
        std::vector<double> recv_buffer(total_size, 0.0);
        std::vector<bool> recv_flags(static_cast<std::size_t>(world_size_), false);

        MPI_Status status;
        MPI_Recv(recv_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);

        std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);
        MPI_Recv(flags_int.data(), world_size_, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

        for (int r = 0; r < world_size_; ++r) {
          if (flags_int[static_cast<std::size_t>(r)] != 0) {
            std::size_t start_idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(local_count);
            for (int j = 0; j < local_count; ++j) {
              gather_buffer[start_idx + static_cast<std::size_t>(j)] =
                  recv_buffer[start_idx + static_cast<std::size_t>(j)];
            }
            received[static_cast<std::size_t>(r)] = true;
          }
        }
      }
    } else if ((world_rank_ % step) == 0) {
      int dest = world_rank_ - step;
      MPI_Send(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

      std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);
      for (int r = 0; r < world_size_; ++r) {
        flags_int[static_cast<std::size_t>(r)] = received[static_cast<std::size_t>(r)] ? 1 : 0;
      }
      MPI_Send(flags_int.data(), world_size_, MPI_INT, dest, 1, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }

  if (world_rank_ == 0 && root != 0) {
    MPI_Send(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, root, 2, MPI_COMM_WORLD);
  } else if (world_rank_ == root && root != 0) {
    MPI_Status status;
    MPI_Recv(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
  }

  MPI_Bcast(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, root, MPI_COMM_WORLD);

  GetOutput().recv_data = std::move(gather_buffer);

  return true;
}

bool BortsovaATransmissionGatherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace bortsova_a_transmission_gather
