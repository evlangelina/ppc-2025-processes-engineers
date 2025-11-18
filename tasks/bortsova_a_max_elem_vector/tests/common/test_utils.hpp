#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bortsova_a_max_elem_vector/common/include/common.hpp"

namespace bortsova_a_max_elem_vector::test_utils {

struct GeneratedData {
  std::vector<ValueType> values;
  ValueType expected = ValueType{};
};

inline std::size_t ClampIndex(std::size_t idx, std::size_t size) {
  if (size == 0) {
    return 0;
  }
  return std::min(idx, size - 1);
}

inline GeneratedData BuildDataset(const TestType &test_case) {
  GeneratedData result{};
  switch (test_case.pattern) {
    case VectorPattern::kSmallMixed: {
      if (test_case.size != 4) {
        throw std::invalid_argument("kSmallMixed expects exactly 4 elements");
      }
      result.values = {-120, 512, -3, 64};
      result.expected = 512;
      break;
    }
    case VectorPattern::kNegativeOnly: {
      if (test_case.size != 6) {
        throw std::invalid_argument("kNegativeOnly expects exactly 6 elements");
      }
      result.values = {-8, -32, -7, -15, -1, -3};
      result.expected = -1;
      break;
    }
    case VectorPattern::kIntLimits: {
      if (test_case.size < 3) {
        throw std::invalid_argument("kIntLimits requires at least 3 elements");
      }
      result.values.assign(test_case.size, static_cast<ValueType>(-50));
      result.values.front() = std::numeric_limits<ValueType>::min();
      const std::size_t middle = test_case.size / 2;
      result.values[middle] = std::numeric_limits<ValueType>::max();
      const std::size_t third = test_case.size / 3;
      result.values[third] = static_cast<ValueType>(1024);
      result.values.back() = static_cast<ValueType>(-3);
      result.expected = std::numeric_limits<ValueType>::max();
      break;
    }
    case VectorPattern::kPlateauWithPeaks: {
      if (test_case.size < 10) {
        throw std::invalid_argument("kPlateauWithPeaks requires at least 10 elements");
      }
      result.values.assign(test_case.size, static_cast<ValueType>(77));
      const std::size_t plateau_start = test_case.size / 2;
      const std::size_t plateau_length = std::min<std::size_t>(static_cast<std::size_t>(321),
                                                               test_case.size - ClampIndex(plateau_start, test_case.size));
      for (std::size_t i = 0; i < plateau_length && plateau_start + i < result.values.size(); ++i) {
        result.values[plateau_start + i] = static_cast<ValueType>(580);
      }
      const std::size_t peak_index = ClampIndex(test_case.size - 1, test_case.size);
      result.values[peak_index] = static_cast<ValueType>(1'000'001);
      result.expected = result.values[peak_index];
      break;
    }
    case VectorPattern::kHugeArray: {
      if (test_case.size < 2) {
        throw std::invalid_argument("kHugeArray requires at least 2 elements");
      }
      result.values.assign(test_case.size, static_cast<ValueType>(-17));
      ValueType best = result.values.front();
      const std::size_t stride = 131071;
      for (std::size_t idx = stride; idx < test_case.size; idx += stride) {
        const ValueType candidate = static_cast<ValueType>((idx % 2003) - 1001);
        result.values[idx] = candidate;
        best = std::max(best, candidate);
      }
      const std::size_t pivot = ClampIndex(test_case.size - 97, test_case.size);
      result.values[pivot] = static_cast<ValueType>(987'654'321);
      best = std::max(best, result.values[pivot]);
      result.expected = best;
      break;
    }
    default:
      throw std::invalid_argument("Unknown vector pattern");
  }
  if (result.values.size() != test_case.size) {
    throw std::runtime_error("Generated dataset size mismatch");
  }
  return result;
}

}  // namespace bortsova_a_max_elem_vector::test_utils

