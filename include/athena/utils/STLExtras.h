//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_STLEXTRAS_H
#define ATHENA_STLEXTRAS_H

#include <algorithm>
#include <cstddef>

namespace athena::utils {

namespace detail {
template <typename RealIterator> class EnumeratorIterator;
template <typename RealIterator> class EnumeratorValue {
public:
  EnumeratorValue(RealIterator it, size_t idx) : mIterator(it), mIndex(idx) {}

  auto value() -> typename RealIterator::reference { return *mIterator; }
  auto index() -> size_t { return mIndex; }
private:
  friend EnumeratorIterator<RealIterator>;
  RealIterator mIterator;
  size_t mIndex;
};
template <typename RealIterator> class EnumeratorIterator {
public:
  using value_type = EnumeratorValue<RealIterator>;
  using reference = EnumeratorValue<RealIterator>&;
  using const_reference = const EnumeratorValue<RealIterator>&;
  using pointer = EnumeratorValue<RealIterator>*;

  using difference_type = std::ptrdiff_t;

  using self_type = EnumeratorIterator<RealIterator>;

  EnumeratorIterator(RealIterator it) : mEnum(EnumeratorValue(it, 0)) {}
  EnumeratorIterator(RealIterator it, size_t idx)
      : mEnum(EnumeratorValue(it, idx)) {}

  self_type operator++() {
    self_type copy = *this;
    copy.mEnum.mIterator++;
    copy.mEnum.mIndex++;

    return copy;
  }

  self_type operator++(int) {
    mEnum.mIterator++;
    mEnum.mIndex++;
  }

  reference operator*() { return mEnum; }

  pointer operator->() { return &mEnum; }

  bool operator==(const self_type& rhs) {
    return mEnum.mIterator == rhs.mEnum.mIterator;
  }
  bool operator!=(const self_type& rhs) {
    return mEnum.mIterator != rhs.mEnum.mIterator;
  }

private:
  EnumeratorValue<RealIterator> mEnum;
};

template <typename T> class Enumerator {
public:
  using iterator = EnumeratorIterator<T>;

  template <typename Container>
  Enumerator(Container& cont)
      : mBegin(EnumeratorIterator<T>(cont.begin(), 0)),
        mEnd(EnumeratorIterator<T>(cont.end(), std::distance(cont.begin(), cont.end()))){};

  iterator begin() { return mBegin; }
  iterator end() { return mEnd; }

private:
  EnumeratorIterator<T> mBegin;
  EnumeratorIterator<T> mEnd;
};
} // namespace detail

template <typename Container> auto enumerate(Container& cont) {
  return detail::Enumerator<typename Container::const_iterator>(cont);
}

} // namespace athena::utils

#endif // ATHENA_STLEXTRAS_H
