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

#include <cblas.h>

#include <gtest/gtest.h>

TEST(PicomathBlas, sgemm_square) {
  std::vector<float> a{1, 2, 3, 4};
  std::vector<float> b{1, 2, 3, 4};
  std::vector<float> c;
  std::vector<float> cCorrect{7, 10, 15, 22};

  c.resize(4);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1, a.data(),
              2, b.data(), 2, 0, c.data(), 2);

  for (int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(c[i], cCorrect[i]);
  }
}

TEST(PicomathBlas, sgemm_rect) {
  std::vector<float> a{1, 1, 2, 2, 3, 3, 4, 4};
  std::vector<float> b{1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<float> cRes;
  cRes.resize(16);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 2, 1, a.data(),
              4, b.data(), 2, 0, cRes.data(), 4);

  EXPECT_FLOAT_EQ(cRes[0], 2);
  EXPECT_FLOAT_EQ(cRes[1], 4);
  EXPECT_FLOAT_EQ(cRes[2], 6);
  EXPECT_FLOAT_EQ(cRes[3], 8);
  EXPECT_FLOAT_EQ(cRes[4], 4);
  EXPECT_FLOAT_EQ(cRes[5], 8);
  EXPECT_FLOAT_EQ(cRes[6], 12);
  EXPECT_FLOAT_EQ(cRes[7], 16);
  EXPECT_FLOAT_EQ(cRes[8], 6);
  EXPECT_FLOAT_EQ(cRes[9], 12);
  EXPECT_FLOAT_EQ(cRes[10], 18);
  EXPECT_FLOAT_EQ(cRes[11], 24);
  EXPECT_FLOAT_EQ(cRes[12], 8);
  EXPECT_FLOAT_EQ(cRes[13], 16);
  EXPECT_FLOAT_EQ(cRes[14], 24);
  EXPECT_FLOAT_EQ(cRes[15], 32);
}