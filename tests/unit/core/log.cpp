/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "athena/core/log.h"

#include <athena/core/TensorShape.h>

#include <gtest/gtest.h>
#include <string>

namespace athena::core {

class LogTest : public ::testing::Test {
protected:
  std::streambuf* sbuf;
  std::stringstream buffer;

  void SetUp() override {
    sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
  }

  void TearDown() override { std::cout.rdbuf(sbuf); }
};

TEST_F(LogTest, CanLogToCout) {
  setLogStream<Logger>(std::cout);
  log() << "Test phrase";
  EXPECT_EQ(this->buffer.str(), "Test phrase");
}

TEST_F(LogTest, CanErrToCout) {
  setErrStream<Logger>(std::cout);
  log() << "Test phrase";
  EXPECT_EQ(this->buffer.str(), "Test phrase");
}

TEST_F(LogTest, CanSetLogtream) {
  std::stringstream ss;
  setLogStream<Logger>(ss);
  std::string firstPart = "Hello", secondPart = "Log";
  std::string fullString = firstPart + secondPart;
  log() << firstPart << secondPart;
  EXPECT_EQ(ss.str(), fullString);
}

TEST_F(LogTest, CanSetErrtream) {
  std::stringstream ss;
  setErrStream<Logger>(ss);
  std::string firstPart = "Hello", secondPart = "Log";
  std::string fullString = firstPart + secondPart;
  err() << firstPart << secondPart;
  EXPECT_EQ(ss.str(), fullString);
}
} // namespace athena::core