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

#include <effcee/effcee.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
  std::string content;
  std::stringstream contentStream;
  while (std::getline(std::cin, content)) {
    contentStream << content << "\n";
  }

  content = contentStream.str();

  std::ifstream checksFile(argv[1]);
  std::string checks((std::istreambuf_iterator<char>(checksFile)),
                     std::istreambuf_iterator<char>());

  auto result =
      effcee::Match(content, checks, effcee::Options().SetChecksName("checks"));

  if (!result) {
    switch (result.status()) {
    case effcee::Result::Status::NoRules:
      std::cout << "error: Expected check rules\n";
      break;
    case effcee::Result::Status::Fail:
      std::cout << "The input failed to match check rules:\n";
      break;
    default:
      break;
    }
    std::cout << result.message() << std::endl;
    return -1;
  }

  return 0;
}