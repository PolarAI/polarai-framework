/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <boost/filesystem.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <options.h>
#include <yaml-cpp/yaml.h>

using path = boost::filesystem::path;

bool isCi() {
    return std::getenv("ATHENA_TEST_ENVIRONMENT") &&
           strncmp(std::getenv("ATHENA_TEST_ENVIRONMENT"), "ci", 2) == 0;
}

std::string replaceAll(std::string where, std::string what, std::string with) {
    size_t index = 0;
    while (true) {
        index = where.find(what, index);
        if (index == std::string::npos) break;

        where.replace(index, what.size(), with);

        index += with.size();
    }

    return where;
}

std::string expandMacro(std::string str) {
    std::string res = replaceAll(str, "@CMAKE_BINARY_DIR@", CMAKE_BINARY_DIR);

    if (isCi()) {
        res = replaceAll(str, "@ATHENA_BINARY_DIR@", CMAKE_BINARY_DIR);
    }

    return res;
}

void processSetNode(YAML::Node node) {
    if (node["name"]) {
        auto name = node["name"].as<std::string>();
        if (node["value"]) {
            std::string value = "";
            if (node["value"].IsMap() && isCi() &&
                node["value"]["ci"]) {  // todo improve bool handling
                value = node["value"]["ci"].as<std::string>();
            } else if (node["value"].IsMap() && node["value"]["dev"]) {
                value = node["value"]["dev"].as<std::string>();
            } else {
                value = node["value"].as<std::string>();
            }
            value = expandMacro(value);
            setenv(name.c_str(), value.c_str(), 1);
        } else {
            std::cerr << "Wrong set command syntax\n";
            std::cerr << "No value defined\n";
            exit(-1);
        }
    } else {
        std::cerr << "Wrong set command syntax\n";
        std::cerr << "No name defined\n";
        exit(-1);
    }
}

void parseCommandQueue(YAML::Node &commands) {
    for (auto command : commands) {
        if (command["set"]) {
            processSetNode(command["set"]);
        }
    }
}

void parseConfig(std::string configFile) {
    YAML::Node config = YAML::LoadFile(configFile);

    if (config["commands"]["common"]) {
        auto commonCommands = config["commands"]["common"];
        parseCommandQueue(commonCommands);
    }

#ifdef __APPLE__
    if (config["commands"]["macos"]) {
        auto macosCommands = config["commands"]["macos"];
        parseCommandQueue(macosCommands);
    }
#endif

#ifdef __linux__
    if (config["commands"]["linux"]) {
        auto linuxCommands = config["commands"]["linux"];
        parseCommandQueue(linuxCommands);
    }
#endif
}

int main(int argc, char **argv) {
    std::string exeName = path(argv[0]).filename().string();
    parseConfig(exeName + ".yml");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}