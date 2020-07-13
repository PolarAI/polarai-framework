include_directories(${PROJECT_SOURCE_DIR}/src/backend/generic)

set(INT_TEST_SRC 
  AddOperation.cpp
  CombineOperation.cpp
  DivideOperation.cpp
  LogLossOperation.cpp
  MatMulOperation.cpp
  SigmoidOperation.cpp
  MulConcatOperation.cpp
  MulOperation.cpp)

add_polar_executable(
        TestIntegrationOperationsRunnable
        ${INT_TEST_SRC})

find_package(Threads)
target_link_libraries(TestIntegrationOperationsRunnable PRIVATE
        PolarAI::framework
        PolarAI::generic_backend
        PolarAI::generic_backend_driver
        AthenaDep::googletest
        Threads::Threads)

add_test(NAME OperationsIntegrationTest COMMAND TestIntegrationOperationsRunnable)
