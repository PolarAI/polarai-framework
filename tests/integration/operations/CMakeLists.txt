include_directories(${PROJECT_SOURCE_DIR}/src/backend/generic)

set(INT_TEST_SRC 
  AddOperation.cpp
  Conv2DOperation.cpp
  CombineOperation.cpp
  DivideOperation.cpp
  LogLossOperation.cpp
  MatMulOperation.cpp
  Pool2DOperation.cpp
  ReLUOperation.cpp
  SigmoidOperation.cpp
  SoftmaxOperation.cpp
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
